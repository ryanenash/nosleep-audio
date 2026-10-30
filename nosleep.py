#!/usr/bin/env python3
"""Turn an r/nosleep post into a narrated .m4b (iPod) and .m4a (podcast feed).

Usage: nosleep.py build <post url> [--voice bm_george]
"""
import argparse
import html
import json
import re
import subprocess
import sys
import textwrap
import time
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
UA = "linux:nosleep-audio:0.1 (personal audiobook builder; +https://github.com/ryanenash/nosleep-audio)"
ATOM = {"a": "http://www.w3.org/2005/Atom"}
SR = 24000  # Kokoro output rate
CHAPTER_WORDS = 750  # ~5 min of narration
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def fetch_post(url):
    """First entry of the post's Atom feed. RSS works anonymously from this box; .json and jina get 403."""
    m = re.search(r"/comments/([a-z0-9]+)", url)
    if not m:
        sys.exit(f"not a reddit post url: {url}")
    feed = f"https://www.reddit.com/r/nosleep/comments/{m.group(1)}/.rss"
    for _ in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(feed, headers={"User-Agent": UA}), timeout=30) as r:
                root = ET.fromstring(r.read())
            break
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            # anonymous limit is ~1 request/minute
            time.sleep(int(float(e.headers.get("x-ratelimit-reset", 60))) + 2)
    else:
        sys.exit("rate limited by reddit")
    e = root.find("a:entry", ATOM)
    return {
        "id": m.group(1),
        "title": html.unescape(e.find("a:title", ATOM).text),
        "author": e.find("a:author/a:name", ATOM).text.removeprefix("/u/"),
        "url": e.find("a:link", ATOM).get("href"),
        "published": e.find("a:published", ATOM).text,
        "html": e.find("a:content", ATOM).text,
    }


class Blocks(HTMLParser):
    """Collect paragraph text from the selftext div; a horizontal rule marks a section break."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks, self.buf, self.depth, self.done = [], [], 0, False

    def flush(self):
        t = " ".join("".join(self.buf).split())
        if t:
            self.blocks.append(t)
        self.buf = []

    def handle_starttag(self, tag, attrs):
        if self.done:
            return
        if tag == "div" and ("class", "md") in attrs:
            self.depth = 1
        elif self.depth and tag == "div":
            self.depth += 1
        elif self.depth and tag in ("p", "li", "h1", "h2", "h3", "blockquote", "br"):
            self.flush()
        elif self.depth and tag == "hr":
            self.flush()
            self.blocks.append(None)

    def handle_endtag(self, tag):
        if self.depth and tag == "div":
            self.depth -= 1
            if not self.depth:
                self.flush()
                self.done = True
        elif self.depth and tag in ("p", "li", "h1", "h2", "h3", "blockquote"):
            self.flush()

    def handle_data(self, data):
        if self.depth:
            self.buf.append(data)


EDIT_RE = re.compile(r"^\W*(edit|update)\b[\s\d#]*[:\-–—]", re.I)
AWARD_RE = re.compile(r"thank(s| you).{0,60}\b(gold|silver|platinum|awards?)\b", re.I)


def clean(t):
    t = re.sub(r"https?://\S+|www\.\S+", "", t)
    t = re.sub(r"[*_~#>`]+", "", t)  # stray markdown
    t = re.sub(r"\[([^\]]*)\]", r"\1", t)  # "[Redacted]" -> "Redacted"
    return " ".join(t.split())


def spoken_name(author):
    # "HorrorJunkie123" reads better as "Horror Junkie 123"
    return re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z])(?=\d)|_", " ", author)


def script(post):
    """Return chapters as [(name, [paragraph, ...])]: title line, body by section/length, edits last."""
    p = Blocks()
    p.feed(post["html"])
    body, edits = [], []
    for b in p.blocks:
        if b is None:
            body.append(None)
            continue
        b = clean(b)
        if not b or AWARD_RE.search(b):
            continue
        (edits if EDIT_RE.match(b) else body).append(b)

    # Split on horizontal rules if the author used them; otherwise every ~CHAPTER_WORDS words.
    sections, cur = [], []
    for b in body:
        if b is None:
            if cur:
                sections.append(cur)
            cur = []
        else:
            cur.append(b)
    if cur:
        sections.append(cur)
    if len(sections) == 1:
        paras, sections, cur, n = sections[0], [], [], 0
        total = sum(len(x.split()) for x in paras)
        target = total / max(1, round(total / CHAPTER_WORDS))
        for x in paras:
            cur.append(x)
            n += len(x.split())
            if n >= target * (len(sections) + 1) and len(sections) < round(total / target) - 1:
                sections.append(cur)
                cur = []
        if cur:
            sections.append(cur)

    intro = f"{post['title']}, by u slash {spoken_name(post['author'])}."
    chapters = [(f"Part {i + 1}", s) for i, s in enumerate(sections)]
    chapters[0] = (chapters[0][0], [intro] + chapters[0][1])
    if edits:
        chapters.append(("Author's notes", edits))
    return chapters


def narrate(chapters, voice, wav):
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    pipe = KPipeline(lang_code=voice[0], repo_id="hexgrad/Kokoro-82M")
    gap, chap_gap = np.zeros(int(SR * 0.45), "float32"), np.zeros(int(SR * 1.2), "float32")
    audio, starts, pos = [], [], 0
    for ci, (_, paras) in enumerate(chapters):
        if ci:
            audio.append(chap_gap)
            pos += len(chap_gap)
        starts.append(pos)
        for i, para in enumerate(paras):
            for r in pipe(para, voice=voice, split_pattern=None):
                a = r.audio.numpy()
                audio.append(a)
                pos += len(a)
            audio.append(gap)
            pos += len(gap)
        print(f"  chapter {ci + 1}/{len(chapters)} done, {pos / SR / 60:.1f} min", flush=True)
    sf.write(wav, np.concatenate(audio), SR)
    return [s / SR for s in starts], pos / SR


def cover(post, path):
    title = "\n".join(textwrap.wrap(post["title"], 22)[:9])
    txt = path.with_suffix(".txt")
    txt.write_text(title)
    by = path.with_suffix(".by.txt")
    by.write_text(f"u/{post['author']}")
    vf = (
        f"drawtext=fontfile={FONT}:textfile={txt}:fontcolor=white:fontsize=78:line_spacing=18:x=110:y=(h-text_h)/2-60,"
        f"drawtext=fontfile={FONT}:text='r/nosleep':fontcolor=0xb22222:fontsize=64:x=110:y=110,"
        f"drawtext=fontfile={FONT}:textfile={by}:fontcolor=0xaaaaaa:fontsize=54:x=110:y=h-170"
    )
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=0x121212:s=1400x1400",
                    "-vf", vf, "-frames:v", "1", "-q:v", "2", str(path)], check=True)
    txt.unlink()
    by.unlink()


def encode(post, chapters, starts, duration, wav, art, dest, series=None):
    meta = [";FFMETADATA1"]
    ends = starts[1:] + [duration]
    for (name, _), s, e in zip(chapters, starts, ends):
        meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(s * 1000)}", f"END={int(e * 1000)}", f"title={name}"]
    mfile = dest.with_suffix(".chapters.txt")
    mfile.write_text("\n".join(meta) + "\n")
    tags = {
        "title": post["title"],
        "artist": f"u/{post['author']}",
        "album_artist": f"u/{post['author']}",
        "album": series["name"] if series else post["title"],
        "genre": "Horror",
        "date": post["published"][:10],
        "comment": post["url"],
        "media_type": "2",  # stik=Audiobook, so iTunes/Books files it under Audiobooks
    }
    if series:
        tags["track"] = str(series["part"])
        tags["show"] = series["name"]
    m4b = dest.with_suffix(".m4b")
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-i", str(mfile), "-i", str(art),
           "-map", "0:a", "-map", "2:v", "-map_metadata", "1", "-map_chapters", "1",
           "-c:a", "aac", "-profile:a", "aac_low", "-b:a", "64k", "-ac", "1", "-ar", "44100",
           "-c:v", "copy", "-disposition:v", "attached_pic", "-movflags", "+faststart", "-f", "ipod", "-brand", "M4B "]
    for k, v in tags.items():
        cmd += ["-metadata", f"{k}={v}"]
    subprocess.run(cmd + [str(m4b)], check=True)
    # Same audio, M4A brand for the podcast feed. Map only audio+art: the chapter text track is rebuilt from metadata.
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(m4b), "-map", "0:a", "-map", "0:v", "-c", "copy",
                    "-map_chapters", "0", "-metadata", "media_type=1", "-disposition:v", "attached_pic",
                    "-movflags", "+faststart", "-f", "ipod", str(dest.with_suffix(".m4a"))], check=True)
    mfile.unlink()


def build(args):
    post = fetch_post(args.url)
    d = OUT / post["id"]
    d.mkdir(parents=True, exist_ok=True)
    chapters = script(post)
    words = sum(len(p.split()) for _, ps in chapters for p in ps)
    print(f"{post['title']} by u/{post['author']}: {words} words, {len(chapters)} chapters", flush=True)
    wav = d / "narration.wav"
    t0 = time.time()
    starts, duration = narrate(chapters, args.voice, wav)
    print(f"narrated {duration / 60:.1f} min in {(time.time() - t0) / 60:.1f} min", flush=True)
    cover(post, d / "cover.jpg")
    encode(post, chapters, starts, duration, wav, d / "cover.jpg", d / "story")
    wav.unlink()
    (d / "meta.json").write_text(json.dumps({
        **{k: v for k, v in post.items() if k != "html"}, "voice": args.voice, "duration": duration,
        "chapters": [{"title": n, "start": s} for (n, _), s in zip(chapters, starts)],
    }, indent=2))
    print(d)


def r2():
    keys = json.loads(Path.home().joinpath(".credentials/api-keys.json").read_text())
    cfg = keys["nosleep_r2"]
    return {**cfg, "token": keys["cloudflare_token"],
            "api": f"https://api.cloudflare.com/client/v4/accounts/{keys['cloudflare_account_id']}/r2/buckets/{cfg['bucket']}/objects"}


def put(cfg, src, key, ctype):
    # Cloudflare REST API, not the S3 endpoint: <acct>.r2.cloudflarestorage.com fails the TLS handshake from this box.
    req = urllib.request.Request(f"{cfg['api']}/{cfg['prefix']}/{key}", data=Path(src).read_bytes(), method="PUT",
                                 headers={"Authorization": f"Bearer {cfg['token']}", "Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=300) as r:
        if not json.load(r)["success"]:
            sys.exit(f"upload failed: {key}")


def feed_xml(base, items):
    from email.utils import format_datetime
    from datetime import datetime
    from xml.sax.saxutils import escape as x

    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">', "<channel>",
           "<title>NoSleep Reader (private)</title>", f"<link>{base}/feed.xml</link>",
           "<description>Private narrations of r/nosleep stories. Personal use only.</description>",
           "<language>en</language>", "<itunes:block>Yes</itunes:block>", "<itunes:explicit>true</itunes:explicit>",
           f'<itunes:image href="{base}/show.jpg"/>', "<itunes:author>r/nosleep</itunes:author>"]
    for m in items:
        dur = int(m["duration"])
        out += ["<item>", f"<title>{x(m['title'])}</title>",
                f"<description>{x('by u/' + m['author'] + ' — ' + m['url'])}</description>",
                f'<enclosure url="{base}/{m["id"]}.m4a" length="{m["m4a_bytes"]}" type="audio/x-m4a"/>',
                f'<guid isPermaLink="false">nosleep-{m["id"]}</guid>',
                f"<pubDate>{format_datetime(datetime.fromisoformat(m['published']))}</pubDate>",
                f"<itunes:duration>{dur // 3600}:{dur // 60 % 60:02d}:{dur % 60:02d}</itunes:duration>",
                f'<itunes:image href="{base}/{m["id"]}.jpg"/>', "</item>"]
    return "\n".join(out + ["</channel>", "</rss>"]) + "\n"


def publish(args):
    cfg = r2()
    base = f"{cfg['public_base']}/{cfg['prefix']}"
    items = []
    for meta in sorted(OUT.glob("*/meta.json")):
        m, d = json.loads(meta.read_text()), meta.parent
        m["m4a_bytes"] = (d / "story.m4a").stat().st_size
        if not m.get("uploaded"):
            put(cfg, d / "story.m4a", f"{m['id']}.m4a", "audio/x-m4a")
            put(cfg, d / "cover.jpg", f"{m['id']}.jpg", "image/jpeg")
            m["uploaded"] = True
            meta.write_text(json.dumps(m, indent=2))
        items.append(m)
    items.sort(key=lambda m: m["published"], reverse=True)
    show = OUT / "show.jpg"
    if not show.exists():
        cover({"title": "NoSleep Reader", "author": "private"}, show)
        put(cfg, show, "show.jpg", "image/jpeg")
    f = OUT / "feed.xml"
    f.write_text(feed_xml(base, items))
    put(cfg, f, "feed.xml", "application/rss+xml")
    print(f"{base}/feed.xml")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    b = sub.add_parser("build")
    b.add_argument("url")
    b.add_argument("--voice", default="bm_george")
    b.set_defaults(fn=build)
    sub.add_parser("publish").set_defaults(fn=publish)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
