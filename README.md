# nosleep-audio

Turns r/nosleep posts into narrated audiobooks for personal listening.

- Fetches the post through Reddit's public RSS. There's no API key, and the limit is about 1 request a minute.
- Narrates it on CPU with [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M).
- Writes AAC `.m4b` (audiobook) and `.m4a` (podcast) files with chapters, a generated cover and tags.
- Publishes the `.m4a` files to a private podcast feed on Cloudflare R2.

```sh
python nosleep.py build <reddit post url> [--voice bm_george]
python nosleep.py publish
```

Needs `ffmpeg` and a venv with `kokoro`, `soundfile` and `numpy`. `publish` reads `cloudflare_token`,
`cloudflare_account_id` and `nosleep_r2` (`bucket`, `public_base`, `prefix`) from `~/.credentials/api-keys.json`.

The stories belong to their authors. This tool is for private listening only, so don't publish its output.
