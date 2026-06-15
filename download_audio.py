#!/usr/bin/env python3
"""Download audio from a Bilibili video using bilibili-api-python."""

import sys
import asyncio
from pathlib import Path

from bilibili_api import video, Credential
from bilibili_api.utils.network import get_session

OUTPUT_DIR = Path("/home/huyue/projects/talk-extract")

async def main(bvid: str):
    # No credential needed for public videos
    v = video.Video(bvid=bvid)

    info = await v.get_info()
    title = info["title"]
    duration = info["duration"]
    pages = info["pages"]
    print(f"Title: {title}")
    print(f"Duration: {duration}s ({duration // 60}min {duration % 60}s)")
    print(f"Pages: {len(pages)}")

    # Get download URL (first page)
    cid = pages[0]["cid"]
    print(f"CID: {cid}")

    download_url = await v.get_download_url(cid=cid)
    detecter = video.VideoDownloadURLDataDetecter(data=download_url)

    # Find audio-only streams from DASH data
    dash = download_url.get("dash", {})
    audios = dash.get("audio", [])
    print(f"Audio streams available: {len(audios)}")

    if not audios:
        print("No audio streams in DASH. Trying detect_best_streams...")
        # Fallback: try to get streams without video
        streams = detecter.detect_best_streams(
            video_max_quality=video.VideoQuality._360P,
            audio_max_quality=video.AudioQuality._192K,
        )
        if streams:
            for s in streams:
                print(f"  Stream: {s.quality_desc} ({s.width}x{s.height}) url_present: {bool(s.url)}")

    # Pick best audio
    best_audio = None
    best_quality = -1
    for a in audios:
        q = a.get("id", 0)
        if q > best_quality:
            best_quality = q
            best_audio = a

    if not best_audio:
        print("Could not find an audio stream.")
        return

    print(f"Selected audio: id={best_audio.get('id')}, "
          f"codec={best_audio.get('codecs', '?')}, "
          f"bitrate={best_audio.get('bandwidth', 0)//1000}kbps")

    audio_url = best_audio.get("base_url") or best_audio.get("baseUrl", "")
    # Fix URL (may need host)
    if audio_url.startswith("//"):
        audio_url = "https:" + audio_url
    print(f"Audio URL: {audio_url[:120]}...")

    # Download
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)
    out_path = OUTPUT_DIR / f"{safe_title}.m4a"
    print(f"\nDownloading to: {out_path}")

    session = get_session()
    async with session.get(audio_url, headers={
        "Referer": "https://www.bilibili.com/",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    }) as resp:
        if resp.status != 200:
            print(f"Download failed: HTTP {resp.status}")
            return
        content = await resp.read()

    out_path.write_bytes(content)
    size_mb = len(content) / (1024 * 1024)
    print(f"Downloaded: {size_mb:.1f} MB -> {out_path}")
    print(f"Done! Audio saved to: {out_path}")

if __name__ == "__main__":
    bvid = sys.argv[1] if len(sys.argv) > 1 else "BV1nioTBcEvj"
    asyncio.run(main(bvid))
