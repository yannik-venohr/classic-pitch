"""Local web demo for ClassicPitch: search YouTube, transcribe, view + listen.

Run with:
    python webapp/demo/app.py
then open http://127.0.0.1:5001

Everything that isn't YouTube-specific lives in webapp/shared.
"""
import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pathlib import Path

from flask import request, jsonify
import yt_dlp

from webapp.shared import decode
from webapp.shared.predict import get_predictor
from webapp.shared.server import PROJECT_ROOT, cached_meta, create_app, write_meta
from webapp.shared.synth import audio_duration

import pathlib
import platform

if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

APP_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "data" / "webapp_demo_cache"

MAX_DURATION_SECONDS = 20 * 60
VIDEO_ID_RE = re.compile(r"^[\w-]{6,20}$")

# YouTube's signature challenge needs a JS runtime; yt-dlp only enables `deno`
# by default, so opt into `node` too (widely available, e.g. via conda/npm).
YDL_JS_RUNTIME_OPTS = {"js_runtimes": {"node": {}}}


def cached_audio_path(video_id):
    return CACHE_DIR / video_id / "audio.wav"


# Passing source_audio turns on /api/decode, which the "Inside the model"
# panel drives: the downloaded recording is what a re-decoded MIDI track
# has to be rendered against, and for this app it lives in the cache.
app = create_app(__name__, APP_ROOT / "static", CACHE_DIR, VIDEO_ID_RE,
                 source_audio=cached_audio_path)


def is_probably_url(text):
    return bool(re.match(r"^https?://", text.strip(), re.IGNORECASE))


def extract_video_id(url):
    m = re.search(r"(?:v=|/videos/|embed/|youtu\.be/|/shorts/|/live/)([\w-]{11})", url)
    return m.group(1) if m else None


def fetch_video_info(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {"quiet": True, "noplaylist": True, **YDL_JS_RUNTIME_OPTS}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "id": info.get("id", video_id),
        "title": info.get("title", video_id),
        "duration": info.get("duration"),
        "thumbnail": info.get("thumbnail"),
        "uploader": info.get("uploader"),
    }


def download_audio(video_id, audio_path: Path):
    url = f"https://www.youtube.com/watch?v={video_id}"
    outtmpl = str(audio_path.with_suffix(""))
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "noplaylist": True,
        "quiet": True,
        "outtmpl": outtmpl,
        **YDL_JS_RUNTIME_OPTS,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


@app.route("/api/search")
def api_search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "empty query"}), 400

    if is_probably_url(query):
        video_id = extract_video_id(query)
        if video_id is None:
            return jsonify({"error": "could not parse a video id from that url"}), 400
        try:
            info = fetch_video_info(video_id)
        except Exception as e:
            return jsonify({"error": f"could not look up video: {e}"}), 400
        return jsonify({"results": [info]})

    ydl_opts = {
        "quiet": True,
        "noplaylist": True,
        "extract_flat": "in_playlist",
        **YDL_JS_RUNTIME_OPTS,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(f"ytsearch8:{query}", download=False)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    results = []
    for e in (result.get("entries") or []):
        if not e:
            continue
        thumbs = e.get("thumbnails") or []
        results.append({
            "id": e.get("id"),
            "title": e.get("title"),
            "duration": e.get("duration"),
            "thumbnail": e.get("thumbnail") or (thumbs[-1]["url"] if thumbs else None),
            "uploader": e.get("uploader") or e.get("channel"),
        })
    return jsonify({"results": results})


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    data = request.get_json(force=True, silent=True) or {}
    video_id = str(data.get("item_id", "")).strip()
    if not VIDEO_ID_RE.match(video_id):
        return jsonify({"error": "invalid item_id"}), 400

    video_dir = CACHE_DIR / video_id
    audio_path = cached_audio_path(video_id)

    # A cached item whose posteriograms predate this feature has to go
    # back through the model once — there is no way to recover the raw
    # output from the note list, and without it the model panel's
    # threshold controls have nothing to re-decode.
    cached = cached_meta(app, video_id)
    if cached is not None and decode.has_posteriograms(video_dir):
        # Cheap, and usually a no-op: only redraws the posteriogram PNGs
        # when their tone curve has changed since they were written.
        refreshed = decode.refresh_posteriogram_images(video_dir, video_id, cached)
        if refreshed is not None:
            cached = write_meta(app, video_id, {**cached, **refreshed})
        return jsonify(cached)

    if cached is not None and audio_path.exists():
        # Re-running such an item: the recording is already downloaded, so
        # YouTube isn't touched at all — and it may well no longer be
        # reachable, which is exactly when a cached copy matters.
        print(f"[demo] re-running {video_id} to store its posteriograms")
        info = {"title": cached.get("title", video_id), "thumbnail": cached.get("thumbnail")}
    else:
        try:
            info = fetch_video_info(video_id)
        except Exception as e:
            return jsonify({"error": f"could not look up video: {e}"}), 400

        duration = info.get("duration") or 0
        if duration and duration > MAX_DURATION_SECONDS:
            return jsonify({
                "error": f"video is {duration / 60:.1f} min long; this demo caps at "
                         f"{MAX_DURATION_SECONDS // 60} min"
            }), 400

        # Only create the cache directory once we know the video is real and
        # short enough to process — otherwise a failed lookup litters an empty,
        # meta.json-less folder that then needs manual cleanup.
        video_dir.mkdir(parents=True, exist_ok=True)

        try:
            download_audio(video_id, audio_path)
        except Exception as e:
            return jsonify({"error": f"download failed: {e}"}), 500

    try:
        _, outputs, times = get_predictor().predict(str(audio_path), verbose=True)
    except Exception as e:
        return jsonify({"error": f"transcription failed: {e}"}), 500

    meta = {
        "item_id": video_id,
        "title": info.get("title", video_id),
        "thumbnail": info.get("thumbnail"),
        "source_url": f"https://www.youtube.com/watch?v={video_id}",
        "duration": audio_duration(audio_path),
        "audio_url": f"/media/{video_id}/audio.wav",
    }
    meta.update(decode.save_posteriograms(video_dir, video_id, outputs, times))
    # The midi predict() already decoded is discarded and decoded again
    # here, so the notes, the rendered wav and the "decoding" the UI
    # shows all come from the one code path /api/decode also uses.
    meta.update(decode.apply_decoding(
        video_dir, video_id, outputs, times, decode.DEFAULT_DECODING, audio_path))

    return jsonify(write_meta(app, video_id, meta))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
