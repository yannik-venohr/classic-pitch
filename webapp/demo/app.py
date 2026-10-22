"""Local web demo for ClassicPitch: search YouTube, transcribe, view + listen.

Run with:
    python webapp/demo/app.py
then open http://127.0.0.1:5001

Everything that isn't YouTube-specific lives in webapp/shared.
"""
import sys
import os
import re
import hashlib
import json
import shutil
import subprocess
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pathlib import Path

from flask import request, jsonify
import yt_dlp

from classic_pitch import DEFAULT_VARIANT, available_variants, variant_label
from webapp.shared import decode
from webapp.shared.predict import get_classic_pitch
from webapp.shared.server import PROJECT_ROOT, cached_meta, create_app, write_meta
from webapp.shared.synth import audio_duration

APP_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "data" / "webapp_demo_cache"

# Only the start of a recording is transcribed: the stored recording is
# cut to this length, so the original, the transcription and the rolls
# all share one timeline.
MAX_TRANSCRIBE_SECONDS = 10 * 60

# A recording is stored once, under its "base" id (a YouTube video id, or
# "local_<hash>" for an uploaded file), in CACHE_DIR/<base>/audio.wav.
# What a checkpoint made of it is cached beside that under
# "<base>~<variant>", the same keying the rubato app uses, so two models
# on the same recording never overwrite each other.
BASE_ID = r"[\w-]{6,40}"
VARIANT_ID = r"[A-Za-z0-9][A-Za-z0-9_]{1,63}"
ITEM_ID_RE = re.compile(rf"^({BASE_ID})~({VARIANT_ID})$")
# What /media guards with: a recording's directory or a transcription's.
MEDIA_ID_RE = re.compile(rf"^{BASE_ID}(?:~{VARIANT_ID})?$")

VARIANTS = available_variants()
DEFAULT_CHECKPOINT = DEFAULT_VARIANT if DEFAULT_VARIANT in VARIANTS else (VARIANTS[0] if VARIANTS else "")

# YouTube's signature challenge needs a JS runtime; yt-dlp only enables `deno`
# by default, so opt into `node` too (widely available, e.g. via conda/npm).
YDL_JS_RUNTIME_OPTS = {"js_runtimes": {"node": {}}}

SOURCE_INFO = "source.json"  # title etc. of a stored recording, next to its audio.wav


def split_cache_id(cid):
    base, _, variant = cid.partition("~")
    return base, variant


def is_upload(base):
    # YouTube ids are always 11 characters, so this can't collide with one.
    return len(base) == 18 and base.startswith("local_")


def cached_audio_path(cid):
    return CACHE_DIR / split_cache_id(cid)[0] / "audio.wav"


# Passing source_audio turns on /api/decode, which the "Inside the model"
# panel drives: the downloaded recording is what a re-decoded MIDI track
# has to be rendered against, and for this app it lives in the cache.
app = create_app(__name__, APP_ROOT / "static", CACHE_DIR, MEDIA_ID_RE,
                 source_audio=cached_audio_path)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024


def length_note(source_duration):
    """Say so when only the start of a recording is transcribed."""
    if source_duration and source_duration > MAX_TRANSCRIBE_SECONDS + 1:
        return (f"{format_clock(source_duration)} long · only the first "
                f"{format_clock(MAX_TRANSCRIBE_SECONDS)} is transcribed")
    return None


def format_clock(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def trim_to_wav(src, dst, seconds=MAX_TRANSCRIBE_SECONDS):
    """Decode any audio/video file to a wav of at most ``seconds``."""
    tmp = Path(f"{dst}.tmp.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-t", str(seconds),
             "-c:a", "pcm_s16le", str(tmp)],
            check=True, capture_output=True,
        )
        os.replace(tmp, dst)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(e.stderr.decode(errors="replace").strip() or "ffmpeg failed")
    finally:
        tmp.unlink(missing_ok=True)


def probe_duration(path):
    """Full length in seconds of a media file, before any trimming."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return None


def read_source_info(base):
    """What is known about a stored recording, or None."""
    base_dir = CACHE_DIR / base
    path = base_dir / SOURCE_INFO
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    # Recordings stored before checkpoints were cached separately have
    # their title in the old combined meta.json instead.
    legacy = cached_meta(app, base)
    if legacy is not None:
        return {
            "title": legacy.get("title", base),
            "thumbnail": legacy.get("thumbnail"),
            "source_url": legacy.get("source_url"),
            "source_duration": legacy.get("duration"),
        }
    return None


def write_source_info(base, info):
    (CACHE_DIR / base / SOURCE_INFO).write_text(json.dumps(info))


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
        "transcribed_seconds": min(info.get("duration") or 0, MAX_TRANSCRIBE_SECONDS),
        "thumbnail": info.get("thumbnail"),
        "uploader": info.get("uploader"),
    }


def download_audio(video_id, audio_path: Path):
    """Download a video's audio as a wav of at most MAX_TRANSCRIBE_SECONDS."""
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
    # Asking the downloader for just the start saves fetching a long
    # video whole, but ffmpeg's ranged fetch is not reliable on every
    # stream — so fall back to the full file, trimmed below.
    ranged = {**ydl_opts, "download_ranges": yt_dlp.utils.download_range_func(
        None, [(0, MAX_TRANSCRIBE_SECONDS)])}
    try:
        with yt_dlp.YoutubeDL(ranged) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"[demo] ranged download failed ({e}); fetching the whole video")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    if audio_duration(audio_path) > MAX_TRANSCRIBE_SECONDS + 1:
        trim_to_wav(audio_path, audio_path)


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
            "transcribed_seconds": min(e.get("duration") or 0, MAX_TRANSCRIBE_SECONDS),
            "thumbnail": e.get("thumbnail") or (thumbs[-1]["url"] if thumbs else None),
            "uploader": e.get("uploader") or e.get("channel"),
        })
    return jsonify({"results": results})


@app.route("/api/models")
def api_models():
    return jsonify({
        "checkpoints": [{"id": v, "label": variant_label(v)} for v in VARIANTS],
        "default_checkpoint": DEFAULT_CHECKPOINT,
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Store a local audio file as a recording, ready to be transcribed."""
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"error": "no file"}), 400

    tmp_dir = CACHE_DIR / f".upload-{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True)
    try:
        raw = tmp_dir / "upload"
        digest = hashlib.sha1()
        with open(raw, "wb") as out:
            while chunk := file.stream.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
        # The same file again lands on the same id, and so on the cache.
        base = f"local_{digest.hexdigest()[:12]}"
        base_dir = CACHE_DIR / base
        if (base_dir / "audio.wav").exists() and read_source_info(base) is not None:
            return jsonify({"id": base, "title": read_source_info(base)["title"]})

        full_duration = probe_duration(raw)
        base_dir.mkdir(parents=True, exist_ok=True)
        try:
            trim_to_wav(raw, base_dir / "audio.wav")
        except RuntimeError as e:
            shutil.rmtree(base_dir, ignore_errors=True)
            return jsonify({"error": f"could not read that file as audio: {e}"}), 400
        title = Path(file.filename).stem or base
        write_source_info(base, {"title": title, "source_duration": full_duration})
        return jsonify({"id": base, "title": title})
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    data = request.get_json(force=True, silent=True) or {}
    match = ITEM_ID_RE.match(str(data.get("item_id", "")).strip())
    if match is None:
        return jsonify({"error": "invalid item_id"}), 400
    cid, base, variant = match.group(0), match.group(1), match.group(2)
    if variant not in VARIANTS:
        return jsonify({"error": f"unknown checkpoint '{variant}'"}), 400

    item_dir = CACHE_DIR / cid
    audio_path = cached_audio_path(cid)

    # A cached item whose posteriograms predate this feature has to go
    # back through the model once — there is no way to recover the raw
    # output from the note list, and without it the model panel's
    # threshold controls have nothing to re-decode.
    cached = cached_meta(app, cid)
    if cached is not None and decode.has_posteriograms(item_dir):
        # Cheap, and usually a no-op: only redraws the posteriogram PNGs
        # when their tone curve has changed since they were written.
        refreshed = decode.refresh_posteriogram_images(item_dir, cid, cached)
        if refreshed is not None:
            cached = write_meta(app, cid, {**cached, **refreshed})
        return jsonify(cached)

    # The recording is fetched once per video and reused by every
    # checkpoint; YouTube isn't touched at all when it is already stored
    # — and it may well no longer be reachable, which is exactly when a
    # cached copy matters.
    info = read_source_info(base) if audio_path.exists() else None
    if info is None:
        if is_upload(base):
            return jsonify({"error": "that uploaded file is no longer cached — choose it again"}), 404
        try:
            video = fetch_video_info(base)
        except Exception as e:
            return jsonify({"error": f"could not look up video: {e}"}), 400

        # Only create the cache directory once we know the video is real —
        # otherwise a failed lookup litters an empty folder that then
        # needs manual cleanup.
        (CACHE_DIR / base).mkdir(parents=True, exist_ok=True)
        try:
            download_audio(base, audio_path)
        except Exception as e:
            return jsonify({"error": f"download failed: {e}"}), 500
        info = {
            "title": video.get("title", base),
            "thumbnail": video.get("thumbnail"),
            "source_url": f"https://www.youtube.com/watch?v={base}",
            "source_duration": video.get("duration"),
        }
        write_source_info(base, info)
    else:
        print(f"[demo] transcribing stored recording {base} with {variant}")

    item_dir.mkdir(parents=True, exist_ok=True)
    try:
        _, outputs, times = get_classic_pitch(variant).predict(
            str(audio_path), verbose=True, max_seconds=MAX_TRANSCRIBE_SECONDS)
    except Exception as e:
        return jsonify({"error": f"transcription failed: {e}"}), 500

    note = length_note(info.get("source_duration"))
    meta = {
        "item_id": cid,
        "title": info.get("title", base),
        "subtitle": " · ".join(x for x in (variant_label(variant), note) if x),
        "thumbnail": info.get("thumbnail"),
        "source_url": info.get("source_url"),
        "duration": audio_duration(audio_path),
        "audio_url": f"/media/{base}/audio.wav",
    }
    meta.update(decode.save_posteriograms(item_dir, cid, outputs, times))
    # The midi predict() already decoded is discarded and decoded again
    # here, so the notes, the rendered wav and the "decoding" the UI
    # shows all come from the one code path /api/decode also uses.
    meta.update(decode.apply_decoding(
        item_dir, cid, outputs, times, decode.DEFAULT_DECODING, audio_path))

    return jsonify(write_meta(app, cid, meta))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
