"""Flask plumbing shared by every ClassicPitch web app.

The apps differ only in where their audio comes from — YouTube, a local
dataset — so everything downstream of "we have a wav and a note list"
lives here: serving the cached media, listing what has already been
processed, and serving the shared frontend.
"""
import json
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, send_file, abort

from webapp.shared import decode

SHARED_STATIC = Path(__file__).resolve().parent / "static"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MEDIA_FILES = ("audio.wav", "midi.wav")


def create_app(import_name, static_dir, cache_dir, id_re,
               media_files=DEFAULT_MEDIA_FILES, source_audio=None):
    """Build a Flask app with the routes every ClassicPitch app needs.

    ``id_re`` is supplied by the caller rather than fixed here because it
    is what keeps ``/media`` from walking out of the cache directory, and
    the two apps' ids look nothing alike (an 11-char YouTube id vs. a
    ``Bach_BWV1007-01_OV-Carr2014``-style dataset filename).

    ``source_audio`` turns on the "Inside the model" panel's backend: pass
    an ``item_id -> Path`` callable and the app gains ``/api/decode``,
    which re-thresholds a stored posteriogram. It has to be supplied per
    app because re-rendering the MIDI track needs the *original*
    recording to match its length against, and only the app knows where
    that lives (the cache, for the YouTube demo).
    An app that leaves it out simply has no model panel.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if source_audio is not None:
        media_files = tuple(media_files) + decode.MEDIA_FILES

    app = Flask(import_name, static_folder=str(static_dir), static_url_path="")
    app.config["CACHE_DIR"] = cache_dir
    app.config["ID_RE"] = id_re

    @app.route("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    # The frontend (app.js, notation.js, style.css, ...) is identical
    # across apps and lives in shared/static; only index.html and
    # info-panel.html are per-app and served from the app's own static dir.
    @app.route("/shared/<path:filename>")
    def shared_static(filename):
        return send_from_directory(SHARED_STATIC, filename)

    @app.route("/api/library")
    def api_library():
        items = []
        for meta_path in cache_dir.glob("*/meta.json"):
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                continue
            items.append({
                "item_id": meta.get("item_id", meta_path.parent.name),
                "title": meta.get("title", meta_path.parent.name),
                "duration": meta.get("duration"),
                "mtime": meta_path.stat().st_mtime,
            })
        items.sort(key=lambda it: it["mtime"], reverse=True)
        for it in items:
            del it["mtime"]
        return jsonify({"items": items})

    @app.route("/media/<item_id>/<filename>")
    def media(item_id, filename):
        if not id_re.match(item_id) or filename not in media_files:
            abort(404)
        path = cache_dir / item_id / filename
        if not path.exists():
            abort(404)
        return send_file(path, conditional=True)

    # Re-run only the decoding step of an already-transcribed item, at
    # thresholds the user picked. Cheap enough (a second or two) to sit
    # behind live controls; the model is not touched.
    def api_decode():
        data = request.get_json(force=True, silent=True) or {}
        item_id = str(data.get("item_id", "")).strip()
        if not id_re.match(item_id):
            return jsonify({"error": "invalid item_id"}), 400

        meta = cached_meta(app, item_id)
        if meta is None:
            return jsonify({"error": "item has not been transcribed yet"}), 404

        item_dir = cache_dir / item_id
        loaded = decode.load_posteriograms(item_dir)
        if loaded is None:
            return jsonify({"error": "no stored model output for this item — transcribe it again"}), 409

        outputs, times = loaded
        try:
            meta.update(decode.apply_decoding(
                item_dir, item_id, outputs, times,
                decode.clean_decoding(data), source_audio(item_id),
            ))
        except Exception as e:
            return jsonify({"error": f"decoding failed: {e}"}), 500

        return jsonify(write_meta(app, item_id, meta))

    if source_audio is not None:
        app.add_url_rule("/api/decode", view_func=api_decode, methods=["POST"])

    return app


def cached_meta(app, item_id):
    """Already-processed meta for ``item_id``, or None."""
    meta_path = Path(app.config["CACHE_DIR"]) / item_id / "meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return None


def write_meta(app, item_id, meta):
    item_dir = Path(app.config["CACHE_DIR"]) / item_id
    item_dir.mkdir(parents=True, exist_ok=True)
    (item_dir / "meta.json").write_text(json.dumps(meta))
    return meta
