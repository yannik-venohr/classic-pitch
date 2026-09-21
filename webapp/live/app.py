"""Local web app: live transcription from the microphone.

The third of the three apps, and the only one with nothing to cache —
there is no track, no file and no library, just the last few seconds of
a microphone going through the model again and again. The browser owns
the audio (capture, the rolling context window, the display) and this
serves one route that turns a window of samples into posteriograms; see
webapp/shared/live.py for what that costs and why it is shaped this way.

Because a page's audio never reaches disk, nothing here is cached and
nothing is shared between tabs: each one keeps its own window and asks
for its own predictions.

Run with:
    python webapp/live/app.py
then open http://127.0.0.1:5004 and allow microphone access.
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from webapp.shared import live
from webapp.shared.server import SHARED_STATIC

APP_ROOT = Path(__file__).resolve().parent

# What the frontend is built to send: a 6-second window of float32 mono
# at 22050 Hz is ~529KB. The cap is generous enough for a browser that
# hands us a 48kHz capture instead, and small enough that a stray client
# can't ask the process to allocate something silly.
MAX_WINDOW_SECONDS = 12.0
MIN_WINDOW_SECONDS = 1.0

app = Flask(__name__, static_folder=str(APP_ROOT / "static"), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# Same arrangement as the other apps: style.css and the rest of
# webapp/shared/static are served at /shared/. The live view has its own
# player, so it uses only the stylesheet — but it should look like the
# others, which means reading the same variables and classes.
@app.route("/shared/<path:filename>")
def shared_static(filename):
    return send_from_directory(SHARED_STATIC, filename)


@app.route("/api/listen", methods=["POST"])
def api_listen():
    """One window of raw float32 mono audio in, two posteriograms out.

    Stateless on purpose: the browser already keeps the rolling window
    it is recording into, so sending the whole thing every time (rather than
    just what is new) means no per-session buffers here, no cleanup when
    a tab goes away, and no way for two tabs to interfere.
    """
    try:
        sr = int(request.args.get("sr", 22050))
    except ValueError:
        return jsonify({"error": "invalid sample rate"}), 400
    if not 8000 <= sr <= 192000:
        return jsonify({"error": "unsupported sample rate"}), 400

    raw = request.get_data(cache=False)
    if len(raw) % 4:
        return jsonify({"error": "audio must be float32"}), 400
    audio = np.frombuffer(raw, dtype=np.float32)
    seconds = len(audio) / sr
    if not MIN_WINDOW_SECONDS <= seconds <= MAX_WINDOW_SECONDS:
        return jsonify({"error": f"window must be {MIN_WINDOW_SECONDS}-{MAX_WINDOW_SECONDS}s"}), 400

    started = time.perf_counter()
    try:
        outputs, axes = live.predict_window(audio, sr)
    except Exception as e:
        return jsonify({"error": f"inference failed: {e}"}), 500

    return jsonify({
        **axes,
        # How many of the samples that were sent this window covers, so
        # the browser can line the returned frames up against its own
        # sample counter without assuming a sample rate.
        "window_samples": len(audio),
        "compute_ms": round((time.perf_counter() - started) * 1000, 1),
        **{head: live.encode_posteriogram(outputs[head]) for head in live.HEADS},
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5004, debug=False, threaded=True)
