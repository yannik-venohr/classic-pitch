// ClassicPitch live view — microphone in, scrolling piano roll out.
//
// This app does not share /shared/app.js with the other two: that file
// is built around a track (an audio element, a duration, a seek bar, a
// canvas as long as the piece), and none of those exist here. What it
// does share is the look — /shared/style.css — and the way the model's
// output is read: the browser gets the two posteriograms and decodes
// them itself as it paints, with a port of the file apps' decoder.
//
// The shape of the thing:
//
//   worklet  →  rolling 6s window  →  POST /api/listen  →  posteriograms
//                                                            ↓
//   canvas   ←  display buffers, one column per model frame  ←
//
// Windows overlap heavily (a new one every ~250ms over 6 seconds of
// context), so most of what is on screen has been predicted many times
// over. Those repeats are averaged, which is what separates the settled
// part of the display from the live part: on the right you see the
// model's first guess at audio it has only just heard, and as that
// scrolls left it firms up into the average of a dozen or more passes.

// ---- Constants ----

// Seconds of audio sent per inference. Below about 4s the lowest CQT
// filter (16Hz, the h=0.5 harmonic of C1) no longer fits in the window,
// and the bottom of the range degrades; 6s leaves room to spare.
const CONTEXT_SECONDS = 6.0;

// Don't ask again until this much new audio exists. The real limit is
// how long a window takes (~250ms), so this mostly just stops a fast
// machine from re-predicting nearly-identical windows.
const MIN_NEW_SECONDS = 0.2;

// The rightmost strip, drawn as the raw latest prediction rather than an
// average. It is not a stylistic choice: a frame this close to now has
// been seen by only a window or two, and it takes about this long for
// the averaging to have anything to say.
const LIVE_ZONE_SECONDS = 2.0;

// Frames this close to the right edge of a *window* are left out of the
// averages (they still feed the live zone). The model is convolutional
// in time with a receptive field of about ±30 frames for the note head
// and ±42 for the onset head at hop 512, so the last second of any
// window is predicted partly from zero padding. Dropping it from the
// average is the difference between a settled display and one with a
// permanent smear of edge artefacts in it.
const EDGE_TRIM_SECONDS = 1.0;

const TARGET_SAMPLE_RATE = 22050;

// Canvas layout, in CSS pixels.
const PLOT_HEIGHT = 380;
const RULER_HEIGHT = 22;
const AXIS_WIDTH = 34;

// Same palette as the piano rolls and posteriograms in the other apps:
// note bodies blue, onsets yellow, on the roll container's own floor.
const BG = [24, 26, 32];
const BG_BLACK_KEY = [16, 18, 22];
const NOTE_COLOR = [90, 169, 255];
const ONSET_COLOR = [255, 211, 90];
const BLACK_KEY_PC = new Set([1, 3, 6, 8, 10]);

// Probability → brightness, the same curve webapp/shared/decode.py uses
// for its posteriogram images (gamma 0.6 to lift the midrange, damped
// below a 0.2 knee so the noise floor stays dark). A 256-entry table
// because it is evaluated once per cell per frame, ~30k times.
const TONE = new Float32Array(256);
for (let i = 0; i < 256; i++) {
  const x = i / 255;
  const u = Math.min(1, x / 0.2);
  const gate = u * u * (3 - 2 * u);
  TONE[i] = Math.pow(x, 0.6) * gate * gate;
}

// ---- DOM ----

const statusEl = document.getElementById('status');
const liveEl = document.getElementById('live');
const listenBtn = document.getElementById('listen-btn');
const deviceSelect = document.getElementById('device-select');
const windowSelect = document.getElementById('window-select');
const showOnsetsInput = document.getElementById('show-onsets');
const onsetThresholdInput = document.getElementById('onset-threshold');
const frameThresholdInput = document.getElementById('frame-threshold');
const rateReadout = document.getElementById('rate-readout');
const levelFill = document.getElementById('level-fill');
const canvas = document.getElementById('live-canvas');
const notationPanel = document.getElementById('notation-panel');
// The staff's two layers: the SVG /shared/notation.js engraves into, and
// the canvas the noteheads are painted on above it.
const staffWrap = document.getElementById('notation-wrap');
const staffSvgTarget = document.getElementById('notation-svg-target');
const staffCanvas = document.getElementById('staff-canvas');
const keyTonicSelect = document.getElementById('key-tonic-select');
const keyModeSelect = document.getElementById('key-mode-select');
const keyReadout = document.getElementById('key-readout');
const infoDrawer = document.getElementById('info-drawer');
const infoDrawerTab = document.getElementById('info-drawer-tab');
const infoDrawerContent = document.getElementById('info-drawer-content');

// ---- State ----

const state = {
  running: false,
  audioCtx: null,
  stream: null,
  sourceNode: null,
  recorderNode: null,

  // The rolling context window, kept at exactly CONTEXT_SECONDS: new
  // audio is appended at the right and the oldest falls off the left,
  // so whatever is in here is always what gets sent next.
  ring: null,
  sampleRate: TARGET_SAMPLE_RATE,
  samplesCaptured: 0,
  level: 0,

  inFlight: false,
  lastSentEnd: -Infinity,
  pending: [],          // responses waiting to be painted, drained in the frame loop
  windowRate: 0,        // windows/second, smoothed
  computeMs: 0,         // server-side ms per window, smoothed

  // Display buffers. Laid out row-major as [pitch][column]; column
  // cols-1 is "now". Null until the first response says how many frames
  // a second the model produces.
  frameRate: 0,
  minPitch: 24,
  nPitches: 72,
  cols: 0,
  splitCol: 0,
  edgeFrames: 0,
  rightFrame: 0,        // absolute model-frame index of the last column
  frac: 0,              // sub-frame scroll offset, for smooth motion
  sum: null,            // Σ note probability per cell, over overlapping windows
  onsetSum: null,       // Σ onset probability per cell
  count: null,          // how many windows contributed to each column
  latest: null,         // the newest window's raw note probability per cell

  image: null,          // ImageData, cols × nPitches
  offscreen: null,
  dirty: true,

  // Starts of the decoded notes currently on screen, as flat (column,
  // pitch) pairs — found while painting the roll, and what the staff is
  // engraved from.
  noteStarts: [],
  // Decaying pitch-class weights behind automatic key detection.
  pcHistogram: new Float64Array(12),
  detectedKey: null,
  keyDetectedAt: 0,
};

// ---- Status ----

function setStatus(msg, isError, isLoading) {
  statusEl.classList.toggle('error', !!isError);
  statusEl.classList.toggle('loading', !!isLoading);
  if (isLoading) {
    statusEl.innerHTML = `<span class="eq-bars">${'<span></span>'.repeat(9)}</span>` +
      `<span class="status-text"></span>`;
    statusEl.querySelector('.status-text').textContent = msg;
  } else {
    statusEl.textContent = msg || '';
  }
}

// ---- Microphone ----

// Labels are empty until permission has been granted at least once, so
// this is called again after the first successful getUserMedia.
async function refreshDevices() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
  let devices = [];
  try {
    devices = await navigator.mediaDevices.enumerateDevices();
  } catch (err) {
    return;
  }
  const inputs = devices.filter((d) => d.kind === 'audioinput');
  const previous = deviceSelect.value;
  deviceSelect.innerHTML = '';
  const auto = document.createElement('option');
  auto.value = '';
  auto.textContent = 'Default input';
  deviceSelect.appendChild(auto);
  inputs.forEach((d, i) => {
    const opt = document.createElement('option');
    opt.value = d.deviceId;
    opt.textContent = d.label || `Input ${i + 1}`;
    deviceSelect.appendChild(opt);
  });
  if (previous && inputs.some((d) => d.deviceId === previous)) deviceSelect.value = previous;
}

async function startListening() {
  if (state.running) return;
  setStatus('Starting the microphone…', false, true);
  listenBtn.disabled = true;

  try {
    // Every one of these is off on purpose. The browser's defaults are
    // tuned for speech on a laptop mic: echo cancellation and noise
    // suppression both treat sustained harmonic sound as something to
    // remove, and automatic gain control rides over exactly the dynamics
    // the model is being asked to read.
    const deviceId = deviceSelect.value;
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: 1,
      },
    });

    // Asking for the model's own rate lets the browser resample for us,
    // which is both faster and better than anything worth writing here.
    // If it refuses, /api/listen resamples instead — hence sending the
    // rate rather than assuming it.
    state.audioCtx = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: TARGET_SAMPLE_RATE,
      latencyHint: 'interactive',
    });
    await state.audioCtx.audioWorklet.addModule('/recorder.js');
    await state.audioCtx.resume();

    state.sampleRate = state.audioCtx.sampleRate;
    state.ring = new Float32Array(Math.round(CONTEXT_SECONDS * state.sampleRate));
    state.samplesCaptured = 0;
    state.lastSentEnd = -Infinity;
    state.pending = [];
    state.pcHistogram.fill(0);
    state.detectedKey = null;
    resetTimeline();

    state.sourceNode = state.audioCtx.createMediaStreamSource(state.stream);
    state.recorderNode = new AudioWorkletNode(state.audioCtx, 'recorder');
    state.recorderNode.port.onmessage = (e) => onAudioBlock(e.data);
    // Connected through to the destination at zero gain: some browsers
    // only pull a worklet that has a path to the output, and this one
    // must not play the microphone back into the room.
    const mute = state.audioCtx.createGain();
    mute.gain.value = 0;
    state.sourceNode.connect(state.recorderNode).connect(mute).connect(state.audioCtx.destination);

    state.running = true;
    liveEl.hidden = false;
    listenBtn.textContent = 'Stop';
    listenBtn.classList.add('listening');
    resizeCanvas();
    setStatus('Loading the model…', false, true);
    refreshDevices();
  } catch (err) {
    stopListening();
    const denied = err && (err.name === 'NotAllowedError' || err.name === 'SecurityError');
    setStatus(denied
      ? 'Microphone access was denied — allow it for this page and try again.'
      : `Could not start listening: ${err.message || err}`, true);
  } finally {
    listenBtn.disabled = false;
  }
}

function stopListening() {
  state.running = false;
  if (state.recorderNode) {
    state.recorderNode.port.onmessage = null;
    state.recorderNode.disconnect();
  }
  if (state.sourceNode) state.sourceNode.disconnect();
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
  if (state.audioCtx) state.audioCtx.close().catch(() => {});
  state.recorderNode = null;
  state.sourceNode = null;
  state.stream = null;
  state.audioCtx = null;
  state.level = 0;
  levelFill.style.width = '0%';
  listenBtn.textContent = 'Start listening';
  listenBtn.classList.remove('listening');
  // The canvas is left as it was: the last few seconds stay readable
  // after stopping, which is usually the bit you wanted to look at.
}

function onAudioBlock(block) {
  const ring = state.ring;
  if (!ring || block.length === 0) return;
  if (block.length >= ring.length) {
    ring.set(block.subarray(block.length - ring.length));
  } else {
    ring.copyWithin(0, block.length);
    ring.set(block, ring.length - block.length);
  }
  state.samplesCaptured += block.length;

  let peak = 0;
  for (let i = 0; i < block.length; i++) {
    const v = block[i] < 0 ? -block[i] : block[i];
    if (v > peak) peak = v;
  }
  // Instant attack, gentle release, so a meter read at 60fps still shows
  // the peak of a block that lasted 90ms.
  state.level = Math.max(peak, state.level * 0.85);
}

// ---- Inference requests ----

async function maybeRequest() {
  if (!state.running || state.inFlight || !state.ring) return;
  const minNew = MIN_NEW_SECONDS * state.sampleRate;
  if (state.samplesCaptured - state.lastSentEnd < minNew) return;

  state.inFlight = true;
  const endSamples = state.samplesCaptured;
  state.lastSentEnd = endSamples;
  // A copy: the ring keeps being written into while this is in flight.
  const payload = new Float32Array(state.ring);
  const sentAt = performance.now();

  try {
    const res = await fetch(`/api/listen?sr=${Math.round(state.sampleRate)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: payload.buffer,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'inference failed');
    if (!state.running) return;
    state.pending.push({ data, endSamples });

    const elapsed = performance.now() - sentAt;
    const rate = 1000 / Math.max(elapsed, 1);
    // First measurement wins outright, then a slow EMA — the very first
    // window includes loading the model, which is not a useful number to
    // average into the rest.
    state.windowRate = state.windowRate ? state.windowRate * 0.8 + rate * 0.2 : rate;
    state.computeMs = state.computeMs
      ? state.computeMs * 0.8 + data.compute_ms * 0.2 : data.compute_ms;
    if (statusEl.textContent) setStatus('');
  } catch (err) {
    if (state.running) setStatus(`Live transcription stopped: ${err.message || err}`, true);
  } finally {
    state.inFlight = false;
  }
}

// ---- Display buffers ----

function resetTimeline() {
  state.frameRate = 0;
  state.cols = 0;
  state.sum = null;
  state.noteStarts.length = 0;
  state.dirty = true;
}

// Called with the axes the first response describes, and again whenever
// the window length changes. Existing content is kept where it can be:
// changing the zoom should reframe what is on screen, not wipe it.
function ensureTimeline(frameRate, minPitch, nPitches) {
  const cols = Math.max(8, Math.round(parseFloat(windowSelect.value) * frameRate));
  if (state.cols === cols && state.frameRate === frameRate
      && state.minPitch === minPitch && state.nPitches === nPitches) return;

  const old = state.cols && state.nPitches === nPitches ? state : null;
  const keep = old ? Math.min(old.cols, cols) : 0;

  const alloc = (previous) => {
    const next = new Float32Array(nPitches * cols);
    for (let p = 0; keep && p < nPitches; p++) {
      // Both buffers are right-aligned on "now", so the columns that
      // survive a resize are the last `keep` of each row.
      next.set(previous.subarray((p + 1) * old.cols - keep, (p + 1) * old.cols),
               (p + 1) * cols - keep);
    }
    return next;
  };
  const count = new Int32Array(cols);
  if (keep) count.set(old.count.subarray(old.cols - keep), cols - keep);

  state.sum = alloc(old ? old.sum : null);
  state.onsetSum = alloc(old ? old.onsetSum : null);
  state.latest = alloc(old ? old.latest : null);
  state.count = count;

  state.frameRate = frameRate;
  state.minPitch = minPitch;
  state.nPitches = nPitches;
  state.cols = cols;
  state.splitCol = Math.max(0, cols - Math.round(LIVE_ZONE_SECONDS * frameRate));
  state.edgeFrames = Math.round(EDGE_TRIM_SECONDS * frameRate);
  if (!old) state.rightFrame = Math.floor(nowFrameFloat());

  state.offscreen = document.createElement('canvas');
  state.offscreen.width = cols;
  state.offscreen.height = nPitches;
  state.image = state.offscreen.getContext('2d').createImageData(cols, nPitches);
  const data = state.image.data;
  for (let i = 3; i < data.length; i += 4) data[i] = 255;
  state.dirty = true;
}

// Where the right edge of the display is, in model frames. Driven by the
// audio capture clock rather than wall time: the two drift apart over a
// few minutes, and it is the audio clock that the predictions are
// timestamped against.
function nowFrameFloat() {
  return (state.samplesCaptured / state.sampleRate) * state.frameRate;
}

function shiftLeft(array, rows, cols, n) {
  for (let r = 0; r < rows; r++) {
    const start = r * cols;
    array.copyWithin(start, start + n, start + cols);
    array.fill(0, start + cols - n, start + cols);
  }
}

function advanceTo(frame) {
  const n = frame - state.rightFrame;
  if (n <= 0) return;
  state.rightFrame = frame;
  const { cols, nPitches } = state;
  if (n >= cols) {
    state.sum.fill(0);
    state.onsetSum.fill(0);
    state.latest.fill(0);
    state.count.fill(0);
  } else {
    shiftLeft(state.sum, nPitches, cols, n);
    shiftLeft(state.onsetSum, nPitches, cols, n);
    shiftLeft(state.latest, nPitches, cols, n);
    state.count.copyWithin(0, n);
    state.count.fill(0, cols - n);
  }
  state.dirty = true;
}

function decodeBase64(str, expected) {
  const binary = atob(str);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out.length === expected ? out : null;
}

function paintResponse(entry) {
  const { data, endSamples } = entry;
  const frameRate = data.sample_rate / data.hop_length;
  ensureTimeline(frameRate, data.min_pitch, data.n_pitches);

  const nFrames = data.n_frames;
  const note = decodeBase64(data.note, nFrames * data.n_pitches);
  const onset = decodeBase64(data.onset, nFrames * data.n_pitches);
  if (!note || !onset) return;

  // Frame 0 of the response is the start of the window that was sent,
  // which ended at endSamples of the capture.
  const startSeconds = (endSamples - data.window_samples) / state.sampleRate;
  const colBase = state.cols - 1 - state.rightFrame + Math.round(startSeconds * frameRate);

  const first = Math.max(0, -colBase);
  const last = Math.min(nFrames, state.cols - colBase);
  if (first >= last) return;
  // Everything up to here goes into the running average; the rest is
  // too close to the window's edge to trust (see EDGE_TRIM_SECONDS) and
  // only feeds the live zone.
  const avgLast = Math.min(last, nFrames - state.edgeFrames);

  const { cols, nPitches, sum, onsetSum, latest, count } = state;
  for (let p = 0; p < nPitches; p++) {
    const src = p * nFrames;
    const dst = p * cols + colBase;
    for (let i = first; i < avgLast; i++) {
      const v = note[src + i] / 255;
      sum[dst + i] += v;
      onsetSum[dst + i] += onset[src + i] / 255;
      latest[dst + i] = v;
    }
    for (let i = Math.max(first, avgLast); i < last; i++) {
      latest[dst + i] = note[src + i] / 255;
    }
  }
  for (let i = first; i < avgLast; i++) count[colBase + i] += 1;
  state.dirty = true;
}

// ---- Decoding ----
//
// A port of src/postprocess/postprocess.py's output_to_note_events, so
// the notes drawn here are the notes the file apps would decode from the
// same posteriograms: an onset is a strict local maximum in time at or
// above the onset threshold, a note runs from there while the frame
// activation stays at or above the frame threshold (bridging dips of up
// to DECODE_GAP_FRAMES), and anything DECODE_MIN_FRAMES long or shorter
// is dropped. Both constants are that function's defaults, which is what
// webapp/shared/decode.py calls it with.
const DECODE_MIN_FRAMES = 4;
const DECODE_GAP_FRAMES = 5;

// One pitch row. `frame` and `onset` are its activations over `n`
// frames; onsets are looked for only before `onsetLimit`, while a note's
// end may be found anywhere up to `n`. Calls emit(start, end) per note,
// with `end` exclusive — the frame whose time the Python version reports
// as the note's end.
function decodeNotes(frame, onset, n, onsetLimit, frameThreshold, onsetThreshold, emit) {
  // scipy's argrelmax: strictly greater than both neighbours, and never
  // at either end of the array.
  const lastOnset = Math.min(onsetLimit, n - 1);
  for (let start = 1; start < lastOnset; start++) {
    const o = onset[start];
    if (o < onsetThreshold || !(o > onset[start - 1] && o > onset[start + 1])) continue;
    if (start >= n - 1) continue;

    let i = start + 1;
    let k = 0;
    while (i < n - 1 && k < DECODE_GAP_FRAMES) {
      if (frame[i] < frameThreshold) k += 1;
      else k = 0;
      i += 1;
    }
    i -= k;
    if (i - start <= DECODE_MIN_FRAMES) continue;
    emit(start, i);
  }
}

// ---- Painting ----

// Per-row scratch for renderGrid, reused across frames.
let rowFrame = new Float32Array(0);
let rowOnset = new Float32Array(0);
let rowMask = new Uint8Array(0);
const MASK_BODY = 1;
const MASK_ONSET = 2;

function renderGrid() {
  const { cols, nPitches, splitCol, sum, onsetSum, latest, count } = state;
  const data = state.image.data;
  const frameThreshold = parseFloat(frameThresholdInput.value);
  const onsetThreshold = parseFloat(onsetThresholdInput.value);
  const showOnsets = showOnsetsInput.checked;
  // The starts of the decoded notes, collected whether or not the roll
  // is marking them, because the staff is engraved from exactly this
  // list — the checkbox above the roll says what to paint, not what to
  // find. Flat (column, pitch) pairs rather than objects: this is
  // rebuilt every frame.
  const starts = state.noteStarts;
  starts.length = 0;

  if (rowFrame.length !== cols) {
    rowFrame = new Float32Array(cols);
    rowOnset = new Float32Array(cols);
    rowMask = new Uint8Array(cols);
  }

  for (let p = 0; p < nPitches; p++) {
    const pitch = state.minPitch + p;
    const bg = BLACK_KEY_PC.has(((pitch % 12) + 12) % 12) ? BG_BLACK_KEY : BG;
    const rowOff = p * cols;

    // The decoder sees the best estimate there is of each frame: the
    // average wherever windows have been averaged in, and the newest raw
    // pass for the last second, which none have yet. Onsets are only
    // taken from the settled zone, but a note found there keeps going
    // into the live zone until it actually ends — otherwise every note
    // still sounding would be cut at the dashed line, and short ones
    // there dropped as too short.
    for (let c = 0; c < cols; c++) {
      const n = count[c];
      rowFrame[c] = n ? sum[rowOff + c] / n : latest[rowOff + c];
      rowOnset[c] = n ? onsetSum[rowOff + c] / n : 0;
    }
    rowMask.fill(0);
    decodeNotes(rowFrame, rowOnset, cols, splitCol, frameThreshold, onsetThreshold,
      (noteStart, noteEnd) => {
        const bodyEnd = Math.min(noteEnd, splitCol);
        for (let c = noteStart; c < bodyEnd; c++) rowMask[c] |= MASK_BODY;
        rowMask[noteStart] |= MASK_ONSET;
        starts.push(noteStart, pitch);
      });

    // Row 0 of the image is the highest pitch, as in the other apps'
    // rolls; the buffers run the other way, lowest pitch first.
    let out = (nPitches - 1 - p) * cols * 4;

    for (let c = 0; c < cols; c++, out += 4) {
      let color = NOTE_COLOR;
      let alpha = 0;

      if (c < splitCol) {
        const mask = rowMask[c];
        if (mask & MASK_ONSET && showOnsets) {
          color = ONSET_COLOR;
          alpha = 1;
        } else if (mask) {
          alpha = 1;
        } else if (count[c]) {
          // Not part of a decoded note: a faint ghost of the activation,
          // so what the model nearly saw — or saw without an onset to
          // start a note from — is visible without being claimed as one.
          alpha = 0.35 * TONE[(rowFrame[c] * 255) | 0];
        }
      } else {
        // The live zone: the newest window's raw output, brightness
        // straight from the probability. Nothing is decoded here — this
        // is the model thinking out loud.
        alpha = TONE[(latest[rowOff + c] * 255) | 0];
      }

      if (alpha <= 0) {
        data[out] = bg[0]; data[out + 1] = bg[1]; data[out + 2] = bg[2];
      } else {
        data[out] = bg[0] + alpha * (color[0] - bg[0]);
        data[out + 1] = bg[1] + alpha * (color[1] - bg[1]);
        data[out + 2] = bg[2] + alpha * (color[2] - bg[2]);
      }
    }
  }
  state.offscreen.getContext('2d').putImageData(state.image, 0, 0);
  accumulatePitchClasses(starts);
}

function resizeCanvas() {
  const width = canvas.parentElement.clientWidth;
  const height = PLOT_HEIGHT + RULER_HEIGHT;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.height = `${height}px`;
  canvas.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
  state.dirty = true;
}

function draw() {
  const ctx = canvas.getContext('2d');
  const width = canvas.width / (window.devicePixelRatio || 1);
  const plotX = AXIS_WIDTH;
  const plotW = Math.max(1, width - AXIS_WIDTH);

  ctx.fillStyle = '#14161a';
  ctx.fillRect(0, 0, width, PLOT_HEIGHT + RULER_HEIGHT);

  if (!state.cols) {
    ctx.fillStyle = '#8b909c';
    ctx.font = '13px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Waiting for the first prediction…', width / 2, PLOT_HEIGHT / 2);
    return;
  }

  if (state.dirty) {
    renderGrid();
    state.dirty = false;
  }

  const pxPerCol = plotW / state.cols;
  ctx.save();
  ctx.beginPath();
  ctx.rect(plotX, 0, plotW, PLOT_HEIGHT);
  ctx.clip();
  // Shifted by the sub-frame remainder so the roll glides instead of
  // stepping a whole column at a time (~43 steps a second would read as
  // a judder at 60fps).
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(state.offscreen, plotX - state.frac * pxPerCol, 0, plotW, PLOT_HEIGHT);
  ctx.restore();

  // Octave guides, in the gutter and faintly across the plot.
  const rowH = PLOT_HEIGHT / state.nPitches;
  ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let p = 0; p < state.nPitches; p++) {
    const pitch = state.minPitch + p;
    if (pitch % 12 !== 0) continue;
    const y = PLOT_HEIGHT - (p + 0.5) * rowH;
    ctx.fillStyle = '#8b909c';
    ctx.fillText(`C${Math.floor(pitch / 12) - 1}`, AXIS_WIDTH - 8, y);
    ctx.strokeStyle = 'rgba(139, 144, 156, 0.12)';
    ctx.beginPath();
    ctx.moveTo(plotX, Math.round(y) + 0.5);
    ctx.lineTo(width, Math.round(y) + 0.5);
    ctx.stroke();
  }

  // The boundary between averaged and raw, and "now" at the right edge.
  const splitX = plotX + state.splitCol * pxPerCol;
  ctx.strokeStyle = 'rgba(90, 169, 255, 0.5)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(splitX, 0);
  ctx.lineTo(splitX, PLOT_HEIGHT);
  ctx.stroke();
  ctx.setLineDash([]);

  // What the dashed line separates, said once at the top rather than in
  // the ruler, where it would sit on top of a tick label.
  ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif';
  ctx.textBaseline = 'top';
  ctx.fillStyle = 'rgba(90, 169, 255, 0.75)';
  ctx.textAlign = 'right';
  ctx.fillText('averaged', splitX - 6, 5);
  ctx.textAlign = 'left';
  ctx.fillText('live', splitX + 6, 5);

  ctx.fillStyle = '#ff5a5a';
  ctx.fillRect(width - 2, 0, 2, PLOT_HEIGHT);

  drawRuler(ctx, plotX, plotW, width);
}

function drawRuler(ctx, plotX, plotW, width) {
  const seconds = state.cols / state.frameRate;
  const step = seconds > 14 ? 4 : seconds > 7 ? 2 : 1;
  const y = PLOT_HEIGHT;

  ctx.fillStyle = '#1c1f26';
  ctx.fillRect(0, y, width, RULER_HEIGHT);
  ctx.strokeStyle = 'rgba(139, 144, 156, 0.25)';
  ctx.beginPath();
  ctx.moveTo(0, y + 0.5);
  ctx.lineTo(width, y + 0.5);
  ctx.stroke();

  ctx.fillStyle = '#8b909c';
  ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif';
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'center';
  for (let s = step; s < seconds; s += step) {
    const x = plotX + plotW * (1 - s / seconds);
    ctx.fillText(`−${s}s`, x, y + RULER_HEIGHT / 2);
  }
  ctx.textAlign = 'right';
  ctx.fillStyle = '#ff5a5a';
  ctx.fillText('now', width - 4, y + RULER_HEIGHT / 2);
}

// ---- Staff notation ----
//
// The staff is engraved from the notes the roll already decodes, on
// exactly the same time axis, so each notehead sits directly above its
// note's yellow onset. Only the settled zone is notated: notation
// is a commitment, and the live zone is where the model has not made
// one yet.
//
// The staff itself comes from /shared/notation.js: the same grand staff,
// clefs and key signature the other two apps engrave, and the same idea
// of where a pitch sits on it. What differs here is who paints the
// notes. They move every frame, which VexFlow is nowhere near cheap
// enough for, so they go onto a canvas lying over the staff's SVG — and
// the staff underneath is redrawn only when the key or the width
// changes.

// Ledger-line room kept above and below the staff, in diatonic steps (a
// step is one line or one space). The model's range runs C1 to B6, which
// is ten steps above the treble stave's top line and eleven below the
// bass stave's bottom one. Reserved once and never fitted to what is
// playing: a staff that moved vertically would be unreadable, and the
// empty space above and below is the cost of that.
const STAFF_STEPS_ABOVE = 10;
const STAFF_STEPS_BELOW = 11;

// Noteheads are not upright, and are drawn a little wider than tall —
// as in the engraved staves the other apps draw, whose glyphs these are
// standing in for. Sized from the staff's own line spacing, so they fit
// their spaces whatever VexFlow's geometry turns out to be.
const NOTEHEAD_TILT = -0.32;   // radians
const NOTEHEAD_RX_RATIO = 0.62;
const NOTEHEAD_RY_RATIO = 0.46;

// Doubled single glyphs rather than U+1D12A/B: the double accidentals
// are outside the Basic Multilingual Plane and far less widely present
// than the single ones, and they turn up here only in remote spellings
// anyway.
const ACCIDENTAL_GLYPH = { '-2': '♭♭', '-1': '♭', 0: '♮', 1: '♯', 2: '♯♯' };

const INK = '#e8e9ec';

// How fast the pitch-class histogram behind automatic key detection
// forgets. Rebuilt every frame from what is on screen, so this is a
// smoothing constant rather than a memory: about a two-second average,
// enough to stop the key signature flickering between a key and its
// relative minor without making it slow to follow a modulation.
const KEY_DECAY = 0.995;
const KEY_DETECT_INTERVAL_MS = 500;
// Below this the histogram is a handful of stray notes, and detecting a
// key from it would just be picking one at random.
const KEY_MIN_EVIDENCE = 40;

function accumulatePitchClasses(starts) {
  const hist = state.pcHistogram;
  for (let i = 0; i < 12; i++) hist[i] *= KEY_DECAY;
  for (let i = 1; i < starts.length; i += 2) hist[((starts[i] % 12) + 12) % 12] += 1;
}

function updateDetectedKey(now) {
  if (!window.KeyDetection || now - state.keyDetectedAt < KEY_DETECT_INTERVAL_MS) return;
  state.keyDetectedAt = now;
  // detect() weights each note by its duration, so twelve notes whose
  // "length" is a pitch class's weight is precisely the histogram it
  // would otherwise build for itself.
  let total = 0;
  const notes = [];
  for (let pc = 0; pc < 12; pc++) {
    total += state.pcHistogram[pc];
    notes.push({ pitch: pc, start: 0, end: state.pcHistogram[pc] });
  }
  state.detectedKey = total >= KEY_MIN_EVIDENCE ? KeyDetection.detect(notes) : null;
}

// The key the staff is written in: whatever the dropdowns say, or the
// detected one while they are on Auto.
function currentKey() {
  const choice = keyTonicSelect.value;
  if (choice !== 'auto') {
    return {
      letter: choice[0],
      accidental: choice.length > 1 ? (choice[1] === 'b' ? 'flat' : 'sharp') : 'natural',
      mode: keyModeSelect.value,
    };
  }
  const detected = state.detectedKey;
  return detected
    ? { letter: detected.letter, accidental: detected.accidental, mode: detected.mode }
    : { letter: 'C', accidental: 'natural', mode: 'major' };
}

function keyName(key) {
  const accidental = key.accidental === 'flat' ? '♭' : key.accidental === 'sharp' ? '♯' : '';
  return `${key.letter}${accidental} ${key.mode}`;
}

// The staff /shared/notation.js drew, and what it was drawn for. Only
// two things change it — the key signature and the width — so keeping
// that pair is enough to know when it has to be engraved again.
let staff = null;
let staffDrawnFor = '';

function ensureStaff(keySpec, spelling) {
  // A collapsed <details> has nothing to draw into, and nothing to show.
  const width = staffWrap.clientWidth;
  if (!width) return;

  // Also how a failed engraving is remembered: if VexFlow is not there,
  // renderStaves() says so once rather than every frame.
  const spec = `${keySpec}@${width}`;
  if (spec === staffDrawnFor) return;
  staff = Notation.renderStaves(staffSvgTarget, {
    width,
    spelling,
    keySpec,
    stepsAbove: STAFF_STEPS_ABOVE,
    stepsBelow: STAFF_STEPS_BELOW,
  });
  staffDrawnFor = spec;
  if (!staff) return;

  // The canvas covers the SVG exactly, so a note drawn at the staff's
  // own coordinates lands on the line it belongs to.
  const dpr = window.devicePixelRatio || 1;
  staffCanvas.width = Math.round(width * dpr);
  staffCanvas.height = Math.round(staff.height * dpr);
  staffCanvas.style.width = `${width}px`;
  staffCanvas.style.height = `${staff.height}px`;
  staffCanvas.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
}

function drawStaff() {
  // Both come from /shared: without them there is no staff at all,
  // rather than one full of wrong accidentals.
  if (!window.KeyDetection || !window.Notation) return;

  const key = currentKey();
  const spelling = KeyDetection.buildSpelling(key.letter, key.accidental, key.mode);
  ensureStaff(KeyDetection.keySpecString(key.letter, key.accidental, key.mode), spelling);
  if (!staff) {
    // VexFlow is loaded from a CDN, as in the other two apps, so the one
    // way to get here is offline. Say so where the key would be, rather
    // than leaving an empty box and a line in the console.
    if (staffDrawnFor) keyReadout.textContent = 'staff unavailable — VexFlow did not load';
    return;
  }

  keyReadout.textContent = keyTonicSelect.value === 'auto'
    ? (state.detectedKey ? `hearing ${keyName(key)}` : 'listening…')
    : '';

  const width = staffWrap.clientWidth;
  const ctx = staffCanvas.getContext('2d');
  ctx.clearRect(0, 0, width, staff.height);
  if (!state.cols) return;

  const plotW = Math.max(1, width - AXIS_WIDTH);
  const pxPerCol = plotW / state.cols;
  const starts = state.noteStarts;
  const rx = staff.spacing * NOTEHEAD_RX_RATIO;
  const ry = staff.spacing * NOTEHEAD_RY_RATIO;

  ctx.save();
  // Left of this are the clef and the key signature, which stay put
  // while the music slides underneath them — the same arrangement as the
  // other apps' non-scrolling clef overlay, except that here it is the
  // staff that does not move, so the notes are simply clipped to it.
  ctx.beginPath();
  ctx.rect(staff.noteStartX, 0, Math.max(0, width - staff.noteStartX), staff.height);
  ctx.clip();

  ctx.fillStyle = INK;
  ctx.strokeStyle = INK;
  ctx.lineWidth = 1;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.font = `${Math.round(staff.spacing * 1.7)}px serif`;

  for (let i = 0; i < starts.length; i += 2) {
    const x = AXIS_WIDTH + (starts[i] + 0.5 - state.frac) * pxPerCol;
    if (x + rx < staff.noteStartX) continue;
    const { y, ledgers, accidental } = staff.place(starts[i + 1]);

    for (const ly of ledgers) {
      const lpy = Math.round(ly) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x - rx - 3, lpy);
      ctx.lineTo(x + rx + 3, lpy);
      ctx.stroke();
    }

    ctx.beginPath();
    ctx.ellipse(x, y, rx, ry, NOTEHEAD_TILT, 0, Math.PI * 2);
    ctx.fill();

    if (accidental !== null) ctx.fillText(ACCIDENTAL_GLYPH[accidental], x - rx - 2, y);
  }

  // The same two markers the roll carries, so it is obvious that the
  // staff shares its time axis — and that notes appear at the dashed
  // line, not at "now".
  const splitX = AXIS_WIDTH + state.splitCol * pxPerCol;
  ctx.strokeStyle = 'rgba(90, 169, 255, 0.5)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(splitX, 0);
  ctx.lineTo(splitX, staff.height);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#ff5a5a';
  ctx.fillRect(width - 2, 0, 2, staff.height);
  ctx.restore();
}

// ---- Frame loop ----

function frameLoop() {
  requestAnimationFrame(frameLoop);

  if (state.running) {
    levelFill.style.width = `${Math.min(100, Math.round(state.level * 140))}%`;

    if (state.frameRate) {
      const now = nowFrameFloat();
      advanceTo(Math.floor(now));
      state.frac = now - Math.floor(now);
    }
    while (state.pending.length) paintResponse(state.pending.shift());
    maybeRequest();

    if (state.windowRate) {
      rateReadout.textContent =
        `${state.windowRate.toFixed(1)} windows/s · ${Math.round(state.computeMs)} ms model`;
    }
  }

  if (liveEl.hidden) return;
  draw();
  // A collapsed <details> has a zero-width canvas inside it, so there is
  // nothing to draw and no point measuring one.
  if (notationPanel.open) {
    updateDetectedKey(performance.now());
    drawStaff();
  }
}

// ---- Wiring ----

listenBtn.addEventListener('click', () => {
  if (state.running) {
    stopListening();
    setStatus('');
  } else {
    startListening();
  }
});

deviceSelect.addEventListener('change', () => {
  if (!state.running) return;
  stopListening();
  startListening();
});

windowSelect.addEventListener('change', () => {
  if (state.frameRate) ensureTimeline(state.frameRate, state.minPitch, state.nPitches);
  state.dirty = true;
});

for (const input of [onsetThresholdInput, frameThresholdInput]) {
  input.addEventListener('input', () => {
    const out = document.getElementById(`${input.id}-value`);
    if (out) out.textContent = parseFloat(input.value).toFixed(2);
    state.dirty = true;
  });
}
showOnsetsInput.addEventListener('change', () => { state.dirty = true; });

keyTonicSelect.addEventListener('change', () => {
  // The mode belongs to a key you chose; on Auto it is whatever is being
  // heard, so leaving it live would be a control that does nothing.
  keyModeSelect.disabled = keyTonicSelect.value === 'auto';
});
keyModeSelect.disabled = true;

// The staff is not resized here: drawStaff() engraves it again by
// itself as soon as it sees a width it was not drawn for, which also
// covers the width it first gets when the panel is opened.
window.addEventListener('resize', resizeCanvas);

// Space starts and stops listening, the way it plays and pauses in the
// other two apps. Same guards: never steal a key from a control.
document.addEventListener('keydown', (e) => {
  const tag = e.target && e.target.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.code !== 'Space') return;
  e.preventDefault();
  listenBtn.click();
});

// ---- Info drawer ----

let infoLoaded = false;

infoDrawerTab.addEventListener('click', async () => {
  const opening = !infoDrawer.classList.contains('open');
  infoDrawer.classList.toggle('open', opening);
  infoDrawerTab.setAttribute('aria-expanded', String(opening));
  if (opening && !infoLoaded) {
    infoLoaded = true;
    try {
      infoDrawerContent.innerHTML = await (await fetch('/info-panel.html')).text();
    } catch (err) {
      infoDrawerContent.textContent = 'Could not load info panel.';
    }
  }
});

// getUserMedia only exists in a secure context, which 127.0.0.1 is but
// a LAN address served over plain http is not — a likely way to arrive
// here with no microphone at all, and worth saying so plainly rather
// than failing with a TypeError on the first click.
if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
  listenBtn.disabled = true;
  setStatus('This browser will not give the page a microphone. Open the app at '
    + 'http://127.0.0.1:5004 (or over https) — a plain http address on the '
    + 'network is not a secure context.', true);
} else {
  refreshDevices();
}
resizeCanvas();
requestAnimationFrame(frameLoop);
