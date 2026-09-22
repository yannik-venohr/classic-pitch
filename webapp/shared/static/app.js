// ClassicPitch player core — vanilla JS, no build step.
//
// Shared by every app under webapp/. This file owns everything from
// "here is a transcribed track" onwards: the audio graph, the piano
// roll(s), the notation, the key controls and the info drawer. It does
// NOT own how a track gets *chosen* — the demo searches YouTube, the
// live app listens to the microphone — so each app ships its own
// picker script that drives this one through window.ClassicPitch below.

const statusEl = document.getElementById('status');

const player = document.getElementById('player');
const playerTitleEl = document.getElementById('player-title');
const playerThumbnail = document.getElementById('player-thumbnail');
const playBtn = document.getElementById('play-btn');
const backBtn = document.getElementById('back-btn');
const seekBar = document.getElementById('seek-bar');
const timeLabel = document.getElementById('time-label');
const crossfadeSlider = document.getElementById('crossfade-slider');
const notationContainer = document.getElementById('notation-container');
const notationSvgTarget = document.getElementById('notation-svg-target');
const hoverTooltip = document.getElementById('hover-tooltip');
const exampleCaption = document.getElementById('example-caption');
const sourceLink = document.getElementById('source-link');
const infoDrawer = document.getElementById('info-drawer');
const infoDrawerTab = document.getElementById('info-drawer-tab');
const infoDrawerContent = document.getElementById('info-drawer-content');
const keyTonicSelect = document.getElementById('key-tonic-select');
const keyModeSelect = document.getElementById('key-mode-select');
const timeContextSelect = document.getElementById('time-context-select');
const keyResetBtn = document.getElementById('key-reset-btn');

// Collapsible sections. Both are optional — an app that doesn't ship
// them in its index.html just gets the old always-visible layout (the
// model panel additionally needs its backend turned on, see
// create_app's source_audio in webapp/shared/server.py).
const notationPanel = document.getElementById('notation-panel');
const modelPanel = document.getElementById('model-panel');
const onsetThresholdInput = document.getElementById('onset-threshold');
const frameThresholdInput = document.getElementById('frame-threshold');
const mergeNotesInput = document.getElementById('merge-notes');
const decodeStatusEl = document.getElementById('decode-status');

// The tonic dropdown packs letter+accidental into one value (e.g. "Eb",
// "F#", "C") since a key needs both together anyway — this just splits
// it back apart for the (letter, accidental) functions in key-detection.js.
function parseTonic(value) {
  return {
    letter: value[0],
    accidental: value.length > 1 ? (value[1] === 'b' ? 'flat' : 'sharp') : 'natural',
  };
}

// Shift the roll's own content right by the same amount notation.js
// reserves for its clef, so t=0 sits at the same absolute x in both views
// (and, as a side effect, their auto-scroll — computed from that same x —
// tracks in lockstep instead of just their internal playhead math). The
// same goes for END_PADDING on the far side: the two containers need
// identical total widths, or they hit their max scrollLeft at different
// points near the end of the track and the playheads drift apart again.
const ROLL_MARGIN = (window.Notation && Notation.NOTE_AREA_X) || 0;
const ROLL_END_PADDING = (window.Notation && Notation.END_PADDING) || 0;

let audioCtx = null;
let audioOrig = null;
let audioMidi = null;
let sourceOrig = null;
let sourceMidi = null;
let gainOrig = null;
let gainMidi = null;
let limiter = null;

let currentMeta = null;
let isPlaying = false;
let isDraggingSeek = false;
let pixelsPerSecond = 60;
let timeContextSeconds = 5;

// The key KeyDetection picked for the current track, kept so the "↺ Auto"
// button can put a manually-changed key back. Null until a track with a
// usable detection is loaded.
let detectedKey = null;

// A piano-roll view. How many exist is decided by each app's
// index.html, not by this file: the YouTube demo has only the
// transcription, but an app can add a second roll (e.g. for annotated
// ground truth). createRoll returns null when the markup for a roll
// isn't there, so every app runs this same script.
//
// `notesKey` is the field of the meta object a roll draws, which is also
// what makes the second roll optional server-side — a backend with no
// ground truth to offer simply omits `gt_notes`.
function createRoll(prefix, colors, notesKey) {
  const container = document.getElementById(`${prefix}roll-container`);
  const canvas = document.getElementById(`${prefix}roll-canvas`);
  const playhead = document.getElementById(`${prefix}playhead`);
  if (!container || !canvas || !playhead) return null;
  return {
    container,
    canvas,
    playhead,
    colors,
    notesKey,
    // The roll plus whatever labels an app wrapped around it, so a roll
    // with nothing to show can be hidden heading and all.
    section: container.closest('.roll-section') || container,
    notes: [],
    minPitch: 48,
    maxPitch: 72,
    noteHeight: 6,
  };
}

const ROLL_COLORS = {
  prediction: { body: '#5aa9ff', onset: '#ffd35a' },
  groundTruth: { body: '#4fd18b', onset: '#ffd35a' },
};

// The transcription roll — the one roll every app has, and the one that
// stays visible even when it's empty (see drawRolls).
const predictionRoll = createRoll('', ROLL_COLORS.prediction, 'notes');
const groundTruthRoll = createRoll('gt-', ROLL_COLORS.groundTruth, 'gt_notes');
const rolls = [groundTruthRoll, predictionRoll].filter(Boolean);

// A posteriogram view: the model's raw per-(pitch, frame) output for one
// of its two heads, served as a PNG and blitted onto the same time axis
// as the rolls. Same shape as a roll so it can share the playhead and
// scroll machinery; only the painting differs. Null when the app's
// markup has no model panel.
function createPosteriogram(kind) {
  const container = document.getElementById(`${kind}-poster-container`);
  const canvas = document.getElementById(`${kind}-poster-canvas`);
  const playhead = document.getElementById(`${kind}-poster-playhead`);
  if (!container || !canvas || !playhead) return null;
  return { container, canvas, playhead, kind, image: null, imageUrl: null };
}

const posteriograms = [createPosteriogram('onset'), createPosteriogram('frame')].filter(Boolean);

// Everything drawn against the shared time axis, in one list: the rolls
// plus the posteriograms. They all carry a playhead at the same x.
const timelineViews = rolls.concat(posteriograms);

// Every horizontally-scrolling view of the same timeline. They are all
// laid out at identical widths (see ROLL_MARGIN above), so one scroll
// position is meaningful in all of them.
const scrollViews = timelineViews.map((view) => view.container).concat(notationContainer ? [notationContainer] : []);

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
function pitchName(pitch) {
  const octave = Math.floor(pitch / 12) - 1;
  return `${NOTE_NAMES[((pitch % 12) + 12) % 12]}${octave}`;
}

// Real pipeline stages (mirrors predictor.py's own verbose log messages)
// cycled purely for show while waiting — this isn't live backend
// progress, just a truthful description of what's actually happening
// during the wait, rotating so it doesn't look frozen.
const PIPELINE_STAGES = [
  'Loading audio…',
  'Computing harmonic CQT…',
  'Running neural network inference…',
  'Decoding note events…',
  'Merging overlapping notes…',
];

let statusStageTimer = null;

function setStatus(msg, isError, showStages) {
  if (statusStageTimer) {
    clearInterval(statusStageTimer);
    statusStageTimer = null;
  }

  statusEl.classList.toggle('error', !!isError);
  const isLoading = !isError && !!msg;
  statusEl.classList.toggle('loading', isLoading);

  if (!isLoading) {
    statusEl.textContent = msg || '';
    return;
  }

  statusEl.innerHTML = `
    <span class="eq-bars">${'<span></span>'.repeat(9)}</span>
    <span class="status-text">${escapeHtml(msg)}</span>
  `;

  if (showStages) {
    const textEl = statusEl.querySelector('.status-text');
    const baseMsg = msg;
    let i = 0;
    const tick = () => {
      textEl.textContent = `${baseMsg} — ${PIPELINE_STAGES[i % PIPELINE_STAGES.length]}`;
      i++;
    };
    tick();
    statusStageTimer = setInterval(tick, 1800);
  }
}

function formatTime(s) {
  if (!isFinite(s) || s < 0) s = 0;
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${String(sec).padStart(2, '0')}`;
}

function updateTimeLabel(t) {
  timeLabel.textContent = `${formatTime(t)} / ${formatTime(currentMeta ? currentMeta.duration : 0)}`;
}

function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s == null ? '' : s;
  return div.innerHTML;
}

// ---- Transcription ----

// Filled in by whichever picker script the app loads, so this file can
// clear the picker's results and disable its controls for the duration
// of a transcription without knowing what shape that picker takes.
const picker = {
  clearResults() {},
  setBusy(busy) {},
  onLibrary(items) {},
};

async function transcribe(itemId, title, captionHtml) {
  picker.clearResults();
  player.hidden = true;
  setStatus(`Transcribing "${title || itemId}"…`, false, true);
  picker.setBusy(true);
  try {
    const res = await fetch('/api/transcribe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ item_id: itemId }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'transcription failed');
    setStatus('');
    loadPlayer(data, captionHtml);
    loadLibrary();
  } catch (err) {
    setStatus(String(err.message || err), true);
  } finally {
    picker.setBusy(false);
  }
}

// ---- Library of already-transcribed items ----

let libraryItems = [];

async function loadLibrary() {
  try {
    const res = await fetch('/api/library');
    const data = await res.json();
    libraryItems = data.items || [];
  } catch (err) {
    libraryItems = [];
  }
  // Pickers use this to mark what is already cached — worth showing,
  // since a cached item loads instantly and an uncached one means
  // waiting on the model.
  picker.onLibrary(libraryItems);
  return libraryItems;
}

// ---- Player ----

function loadPlayer(meta, captionHtml) {
  currentMeta = meta;
  isPlaying = false;
  playBtn.textContent = '▶';

  if (captionHtml) {
    // innerHTML (not textContent): captions are authored by you in
    // info-panel.html, so lists/formatting/timestamp links in there
    // should render as real markup, not get flattened to plain text.
    exampleCaption.innerHTML = captionHtml;
    exampleCaption.hidden = false;
    exampleCaption.querySelectorAll('[data-seek]').forEach((el) => {
      el.addEventListener('click', (e) => {
        e.preventDefault();
        seekTo(parseFloat(el.getAttribute('data-seek')));
      });
    });
  } else {
    exampleCaption.hidden = true;
    exampleCaption.innerHTML = '';
  }

  teardownAudioGraph();

  // The link back to wherever the recording came from (a YouTube watch
  // page, say) is whatever the backend put in the meta, and its label
  // lives in each app's index.html. An app whose recordings have no such
  // page (local files, say) leaves the element out
  // of its markup entirely, hence the null check as well as the field
  // check.
  if (sourceLink) {
    if (meta.source_url) {
      sourceLink.href = meta.source_url;
      sourceLink.hidden = false;
    } else {
      sourceLink.hidden = true;
    }
  }
  if (meta.thumbnail) {
    playerThumbnail.src = meta.thumbnail;
    playerThumbnail.hidden = false;
  } else {
    playerThumbnail.hidden = true;
    playerThumbnail.src = '';
  }

  createAudioElements(meta.audio_url, meta.midi_audio_url);

  buildAudioGraph();

  playerTitleEl.textContent = meta.title || meta.item_id;
  seekBar.max = String(meta.duration || 0);
  seekBar.value = '0';
  updateTimeLabel(0);

  // Unhidden before measuring/drawing: computePixelsPerSecond() reads
  // the roll container's rendered width, which is 0 while the player is
  // still `[hidden]` (display:none) — draw calls before this point would
  // zoom using a bogus width.
  player.hidden = false;

  pixelsPerSecond = computePixelsPerSecond(meta.duration || 0);
  // Before drawRolls: the previous track's posteriograms must not be
  // painted onto this one's time axis while the new PNGs load.
  clearPosteriogramImages();
  syncDecodingControls(meta);
  drawRolls(meta);
  if (modelPanel && modelPanel.open) loadPosteriogramImages();

  detectedKey = null;
  if (window.KeyDetection) {
    // Detected from whatever the staff will be engraved from, so the
    // key signature and the notes it governs can't disagree.
    const detected = KeyDetection.detect(meta[notationNotesKey()] || []);
    const accChar = detected.accidental === 'flat' ? 'b' : detected.accidental === 'sharp' ? '#' : '';
    keyTonicSelect.value = `${detected.letter}${accChar}`;
    // A <select> silently clears itself when assigned a value no option
    // has. Every spelling detect() can return is in the list, but reading
    // the value back (rather than trusting the assignment) means a future
    // edit to the options can't leave the dropdown — and the reset button
    // that restores it — pointing at nothing.
    if (!keyTonicSelect.value) keyTonicSelect.value = 'C';
    keyModeSelect.value = detected.mode;
    detectedKey = { tonic: keyTonicSelect.value, mode: keyModeSelect.value };
  }
  onKeyChanged();

  updatePlayhead(0);
}

// The reset button is disabled (rather than hidden) while the selected
// key already is the detected one, so the controls row doesn't reflow
// every time the key changes.
function updateKeyResetState() {
  keyResetBtn.disabled = !detectedKey
    || (keyTonicSelect.value === detectedKey.tonic && keyModeSelect.value === detectedKey.mode);
}

function onKeyChanged() {
  updateKeyResetState();
  renderNotationForCurrentKey();
}

function resetKeyToDetected() {
  if (!detectedKey) return;
  keyTonicSelect.value = detectedKey.tonic;
  keyModeSelect.value = detectedKey.mode;
  onKeyChanged();
}

// Which of the meta's note lists the staff is engraved from. Where the
// app has the performance's own annotated MIDI, that is the better
// source by far: it notates cleanly, and — unlike the transcription — it
// doesn't move when the decoding thresholds do, so the staff stays a
// fixed reference to read the model's output against. An app with no
// annotations (the YouTube demo) falls back to the transcription.
function notationNotesKey() {
  const annotated = currentMeta && currentMeta.gt_notes;
  return annotated && annotated.length ? 'gt_notes' : 'notes';
}

// Set while the staff needs a re-render it couldn't do because its panel
// was collapsed; cleared by the render that finally happens on reopen.
let notationPending = false;

function renderNotationForCurrentKey() {
  if (!window.Notation || !window.KeyDetection || !currentMeta) return;
  // VexFlow places noteheads by reading getBBox() off the SVG it just
  // drew, and an SVG inside a collapsed <details> is display:none, where
  // getBBox() reports zeros — rendering there would pile every note up
  // at the left edge. Wait for the panel to open instead.
  if (notationPanel && !notationPanel.open) {
    notationPending = true;
    return;
  }
  notationPending = false;
  const { letter, accidental } = parseTonic(keyTonicSelect.value);
  const mode = keyModeSelect.value;
  const spelling = KeyDetection.buildSpelling(letter, accidental, mode);
  const keySpec = KeyDetection.keySpecString(letter, accidental, mode);
  const notes = currentMeta[notationNotesKey()] || [];
  Notation.render(notes, currentMeta.duration || 0, pixelsPerSecond, spelling, keySpec, playNote);
  // render() resets its own playhead to 0 — put it back where playback
  // actually is (e.g. when the key is changed mid-track).
  Notation.setTime(playheadTime(), false);
}

// ---- "Open the model": decoding controls and posteriograms ----
//
// Note events are not the model's output — they are a thresholding of
// its two posteriograms, which the backend keeps alongside the cached
// transcription. So the thresholds can be changed after the fact, at
// the cost of a re-decode and a re-render of the MIDI track (~a second)
// rather than a full inference pass. Everything below is inert in an
// app whose index.html has no model panel.

const hasModelPanel = !!(modelPanel && onsetThresholdInput && frameThresholdInput && mergeNotesInput);

function currentDecoding() {
  return {
    onset_threshold: parseFloat(onsetThresholdInput.value),
    frame_threshold: parseFloat(frameThresholdInput.value),
    merge_notes: mergeNotesInput.checked,
  };
}

function updateThresholdReadouts() {
  for (const input of [onsetThresholdInput, frameThresholdInput]) {
    const out = document.getElementById(`${input.id}-value`);
    if (out) out.textContent = parseFloat(input.value).toFixed(2);
  }
}

// Reflect whatever the item was last decoded at, so the controls always
// describe the notes actually on screen — including for an item that
// was cached at different settings in an earlier session.
function syncDecodingControls(meta) {
  if (!hasModelPanel) return;
  const decoding = (meta && meta.decoding) || {};
  if (decoding.onset_threshold != null) onsetThresholdInput.value = String(decoding.onset_threshold);
  if (decoding.frame_threshold != null) frameThresholdInput.value = String(decoding.frame_threshold);
  mergeNotesInput.checked = !!decoding.merge_notes;
  updateThresholdReadouts();
  // An item transcribed before posteriograms were stored can't be
  // re-decoded; the panel says so rather than offering dead controls.
  const available = !!(meta && meta.posteriograms);
  setDecodeControlsEnabled(available);
  if (decodeStatusEl) {
    decodeStatusEl.textContent = available ? '' : 'Transcribe this item again to enable re-decoding.';
  }
}

function setDecodeControlsEnabled(enabled) {
  for (const el of [onsetThresholdInput, frameThresholdInput, mergeNotesInput]) el.disabled = !enabled;
}

// Posteriogram PNGs are fetched only once their panel is opened: they
// are the biggest thing the page loads and most sessions never unfold
// the panel at all. They depend only on the item, not the thresholds,
// so a re-decode never invalidates them.
function loadPosteriogramImages() {
  if (!currentMeta || !currentMeta.posteriograms) return;
  for (const view of posteriograms) {
    const url = currentMeta.posteriograms.urls[view.kind];
    if (!url || view.imageUrl === url) continue;
    view.imageUrl = url;
    const img = new Image();
    img.onload = () => {
      // Guard against a slow image for a track that has since been
      // swapped out from under it.
      if (view.imageUrl !== url) return;
      view.image = img;
      if (currentMeta) drawRolls(currentMeta);
    };
    img.src = url;
  }
}

function clearPosteriogramImages() {
  for (const view of posteriograms) {
    view.image = null;
    view.imageUrl = null;
  }
}

// One re-decode at a time, latest settings win: a slider released twice
// in quick succession should end up showing the second value, not
// whichever response happens to land last.
let decodeInFlight = false;
let decodeQueued = false;

async function runDecode() {
  if (!hasModelPanel || !currentMeta || !currentMeta.posteriograms) return;
  if (decodeInFlight) {
    decodeQueued = true;
    return;
  }
  decodeInFlight = true;
  setDecodeControlsEnabled(false);
  if (decodeStatusEl) decodeStatusEl.textContent = 'Re-decoding…';
  try {
    const res = await fetch('/api/decode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ item_id: currentMeta.item_id, ...currentDecoding() }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'decoding failed');
    applyDecoded(data);
    if (decodeStatusEl) decodeStatusEl.textContent = `${(data.notes || []).length} notes`;
  } catch (err) {
    if (decodeStatusEl) decodeStatusEl.textContent = '';
    setStatus(String(err.message || err), true);
  } finally {
    decodeInFlight = false;
    setDecodeControlsEnabled(true);
    if (decodeQueued) {
      decodeQueued = false;
      runDecode();
    }
  }
}

// A re-decode changes the note list and the synthesized track, and
// nothing else — the recording, the ground truth and the key stay as
// they are, so this deliberately does not go back through loadPlayer.
function applyDecoded(data) {
  currentMeta.notes = data.notes || [];
  currentMeta.decoding = data.decoding;
  currentMeta.midi_audio_url = data.midi_audio_url;
  swapMidiTrack(data.midi_audio_url);
  drawRolls(currentMeta);
  // Only when the staff is engraved from the transcription; where it
  // follows the annotated MIDI it is unaffected by the thresholds, and
  // re-engraving it would be a visible reflow for no change.
  if (notationNotesKey() === 'notes') renderNotationForCurrentKey();
  updatePlayhead(playheadTime(), false);
}

// Point the MIDI <audio> element at the freshly rendered track without
// disturbing the audio graph: a MediaElementAudioSourceNode stays bound
// to its element across a src change, so only the element reloads.
function swapMidiTrack(url) {
  if (!audioMidi) return;
  const t = audioOrig ? audioOrig.currentTime : 0;
  const resume = isPlaying;
  audioMidi.addEventListener('loadedmetadata', () => {
    audioMidi.currentTime = t;
    if (resume) audioMidi.play().catch(() => {});
  }, { once: true });
  audioMidi.src = url;
  audioMidi.load();
}

function teardownAudioGraph() {
  if (audioOrig) { audioOrig.pause(); audioOrig.src = ''; }
  if (audioMidi) { audioMidi.pause(); audioMidi.src = ''; }
  if (sourceOrig) sourceOrig.disconnect();
  if (sourceMidi) sourceMidi.disconnect();
  if (gainOrig) gainOrig.disconnect();
  if (gainMidi) gainMidi.disconnect();
  if (limiter) limiter.disconnect();
}

// Safari's audio-session interruptions (route change via Control Center,
// a phone call, Siri, ...) don't just suspend the AudioContext — they can
// pause the underlying <audio> elements directly, the same way Safari
// pauses background video. When that happens unexpectedly (isPlaying is
// still true — we didn't call pause() ourselves), restart it; otherwise
// the context resumes to 'running' but both tracks just sit paused, which
// looks like normal playback everywhere except no sound comes out.
function resumeIfInterrupted() {
  if (!isPlaying) return;
  if (audioOrig.paused) audioOrig.play().catch(() => {});
  if (audioMidi.paused) audioMidi.play().catch(() => {});
}

function createAudioElements(origUrl, midiUrl) {
  audioOrig = new Audio(origUrl);
  audioMidi = new Audio(midiUrl);
  audioOrig.preload = 'auto';
  audioMidi.preload = 'auto';
  audioOrig.addEventListener('ended', onEnded);
  audioOrig.addEventListener('pause', resumeIfInterrupted);
  audioMidi.addEventListener('pause', resumeIfInterrupted);
}

// Beyond Safari's 'interrupted' state (handled in togglePlay/statechange
// above), Chrome has a long-standing issue where switching the system's
// audio output device (e.g. unplugging headphones, picking a different
// output) silently breaks a graph built from createMediaElementSource —
// audioCtx.state stays 'running' the whole time, so there's no signal to
// resume from and no error; the track just goes quiet. The only reliable
// recovery is to rebuild the graph from scratch, since a
// MediaElementAudioSourceNode is permanently bound to the <audio> element
// it was created from (even a fresh AudioContext can't reuse it) — so the
// element itself has to be recreated too, not just the context.
function recoverAudioGraph() {
  if (!currentMeta || !audioOrig) return;
  const t = audioOrig.currentTime;
  const wasPlaying = isPlaying;
  // Before teardownAudioGraph's pause() calls — see the ordering note in
  // togglePlay above; same race applies here.
  isPlaying = false;

  teardownAudioGraph();
  if (audioCtx) { audioCtx.close(); audioCtx = null; }

  createAudioElements(currentMeta.audio_url, currentMeta.midi_audio_url);
  audioOrig.currentTime = t;
  audioMidi.currentTime = t;

  buildAudioGraph();

  if (wasPlaying) {
    togglePlay();
  }
}

if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
  navigator.mediaDevices.addEventListener('devicechange', () => {
    if (currentMeta) recoverAudioGraph();
  });
}

function buildAudioGraph() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    // macOS/Safari can drop the context into a non-standard 'interrupted'
    // state (distinct from 'suspended') on an audio route change — e.g.
    // unplugging headphones or switching to/from AirPods. <audio>.play()
    // still resolves fine in that state, so playback looks normal but
    // produces no sound. Auto-resume so a route change during playback
    // doesn't need a manual pause/play to recover.
    audioCtx.addEventListener('statechange', () => {
      if (isPlaying && audioCtx.state !== 'running') audioCtx.resume();
    });
  }
  sourceOrig = audioCtx.createMediaElementSource(audioOrig);
  sourceMidi = audioCtx.createMediaElementSource(audioMidi);
  gainOrig = audioCtx.createGain();
  gainMidi = audioCtx.createGain();
  // A limiter, since the equal-power curve + MIDI makeup gain below can
  // otherwise clip when both tracks are loud in the middle of the fade.
  limiter = audioCtx.createDynamicsCompressor();
  limiter.threshold.value = -6;
  limiter.knee.value = 6;
  limiter.ratio.value = 12;
  limiter.attack.value = 0.003;
  limiter.release.value = 0.15;

  sourceOrig.connect(gainOrig).connect(limiter);
  sourceMidi.connect(gainMidi).connect(limiter);
  limiter.connect(audioCtx.destination);
  applyCrossfade();
}

// Equal-power (constant loudness) crossfade instead of a linear one: a
// linear fade spends most of the slider's range with the quiet side well
// below perceived audibility, which is why MIDI used to stay silent until
// the slider was almost all the way over. A small extra boost on top
// helps the synthesized MIDI cut through a full recording.
const MIDI_GAIN_BOOST = 1.4;

function applyCrossfade() {
  if (!gainOrig || !gainMidi || !audioCtx) return;
  const v = parseFloat(crossfadeSlider.value);
  const angle = v * Math.PI / 2;
  // A dragged <input type="range"> fires dozens of 'input' events per
  // second, and setting .value directly makes each one an instantaneous
  // gain jump — audible as clicking/stepping ("zipper noise"), worse the
  // faster you drag. setTargetAtTime instead glides to the new value over
  // ~15ms: far too fast to feel laggy, but enough to stay a smooth curve
  // instead of a series of discontinuities.
  const now = audioCtx.currentTime;
  gainOrig.gain.setTargetAtTime(Math.cos(angle), now, 0.015);
  gainMidi.gain.setTargetAtTime(Math.sin(angle) * MIDI_GAIN_BOOST, now, 0.015);
}

// ---- Single-note preview ----

// Clicking a note plays just that note, as a short synthesized blip.
// Deliberately not a slice of the MIDI render: that would need a second
// seeking <audio> element (or a decoded buffer of the whole track) to
// hear one note, and it would also carry whatever else sounds at that
// moment. A plain oscillator is one note, exactly, and starts instantly.
const PREVIEW_PEAK_GAIN = 0.22;
const PREVIEW_MIN_SECONDS = 0.25;
const PREVIEW_MAX_SECONDS = 1.2;

function playNote(note) {
  if (!note) return;
  // Connected straight to the destination rather than through the
  // crossfade graph, so a preview is audible at any slider position —
  // and so it can work before a track's audio graph exists at all.
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  // A click is a user gesture, so this resume() is allowed; it's a no-op
  // when the context is already running.
  if (audioCtx.state !== 'running') audioCtx.resume().catch(() => {});

  const seconds = Math.min(PREVIEW_MAX_SECONDS,
    Math.max(PREVIEW_MIN_SECONDS, (note.end - note.start) || 0));
  const now = audioCtx.currentTime;

  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type = 'triangle';
  osc.frequency.value = 440 * Math.pow(2, (note.pitch - 69) / 12);
  // Short attack and an exponential decay to (near) zero: a hard
  // start/stop on a bare oscillator is an audible click at both ends.
  gain.gain.setValueAtTime(0.0001, now);
  gain.gain.exponentialRampToValueAtTime(PREVIEW_PEAK_GAIN, now + 0.01);
  gain.gain.exponentialRampToValueAtTime(0.0001, now + seconds);
  osc.connect(gain).connect(audioCtx.destination);
  osc.start(now);
  osc.stop(now + seconds + 0.02);
  // Nothing holds a reference to these nodes; they disconnect themselves
  // once the oscillator has stopped.
  osc.onended = () => { gain.disconnect(); };
}

function onEnded() {
  isPlaying = false;
  playBtn.textContent = '▶';
  if (audioMidi) audioMidi.pause();
}

async function togglePlay() {
  if (!audioOrig) return;
  try {
    // Checking only for 'suspended' misses Safari's 'interrupted' state
    // (see buildAudioGraph) — resume() is a harmless no-op if already
    // running, so just call it whenever we're not already running.
    if (audioCtx.state !== 'running') await audioCtx.resume();
    if (isPlaying) {
      // Set isPlaying false before pausing (not after): pause() queues a
      // 'pause' event that resumeIfInterrupted listens on to detect an
      // *unexpected* pause, and it tells the two apart by checking
      // isPlaying — this ordering keeps a deliberate pause from racing it.
      isPlaying = false;
      audioOrig.pause();
      audioMidi.pause();
    } else {
      audioMidi.currentTime = audioOrig.currentTime;
      await Promise.all([audioOrig.play(), audioMidi.play()]);
      isPlaying = true;
      setStatus('');
    }
    playBtn.textContent = isPlaying ? '⏸' : '▶';
  } catch (err) {
    // Without this, a blocked/failed play() fails completely silently —
    // the button state doesn't update and nothing tells you why there's
    // no sound.
    isPlaying = false;
    playBtn.textContent = '▶';
    setStatus(`Playback failed: ${err.message || err}`, true);
  }
}

function seekTo(t) {
  if (!currentMeta) return;
  t = Math.max(0, Math.min(currentMeta.duration || 0, t));
  audioOrig.currentTime = t;
  audioMidi.currentTime = t;
  seekBar.value = String(t);
  updateTimeLabel(t);
  updatePlayhead(t, false);
}

// ---- Piano roll ----

// "Time context" is how many seconds of the track fit across the visible
// width of the roll/notation containers — i.e. the zoom level, expressed
// in a way that doesn't require knowing pixels-per-second directly. A
// generous total-width cap keeps very long tracks at very tight zoom from
// producing an unreasonably large canvas.
function computePixelsPerSecond(duration) {
  // Measured on the transcription roll specifically: it is the one view
  // that is never hidden (a collapsed panel or an absent ground truth
  // would otherwise measure 0 and silently fall back to the default).
  const containerWidth = predictionRoll.container.clientWidth || 900;
  const desired = containerWidth / timeContextSeconds;
  const maxTotalWidth = 60000;
  const capped = Math.min(desired, maxTotalWidth / Math.max(duration, 1));
  return Math.max(15, capped);
}

// Re-draws the roll and notation at the current zoom/key, keeping
// playback position (used when the time-context or key controls change
// mid-track, not just on initial load).
function redrawTimeline() {
  if (!currentMeta) return;
  pixelsPerSecond = computePixelsPerSecond(currentMeta.duration || 0);
  drawRolls(currentMeta);
  renderNotationForCurrentKey();
  updatePlayhead(playheadTime(), true);
}

// The vertical extent every roll is drawn at. Computed across *all* the
// note lists at once rather than per roll: two rolls stacked for
// comparison are only readable if a given height means the same pitch in
// both, which per-roll auto-fitting would break precisely when the
// transcription and the ground truth disagree about the range.
function pitchRange(noteLists) {
  let minPitch = 108;
  let maxPitch = 21;
  for (const notes of noteLists) {
    for (const n of notes) {
      if (n.pitch < minPitch) minPitch = n.pitch;
      if (n.pitch > maxPitch) maxPitch = n.pitch;
    }
  }
  if (minPitch > maxPitch) { minPitch = 48; maxPitch = 72; }
  return {
    minPitch: Math.max(0, minPitch - 2),
    maxPitch: Math.min(127, maxPitch + 2),
  };
}

function drawRoll(roll, notes, duration, minPitch, maxPitch) {
  const ctx = roll.canvas.getContext('2d');
  const noteHeight = 6;
  const numPitches = maxPitch - minPitch + 1;

  roll.notes = notes;
  roll.minPitch = minPitch;
  roll.maxPitch = maxPitch;
  roll.noteHeight = noteHeight;

  // Same formula as notation.js's width — including the Math.max(200, ...)
  // floor — so every container always ends up with identical
  // scrollWidth (see ROLL_MARGIN/ROLL_END_PADDING comment above).
  const width = Math.max(200, ROLL_MARGIN + Math.round(duration * pixelsPerSecond) + ROLL_END_PADDING);
  const height = numPitches * noteHeight;

  roll.canvas.width = width;
  roll.canvas.height = height;
  roll.container.style.height = Math.min(height, 420) + 'px';

  ctx.fillStyle = '#181a20';
  ctx.fillRect(0, 0, width, height);

  const blackKeys = new Set([1, 3, 6, 8, 10]);
  for (let p = minPitch; p <= maxPitch; p++) {
    if (blackKeys.has(((p % 12) + 12) % 12)) {
      const rowFromTop = maxPitch - p;
      ctx.fillStyle = '#101216';
      ctx.fillRect(0, rowFromTop * noteHeight, width, noteHeight);
    }
  }

  const onsetWidth = Math.max(2, Math.min(4, pixelsPerSecond * 0.04));
  for (const n of notes) {
    const x = ROLL_MARGIN + n.start * pixelsPerSecond;
    const w = Math.max(1.5, (n.end - n.start) * pixelsPerSecond);
    const rowFromTop = maxPitch - n.pitch;
    const y = rowFromTop * noteHeight;

    ctx.fillStyle = roll.colors.body;
    ctx.fillRect(x, y + 0.5, w, noteHeight - 1);

    // Onset marker: a brighter sliver at the note's start.
    ctx.fillStyle = roll.colors.onset;
    ctx.fillRect(x, y + 0.5, Math.min(onsetWidth, w), noteHeight - 1);
  }
}

// The posteriogram PNG is a pixel per (pitch, frame) with the highest
// pitch on row 0, so showing a pitch range is a matter of blitting the
// right slice of rows across the right number of seconds. Scaled with
// smoothing off: at every zoom this app offers, the time axis is being
// stretched rather than squeezed, and interpolating across pitch rows
// would smear neighbouring semitones into each other.
function drawPosteriogram(view, meta, duration, minPitch, maxPitch) {
  const ctx = view.canvas.getContext('2d');
  const noteHeight = 6;
  const width = Math.max(200, ROLL_MARGIN + Math.round(duration * pixelsPerSecond) + ROLL_END_PADDING);
  const height = (maxPitch - minPitch + 1) * noteHeight;

  view.canvas.width = width;
  view.canvas.height = height;
  view.container.style.height = Math.min(height, 420) + 'px';

  // Same floor colour the images are tinted from, so the parts of the
  // pitch range the model doesn't cover are simply more of the same.
  ctx.fillStyle = '#181a20';
  ctx.fillRect(0, 0, width, height);

  const info = meta.posteriograms;
  if (!info || !view.image) return;

  // The model's range and the roll's auto-fitted range need not agree,
  // so clip to the overlap and letterbox whatever is left.
  const topPitch = Math.min(maxPitch, info.max_pitch);
  const bottomPitch = Math.max(minPitch, info.min_pitch);
  if (topPitch < bottomPitch) return;

  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(
    view.image,
    0, info.max_pitch - topPitch, info.n_frames, topPitch - bottomPitch + 1,
    ROLL_MARGIN, (maxPitch - topPitch) * noteHeight,
    info.duration * pixelsPerSecond, (topPitch - bottomPitch + 1) * noteHeight,
  );
}

function drawRolls(meta) {
  const duration = meta.duration || 0;
  const lists = rolls.map((roll) => meta[roll.notesKey] || []);
  // A roll whose field the backend didn't send is hidden rather than
  // drawn empty — an empty roll reads as "the model found nothing here",
  // which is a different claim from "this app has no ground truth".
  const shown = rolls.filter((roll, i) => lists[i].length > 0 || roll === predictionRoll);
  const { minPitch, maxPitch } = pitchRange(rolls.map((r, i) => (shown.includes(r) ? lists[i] : [])));

  rolls.forEach((roll, i) => {
    const visible = shown.includes(roll);
    roll.section.hidden = !visible;
    roll.notes = lists[i];
    if (visible) drawRoll(roll, lists[i], duration, minPitch, maxPitch);
  });

  // Drawn at the rolls' pitch range, not their own, so a given height
  // means the same pitch in every stacked view.
  for (const view of posteriograms) {
    drawPosteriogram(view, meta, duration, minPitch, maxPitch);
  }
}

function findNoteAt(notes, time, pitch) {
  // Notes are sorted by start; scan is cheap even for a few thousand notes
  // at mousemove rates. Prefer the most recently started matching note
  // (top of any stack at that pitch/time).
  let best = null;
  for (const n of notes) {
    if (n.start > time) break;
    if (n.end >= time && n.pitch === pitch) best = n;
  }
  return best;
}

// Which note of `roll` a pointer event is over, or null.
function rollNoteAt(roll, e) {
  const rect = roll.canvas.getBoundingClientRect();
  const time = (e.clientX - rect.left - ROLL_MARGIN) / pixelsPerSecond;
  const pitch = roll.maxPitch - Math.floor((e.clientY - rect.top) / roll.noteHeight);
  return findNoteAt(roll.notes, time, pitch);
}

function updateHoverTooltip(roll, e) {
  if (!currentMeta) return;
  const note = rollNoteAt(roll, e);
  if (!note) {
    hoverTooltip.hidden = true;
    return;
  }
  hoverTooltip.hidden = false;
  hoverTooltip.textContent = `${pitchName(note.pitch)}  ·  ${note.start.toFixed(2)}s–${note.end.toFixed(2)}s`;
  // Positioned in viewport coordinates (not canvas-relative) since the
  // tooltip is `position: fixed` — that's what keeps it from being
  // clipped by the piano roll's scrolling container.
  hoverTooltip.style.left = `${e.clientX}px`;
  hoverTooltip.style.top = `${e.clientY}px`;
}

function updatePlayhead(t, autoScroll) {
  const x = ROLL_MARGIN + t * pixelsPerSecond;
  for (const view of timelineViews) {
    view.playhead.style.left = x + 'px';
    if (autoScroll !== false) {
      view.container.scrollLeft = Math.max(0, x - view.container.clientWidth * 0.3);
    }
  }
  if (window.Notation) Notation.setTime(t, autoScroll);
}

// ---- Playhead calibration ----

// A hand-tuned nudge to how far along the piece the playhead is drawn,
// on top of audioOrig.currentTime. Positive moves it later (further
// right); negative pulls it back; 0 draws it exactly at currentTime.
//
// This is calibration by eye, not a derived quantity — an earlier
// attempt to compute the right value from audioCtx.baseLatency +
// outputLatency was wrong, because currentTime is the media element's
// official playback position and the browser has already accounted for
// output latency in it. Whatever residue is left over (paint timing, and
// whatever the output device adds past what the element knows about) is
// not something the page can read, so it is simply dialled in here.
// Change the number if it doesn't match your setup.
const PLAYHEAD_OFFSET_SECONDS = 0.1;

// Only the drawn playhead is shifted. The seek bar and the mm:ss clock
// keep reporting currentTime, which is the position playback resumes
// from, and where a user-driven seek puts the playhead.
function playheadTime() {
  if (!audioOrig) return 0;
  return Math.max(0, audioOrig.currentTime + PLAYHEAD_OFFSET_SECONDS);
}

// ---- Main animation / sync loop ----

// Two independently-decoding <audio> elements drift apart over time even
// when started together, so the MIDI track's currentTime periodically
// gets hard-corrected back to the original's. 250ms (rather than a
// tighter threshold) keeps that correction rare — frequent tiny
// corrections were the real cost, not the drift itself, since a quarter
// second of misalignment between two takes of the same performance isn't
// perceptible here, but each correction briefly interrupts playback.
const MIDI_DRIFT_THRESHOLD = 0.25;

function resyncMidi(t) {
  if (gainMidi && audioCtx) {
    // Resetting currentTime on a playing <audio> element briefly
    // interrupts its output — inaudible while its gain is near zero, but
    // an audible click once the crossfade slider is toward MIDI. Duck the
    // gain around the seek so that interruption lands in near-silence.
    const now = audioCtx.currentTime;
    const current = gainMidi.gain.value;
    gainMidi.gain.cancelScheduledValues(now);
    gainMidi.gain.setValueAtTime(current, now);
    gainMidi.gain.linearRampToValueAtTime(0, now + 0.03);
    audioMidi.currentTime = t;
    gainMidi.gain.setValueAtTime(0, now + 0.03);
    gainMidi.gain.linearRampToValueAtTime(current, now + 0.09);
  } else {
    audioMidi.currentTime = t;
  }
}

function frameLoop() {
  if (isPlaying && audioOrig) {
    const t = audioOrig.currentTime;
    if (Math.abs(audioMidi.currentTime - t) > MIDI_DRIFT_THRESHOLD) {
      resyncMidi(t);
    }
    if (!isDraggingSeek) {
      seekBar.value = String(t);
      updateTimeLabel(t);
    }
    updatePlayhead(playheadTime(), true);
  }
  requestAnimationFrame(frameLoop);
}

// ---- Event wiring ----

// How far the back button and the arrow keys move.
const SEEK_STEP_SECONDS = 5;

crossfadeSlider.addEventListener('input', applyCrossfade);
playBtn.addEventListener('click', togglePlay);
backBtn.addEventListener('click', () => {
  seekTo((audioOrig ? audioOrig.currentTime : 0) - SEEK_STEP_SECONDS);
});
keyTonicSelect.addEventListener('change', onKeyChanged);
keyModeSelect.addEventListener('change', onKeyChanged);
keyResetBtn.addEventListener('click', resetKeyToDetected);
timeContextSelect.addEventListener('change', () => {
  timeContextSeconds = parseFloat(timeContextSelect.value);
  redrawTimeline();
});

// ---- Collapsible panels ----

// A scrolling view inside a closed <details> is display:none and sits at
// scrollLeft 0 the whole time it's shut; line it back up with the views
// that stayed visible when it reopens.
function restoreScroll(view) {
  const reference = scrollViews.find((other) => other !== view && other.clientWidth > 0);
  if (reference) view.scrollLeft = reference.scrollLeft;
}

function wirePanel(panel, onOpen) {
  if (!panel) return;
  panel.addEventListener('toggle', () => {
    if (!panel.open) return;
    if (onOpen) onOpen();
    scrollViews.filter((view) => panel.contains(view)).forEach(restoreScroll);
  });
}

wirePanel(notationPanel, () => {
  if (notationPending) renderNotationForCurrentKey();
});

wirePanel(modelPanel, () => {
  loadPosteriogramImages();
  // The canvases were last sized against whatever zoom was current when
  // they were painted, which may be several zoom changes ago.
  if (currentMeta) drawRolls(currentMeta);
});

if (hasModelPanel) {
  // 'input' only moves the readout; the actual re-decode hangs off
  // 'change', which a range input fires once the slider is released —
  // exactly the "wait until it's fully moved" the work deserves, with no
  // debounce timer to tune.
  for (const input of [onsetThresholdInput, frameThresholdInput]) {
    input.addEventListener('input', updateThresholdReadouts);
    input.addEventListener('change', runDecode);
  }
  mergeNotesInput.addEventListener('change', runDecode);
}

// Transport keys. Handled on the document so they work wherever the
// focus happens to be — including on the play button right after it was
// clicked, which is exactly when someone reaches for the space bar.
// That case is also why space is handled on keydown and unconditionally
// preventDefault()ed: a focused <button> activates on space *keyup*, so
// without this the key would toggle playback twice, once here and once
// through the button.
document.addEventListener('keydown', (e) => {
  if (!currentMeta) return;
  // Never steal a key that is being typed into a control, or one the
  // browser has its own meaning for.
  const tag = e.target && e.target.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (e.target && e.target.isContentEditable)) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;

  if (e.code === 'Space') {
    e.preventDefault();
    togglePlay();
  } else if (e.key === 'ArrowLeft') {
    e.preventDefault();
    seekTo((audioOrig ? audioOrig.currentTime : 0) - SEEK_STEP_SECONDS);
  } else if (e.key === 'ArrowRight') {
    e.preventDefault();
    seekTo((audioOrig ? audioOrig.currentTime : 0) + SEEK_STEP_SECONDS);
  }
});

seekBar.addEventListener('input', () => {
  isDraggingSeek = true;
  const t = parseFloat(seekBar.value);
  updateTimeLabel(t);
  updatePlayhead(t, false);
});
seekBar.addEventListener('change', () => {
  seekTo(parseFloat(seekBar.value));
  isDraggingSeek = false;
});

for (const roll of rolls) {
  roll.canvas.addEventListener('click', (e) => {
    const x = e.clientX - roll.canvas.getBoundingClientRect().left;
    seekTo((x - ROLL_MARGIN) / pixelsPerSecond);
    // Additive: clicking anywhere still seeks (seekTo doesn't auto-scroll,
    // so the view stays put) — landing on a note also plays it.
    playNote(rollNoteAt(roll, e));
  });
  roll.canvas.addEventListener('mousemove', (e) => updateHoverTooltip(roll, e));
  roll.canvas.addEventListener('mouseleave', () => { hoverTooltip.hidden = true; });
}

// Posteriograms seek like the rolls do, but have no notes to hover or
// play — they're a picture of probabilities, not of note events.
for (const view of posteriograms) {
  view.canvas.addEventListener('click', (e) => {
    const x = e.clientX - view.canvas.getBoundingClientRect().left;
    seekTo((x - ROLL_MARGIN) / pixelsPerSecond);
  });
}

// Dragging one view's scrollbar carries every other view with it. During
// playback updatePlayhead already keeps them together, but while paused
// they would otherwise be left showing different moments of the piece —
// which defeats the point of stacking them. The source guard is what
// stops the echo: assigning scrollLeft fires 'scroll' on the target too,
// and the containers can clamp to slightly different maximums near the
// end of a track, so comparing values alone wouldn't settle.
let scrollSyncSource = null;
for (const view of scrollViews) {
  view.addEventListener('scroll', () => {
    if (scrollSyncSource && scrollSyncSource !== view) return;
    scrollSyncSource = view;
    for (const other of scrollViews) {
      if (other !== view) other.scrollLeft = view.scrollLeft;
    }
    requestAnimationFrame(() => { scrollSyncSource = null; });
  });
}

notationSvgTarget.addEventListener('click', (e) => {
  const svg = notationSvgTarget.querySelector('svg');
  if (!svg) return;
  const rect = svg.getBoundingClientRect();
  const x = e.clientX - rect.left;
  // Same margin/scale as the roll (ROLL_MARGIN === Notation.NOTE_AREA_X)
  // — see the ROLL_MARGIN comment near the top of this file.
  seekTo((x - ROLL_MARGIN) / pixelsPerSecond);
});

// ---- Info drawer ----

// Fetched once at startup (not lazily on first open) so the example
// captions are also available to the search-box suggestions dropdown,
// which can be used before the drawer is ever opened.
let infoPanelHtml = null;
const exampleInfoById = {};

async function loadInfoPanel() {
  try {
    const res = await fetch('/info-panel.html');
    infoPanelHtml = await res.text();
    const temp = document.createElement('div');
    temp.innerHTML = infoPanelHtml;
    temp.querySelectorAll('[data-item-id]').forEach((el) => {
      const itemId = el.getAttribute('data-item-id');
      const infoEl = el.closest('li')?.querySelector('.example-info');
      exampleInfoById[itemId] = {
        title: el.getAttribute('data-title') || itemId,
        // Rendered as real markup in the player caption (lists, timestamp
        // links, etc.); plain text is only for the dropdown's truncated
        // one-line preview, where markup would just get cut mid-tag.
        captionHtml: infoEl ? infoEl.innerHTML.trim() : '',
        captionText: infoEl ? infoEl.textContent.trim() : '',
      };
    });
  } catch (err) {
    infoPanelHtml = '';
  }
}

let infoContentRendered = false;

function renderInfoDrawerContent() {
  infoDrawerContent.innerHTML = infoPanelHtml || 'Could not load info panel.';
  infoContentRendered = true;
  infoDrawerContent.querySelectorAll('[data-item-id]').forEach((el) => {
    el.addEventListener('click', (e) => {
      e.preventDefault();
      const itemId = el.getAttribute('data-item-id');
      const info = exampleInfoById[itemId];
      transcribe(itemId, info ? info.title : el.getAttribute('data-title'), info ? info.captionHtml : undefined);
    });
  });
}

infoDrawerTab.addEventListener('click', async () => {
  const opening = !infoDrawer.classList.contains('open');
  infoDrawer.classList.toggle('open', opening);
  infoDrawerTab.setAttribute('aria-expanded', String(opening));

  if (opening && !infoContentRendered) {
    if (infoPanelHtml == null) await loadInfoPanel();
    renderInfoDrawerContent();
  }
});

loadInfoPanel();
requestAnimationFrame(frameLoop);

// ---- Picker API ----

// The surface a per-app picker script drives. Everything a picker needs
// to start a track and report progress, and nothing about how this file
// draws it.
window.ClassicPitch = {
  transcribe,
  setStatus,
  loadLibrary,
  formatTime,
  escapeHtml,
  picker,
  // Populated from the app's info-panel.html; keyed by item id. Pickers
  // read it to flag curated examples in their own listings.
  exampleInfoById,
  get libraryItems() { return libraryItems; },
};
