// Proportional grand-staff notation, rendered with VexFlow.
//
// Isolated from app.js on purpose: for the file apps this file owns its
// own DOM elements (#notation-container / #notation-svg-target /
// #notation-playhead) and exposes exactly two entry points, `render` and
// `setTime`. It knows nothing about audio, transcription, or the rest of
// the player — it just draws notes and moves a line. It also knows
// nothing about music theory: `render()` takes a ready-made spelling
// table (see key-detection.js's buildSpelling()) and just looks things up
// in it.
//
// The live app engraves the same grand staff but paints its own
// noteheads, since its staff scrolls every frame; `renderStaves()` at the
// bottom is what it shares — the staves, and where a pitch sits on
// them.
//
// This is NOT real rhythmic notation: every note is drawn as a plain
// quarter note and then physically shifted so its x-position is
// proportional to its start time (time * pxPerSecond) — the same
// px-per-second scale the piano roll uses, passed in by the caller, so
// the two stay visually aligned during playback. There's no time
// signature, no bar lines, and no rhythm — it's a scrolling staff-shaped
// timeline, not a score.
//
// Notes are split onto a grand staff at middle C (MIDI 60): pitch >= 60
// goes to the treble stave, pitch < 60 to the bass stave.
//
// The clef + key signature are drawn twice: once on the scrolling staves
// themselves, and once more into a separate, non-scrolling overlay
// (#notation-key-overlay) pinned over the container's left edge, so the
// key signature stays readable no matter how far the staff has scrolled.
// Both are produced by the same VexFlow calls at the same coordinates,
// so at scroll position 0 they land exactly on top of each other.
const Notation = (() => {
  const MIDDLE_C = 60;

  const ACCIDENTAL_GLYPH = { '-2': 'bb', '-1': 'b', 0: 'n', 1: '#', 2: '##' };

  // Given a spelling table (from KeyDetection.buildSpelling) and a MIDI
  // pitch, returns the VexFlow key string (e.g. "f#/4") and whether this
  // specific note needs its own accidental mark — only when its spelling
  // disagrees with what the key signature already implies for that
  // letter (a plain letter with no signature entry counts as "implies
  // natural").
  function spellNote(pitch, spelling) {
    const pc = ((pitch % 12) + 12) % 12;
    const { letter, accidentalOffset } = spelling.pcToSpelling[pc];
    const octave = Math.floor(pitch / 12) - 1;
    const suffix = accidentalOffset === 1 ? '#' : accidentalOffset === -1 ? 'b'
      : accidentalOffset === 2 ? '##' : accidentalOffset === -2 ? 'bb' : '';
    const key = `${letter.toLowerCase()}${suffix}/${octave}`;
    const signatureOffset = spelling.keySignatureAccidentals[letter] || 0;
    const needsAccidental = accidentalOffset !== signatureOffset;
    return { key, needsAccidental, accidentalGlyph: ACCIDENTAL_GLYPH[accidentalOffset] };
  }

  // Fixed left margin reserved for the clef + key signature, shared by
  // every note (both staves) and the playhead. Deliberately NOT using
  // VexFlow's own Stave.getNoteStartX() for this: it varies per clef and
  // per key signature (more sharps/flats = wider), so notes on the two
  // staves — and the playhead, which has neither — would each get a
  // different time-zero x position. One shared constant, generous enough
  // for the widest possible signature (7 accidentals), keeps everything
  // on the exact same time axis regardless of which key is selected.
  const STAVE_X = 10;
  const NOTE_AREA_X = STAVE_X + 140;
  // Distance from the treble stave's y to the bass stave's, i.e. how far
  // apart the two halves of the grand staff sit. One constant because
  // the live app is engraved on the same staff (see renderStaves).
  const STAVE_GAP = 100;
  // The key overlay stops a little short of the note area rather than
  // running right up to it, so its edge doesn't sit flush against the
  // first notes. Only the overlay is trimmed — NOTE_AREA_X itself is the
  // t=0 origin shared with the piano rolls and must not move.
  const OVERLAY_TRIM = 14;
  const OVERLAY_WIDTH = NOTE_AREA_X - OVERLAY_TRIM;

  // Extra breathing room after the last note, so it isn't drawn flush
  // against the right edge. Also shared with app.js: the two views must
  // have identical total widths, or their auto-scroll containers hit
  // their max scrollLeft at different times near the end of the track,
  // and the playheads drift apart right when they're about to converge.
  const END_PADDING = 30;

  let container = null;
  let svgTarget = null;
  let playheadEl = null;
  let overlayEl = null;
  let pxPerSecond = 60;
  let ready = false;

  function init() {
    container = document.getElementById('notation-container');
    svgTarget = document.getElementById('notation-svg-target');
    playheadEl = document.getElementById('notation-playhead');
    // Optional on purpose: without it the key signature simply scrolls
    // away like it used to, rather than the whole staff failing to draw.
    overlayEl = document.getElementById('notation-key-overlay');
    ready = !!(container && svgTarget && playheadEl && window.Vex);
    if (!ready) {
      console.warn('[notation] VexFlow or notation DOM elements not found — skipping.');
    }
    return ready;
  }

  // Draws one stave's worth of notes and returns the {group, start} pairs
  // needed to reposition them afterward. Deliberately does NOT measure or
  // move anything yet — see the comment in render() for why that has to
  // wait a frame.
  function drawStaveVoice(ctx, svgEl, stave, notesForStave, spelling, onNoteClick) {
    const { StaveNote, Accidental, Voice, Formatter } = Vex.Flow;
    if (notesForStave.length === 0) return [];

    const clef = stave.getClef();
    const staveNotes = notesForStave.map((n) => {
      const { key, needsAccidental, accidentalGlyph } = spellNote(n.pitch, spelling);
      const note = new StaveNote({ keys: [key], duration: 'q', clef });
      // No stems. This is a proportional timeline, not rhythmic notation,
      // so a stem says nothing a notehead doesn't already say — it just
      // adds clutter, and at tight zoom its neighbours' stems collide.
      // Done through VexFlow's own Stem.setVisibility (which makes
      // Stem.draw() a no-op) rather than by hiding the drawn SVG
      // afterwards, so nothing here depends on VexFlow's internal class
      // names surviving a version bump.
      const stem = note.getStem();
      if (stem) stem.setVisibility(false);
      // VexFlow positions a note by its letter+accidental in the key
      // string alone — that does NOT draw an accidental sign on its own.
      // Only notes whose spelling disagrees with the key signature (a
      // true chromatic alteration, or a natural canceling it) get one;
      // notes the signature already covers are left unmarked.
      if (needsAccidental) {
        note.addModifier(new Accidental(accidentalGlyph), 0);
      }
      return note;
    });

    const voice = new Voice({ num_beats: staveNotes.length, beat_value: 4 }).setStrict(false);
    voice.addTickables(staveNotes);
    new Formatter().joinVoices([voice]).format([voice], Math.max(50, stave.getWidth() - 40));

    // Let VexFlow draw the notes wherever its rhythmic formatter puts
    // them first — it assumes evenly-spaced quarter notes, which is wrong
    // for our real (non-rhythmic) times. Rather than fight VexFlow's
    // internal tick/formatting APIs for the "correct" x before drawing,
    // we measure where each notehead actually landed afterward (see
    // render()) and move its whole glyph group (notehead + accidental
    // together) to the true time-based x.
    const before = svgEl.querySelectorAll('.vf-stavenote').length;
    voice.draw(ctx, stave);
    const groups = Array.from(svgEl.querySelectorAll('.vf-stavenote')).slice(before);

    return groups.map((group, i) => {
      const note = notesForStave[i];
      if (onNoteClick) {
        group.style.cursor = 'pointer';
        group.addEventListener('click', () => onNoteClick(note));
      }
      return { group, start: note.start };
    });
  }

  // Whether VexFlow can actually draw this key signature. An unusual
  // hand-picked dropdown combination has no table entry (D♯ major would
  // need nine sharps; D♭ minor a double flat), and VexFlow raises that
  // not from addKeySignature() but from the stave's draw() much later —
  // by which point the SVG has been cleared and half-redrawn, so the
  // whole view goes blank. Doing the identical lookup up front instead
  // turns that into "draw the staff, just without a signature".
  function canRenderKeySignature(keySpec) {
    if (!keySpec) return false;
    if (typeof Vex.Flow.keySignature !== 'function') {
      // The lookup helper moved or was renamed in a VexFlow upgrade.
      // Rather than risk the mid-draw failure above, fall back to the
      // signature-less rendering, which is always safe (see the note on
      // effectiveSpelling in render()).
      console.warn('[notation] Vex.Flow.keySignature() unavailable — drawing without key signatures.');
      return false;
    }
    try {
      Vex.Flow.keySignature(keySpec);
      return true;
    } catch (err) {
      console.warn('[notation] no key signature for', keySpec, '— drawing without one.');
      return false;
    }
  }

  // The grand staff itself: two staves, their clefs, and (when one can
  // be drawn at all) the key signature. Both entry points go through
  // this, so the live view's staff and the file apps' are the same
  // object drawn by the same calls.
  function drawGrandStaff(ctx, staveWidth, trebleY, keySpec, endBarType) {
    const { Stave } = Vex.Flow;
    const treble = new Stave(STAVE_X, trebleY, staveWidth).addClef('treble');
    const bass = new Stave(STAVE_X, trebleY + STAVE_GAP, staveWidth).addClef('bass');
    if (keySpec) {
      treble.addKeySignature(keySpec);
      bass.addKeySignature(keySpec);
    }
    if (endBarType !== undefined) {
      treble.setEndBarType(endBarType);
      bass.setEndBarType(endBarType);
    }
    treble.setContext(ctx).draw();
    bass.setContext(ctx).draw();
    return { treble, bass };
  }

  // The non-scrolling copy of the clef + key signature, drawn into an
  // element that sits *outside* the scrolling container and is painted
  // over its left edge. Same staves, same x/y, same VexFlow calls as the
  // real ones, just clipped to the margin width — so it reads as the
  // start of the staff staying put while the music slides underneath it.
  function renderKeyOverlay(keySpec, staveYs, height) {
    if (!overlayEl) return;
    const { Renderer, Stave, Barline } = Vex.Flow;
    overlayEl.innerHTML = '';
    overlayEl.style.width = `${OVERLAY_WIDTH}px`;

    const renderer = new Renderer(overlayEl, Renderer.Backends.SVG);
    // Staves below are still built at the full NOTE_AREA_X geometry so
    // the clef and key signature land exactly where they do on the
    // scrolling staff; the narrower viewport just clips the tail of the
    // staff lines, which is what lets them run into the ones behind.
    renderer.resize(OVERLAY_WIDTH, height);
    const ctx = renderer.getContext();

    staveYs.forEach(({ clef, y }) => {
      const stave = new Stave(STAVE_X, y, NOTE_AREA_X - STAVE_X).addClef(clef);
      if (keySpec) stave.addKeySignature(keySpec);
      // No closing barline: the overlay's staff lines have to run straight
      // into the scrolling ones behind it, not look like a bar end.
      stave.setEndBarType(Barline.type.NONE);
      stave.setContext(ctx).draw();
    });
  }

  // `spelling` is a KeyDetection.buildSpelling() result; `keySpec` is a
  // KeyDetection.keySpecString() result (e.g. "Eb", "F#m") used only for
  // the visual key-signature glyphs at the start of each stave.
  // `onNoteClick`, if given, is called with the source note object when
  // its notehead is clicked.
  function render(notes, duration, scale, spelling, keySpec, onNoteClick) {
    if (!ready && !init()) return;

    // When no signature can be drawn, nothing on the staff implies any
    // alteration — so every altered note has to carry its own accidental.
    // Dropping the signature entries from the spelling is exactly what
    // makes spellNote() mark them all.
    const drawSignature = canRenderKeySignature(keySpec);
    const signatureSpec = drawSignature ? keySpec : null;
    const effectiveSpelling = drawSignature
      ? spelling
      : Object.assign({}, spelling, { keySignatureAccidentals: {} });

    pxPerSecond = scale > 0 ? scale : 60;
    svgTarget.innerHTML = '';

    const { Renderer } = Vex.Flow;

    const width = Math.max(200, NOTE_AREA_X + Math.round(duration * pxPerSecond) + END_PADDING);
    const staveWidth = width - STAVE_X - 10;
    const trebleY = 10;
    const bassY = trebleY + STAVE_GAP;
    const height = bassY + 100;

    const renderer = new Renderer(svgTarget, Renderer.Backends.SVG);
    renderer.resize(width, height);
    const ctx = renderer.getContext();

    const { treble: trebleStave, bass: bassStave } =
      drawGrandStaff(ctx, staveWidth, trebleY, signatureSpec);

    const trebleNotes = notes.filter((n) => n.pitch >= MIDDLE_C);
    const bassNotes = notes.filter((n) => n.pitch < MIDDLE_C);

    const svgEl = svgTarget.querySelector('svg');
    const placed = [
      ...drawStaveVoice(ctx, svgEl, trebleStave, trebleNotes, effectiveSpelling, onNoteClick),
      ...drawStaveVoice(ctx, svgEl, bassStave, bassNotes, effectiveSpelling, onNoteClick),
    ];

    // getBBox() on SVG geometry created earlier in this same tick can
    // return stale/zero results in Chromium (confirmed empirically: even
    // forcing a synchronous reflow via getBoundingClientRect() wasn't
    // enough) — so the actual repositioning has to happen on the next
    // frame, once the browser has genuinely finished with this batch.
    requestAnimationFrame(() => {
      placed.forEach(({ group, start }) => {
        const targetX = NOTE_AREA_X + start * pxPerSecond;
        const headEl = group.querySelector('.vf-notehead') || group;
        const dx = targetX - headEl.getBBox().x;
        group.setAttribute('transform', `translate(${dx}, 0)`);
      });
    });

    renderKeyOverlay(signatureSpec, [{ clef: 'treble', y: trebleY }, { clef: 'bass', y: bassY }], height);

    container.style.height = (height + 10) + 'px';
    setTime(0, false);
  }

  function setTime(t, autoScroll) {
    if (!ready) return;
    const x = NOTE_AREA_X + t * pxPerSecond;
    playheadEl.style.left = `${x}px`;
    if (autoScroll !== false) {
      container.scrollLeft = Math.max(0, x - container.clientWidth * 0.3);
    }
  }

  // ---- The live view's staff ----
  //
  // The live app draws its own noteheads, onto a canvas: its staff
  // scrolls every frame, and re-engraving a VexFlow voice sixty times a
  // second is out of the question. What it must not do is invent its own
  // staff — so it gets these staves, drawn by the same calls render()
  // makes, and a place() that says where a pitch sits on them.

  // Diatonic degree: one per letter name, seven per octave, so that
  // everything on a staff is evenly spaced (a line or space is one step,
  // which is half a staff space in pixels) regardless of accidentals.
  const LETTER_DEGREE = { C: 0, D: 1, E: 2, F: 3, G: 4, A: 5, B: 6 };
  const TOP_LINE_DEGREE = {
    treble: LETTER_DEGREE.F + 7 * 5,   // F5
    bass: LETTER_DEGREE.A + 7 * 3,     // A3
  };
  // Half a notehead plus a little, so a note at the very top or bottom of
  // the reserved range isn't clipped by the edge of the drawing.
  const NOTEHEAD_MARGIN = 8;

  // Draws the empty grand staff into `target` and returns the geometry
  // needed to paint notes onto it.
  //
  // `stepsAbove`/`stepsBelow` are how much ledger-line room to keep above
  // the treble stave's top line and below the bass stave's bottom one, in
  // diatonic steps — the caller's pitch range, in other words. They decide
  // the `height` that comes back.
  function renderStaves(target, { width, spelling, keySpec, stepsAbove, stepsBelow }) {
    if (!window.Vex) {
      console.warn('[notation] VexFlow not found — skipping the staff.');
      return null;
    }
    const { Renderer, Stave } = Vex.Flow;

    // Same fallback render() makes: with no signature to imply anything,
    // every altered note has to carry its own accidental, which dropping
    // the signature entries from the spelling is exactly what produces.
    const drawSignature = canRenderKeySignature(keySpec);
    const signatureSpec = drawSignature ? keySpec : null;
    const effectiveSpelling = drawSignature
      ? spelling
      : Object.assign({}, spelling, { keySignatureAccidentals: {} });

    // Out to the right edge, and with no barline there: the staff does
    // not end, it is just as long as the window is wide.
    const staveWidth = Math.max(120, Math.round(width) - STAVE_X);

    // How far below the y it is built at VexFlow puts a stave's first
    // line is its own business (it reserves room above the staff), so ask
    // a stave rather than assume. Geometry only — a Stave knows where its
    // lines fall without ever being drawn, so this costs nothing and
    // paints nothing.
    const probe = new Stave(STAVE_X, 0, staveWidth);
    const lineOffset = probe.getYForLine(0);
    const spacing = probe.getYForLine(1) - probe.getYForLine(0);

    const trebleY = Math.max(0, Math.round(stepsAbove * (spacing / 2) + NOTEHEAD_MARGIN - lineOffset));
    const bassBottom = trebleY + lineOffset + STAVE_GAP + 4 * spacing;
    const height = Math.round(bassBottom + stepsBelow * (spacing / 2) + NOTEHEAD_MARGIN);

    target.innerHTML = '';
    const renderer = new Renderer(target, Renderer.Backends.SVG);
    renderer.resize(Math.round(width), height);
    const staves = drawGrandStaff(renderer.getContext(), staveWidth, trebleY,
      signatureSpec, Vex.Flow.Barline.type.NONE);

    const topY = { treble: staves.treble.getYForLine(0), bass: staves.bass.getYForLine(0) };

    // Where a pitch's notehead goes, the ledger lines it needs to get
    // there, and the accidental it has to carry — all in pixels of the
    // staff just drawn.
    function place(pitch) {
      const pc = ((pitch % 12) + 12) % 12;
      const { letter, accidentalOffset } = effectiveSpelling.pcToSpelling[pc];
      // The octave belongs to the spelled letter's natural pitch, not the
      // sounding one — a C flat is written on the C line above the B it
      // sounds.
      const octave = Math.floor((pitch - accidentalOffset) / 12) - 1;
      const degree = LETTER_DEGREE[letter] + 7 * octave;
      // The same split render() uses, so a note lands on the same stave
      // in both views.
      const clef = pitch >= MIDDLE_C ? 'treble' : 'bass';
      const topDegree = TOP_LINE_DEGREE[clef];
      const yFor = (d) => topY[clef] + (topDegree - d) * (spacing / 2);

      // Ledger lines sit where staff lines would have gone on — every
      // second degree — from just outside the stave out to the note.
      const ledgers = [];
      for (let d = topDegree + 2; d <= degree; d += 2) ledgers.push(yFor(d));
      for (let d = topDegree - 10; d >= degree; d -= 2) ledgers.push(yFor(d));

      const signatureOffset = effectiveSpelling.keySignatureAccidentals[letter] || 0;
      return {
        y: yFor(degree),
        ledgers,
        // An accidental is printed only where the note disagrees with the
        // signature; that disagreement is exactly what one means.
        accidental: accidentalOffset === signatureOffset ? null : accidentalOffset,
      };
    }

    return {
      height,
      spacing,
      // Where notes may start: everything left of it is clef and key
      // signature. The live view clips its noteheads to this instead of
      // drawing the second, opaque copy render() needs, because its staff
      // never scrolls — only the notes on it do.
      noteStartX: Math.max(staves.treble.getNoteStartX(), staves.bass.getNoteStartX()),
      place,
    };
  }

  // NOTE_AREA_X and END_PADDING are exposed so app.js can shift the piano
  // roll's own content by the same amount — see the comment on
  // NOTE_AREA_X above for why the two views otherwise have different left
  // margins.
  return { render, setTime, renderStaves, NOTE_AREA_X, END_PADDING };
})();

// Explicit assignment: a top-level `const` does not become a `window`
// property on its own, and app.js checks `window.Notation` to stay
// robust if this script or VexFlow ever fails to load.
window.Notation = Notation;
