// Key detection and key theory.
//
// Isolated on purpose (same pattern as notation.js): pure music-theory
// functions — detecting a key from notes, and turning a chosen key into a
// diatonic spelling table. Knows nothing about rendering, audio, or the
// DOM; notation.js consumes what this produces, it doesn't compute any of
// it itself.
const KeyDetection = (() => {
  // ---- Key detection (Krumhansl-Schmuckler) ----
  //
  // Build a duration-weighted pitch-class histogram (12 bins, one per
  // semitone regardless of octave), then correlate it against the
  // Krumhansl-Kessler major/minor key profiles at all 12 rotations each
  // (24 total) and pick the best-correlated match. Standard, well-known
  // algorithm — not a custom heuristic.
  const MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
  const MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

  // ---- Shared key-theory building blocks ----
  const LETTER_NATURAL_PC = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
  const LETTER_CYCLE = ['C', 'D', 'E', 'F', 'G', 'A', 'B'];
  const MAJOR_SCALE_STEPS = [0, 2, 4, 5, 7, 9, 11];
  const MINOR_SCALE_STEPS = [0, 2, 3, 5, 7, 8, 10]; // natural minor

  // Major keys conventionally written with flats (circle of fifths, flat
  // side). Everything else — including the sharp/flat-ambiguous F#/Gb,
  // resolved to F# here — defaults to sharps. A minor key follows its
  // relative major's accidentals (3 semitones up), handled by callers.
  const FLAT_MAJOR_PITCH_CLASSES = new Set([1, 3, 5, 8, 10]); // Db, Eb, F, Ab, Bb

  function correlate(a, b) {
    const n = a.length;
    const meanA = a.reduce((s, v) => s + v, 0) / n;
    const meanB = b.reduce((s, v) => s + v, 0) / n;
    let num = 0;
    let denA = 0;
    let denB = 0;
    for (let i = 0; i < n; i++) {
      const da = a[i] - meanA;
      const db = b[i] - meanB;
      num += da * db;
      denA += da * da;
      denB += db * db;
    }
    const den = Math.sqrt(denA * denB);
    return den > 0 ? num / den : 0;
  }

  function pitchClassFromLetterAccidental(letter, accidental) {
    const base = LETTER_NATURAL_PC[letter];
    const offset = accidental === 'sharp' ? 1 : accidental === 'flat' ? -1 : 0;
    return ((base + offset) % 12 + 12) % 12;
  }

  function prefersFlatsFor(letter, accidental, mode) {
    const tonicPc = pitchClassFromLetterAccidental(letter, accidental);
    const effectivePc = mode === 'minor' ? (tonicPc + 3) % 12 : tonicPc;
    return FLAT_MAJOR_PITCH_CLASSES.has(effectivePc);
  }

  // VexFlow key-signature spec string, e.g. "Eb", "F#m", "C", "Am".
  function keySpecString(letter, accidental, mode) {
    const accChar = accidental === 'flat' ? 'b' : accidental === 'sharp' ? '#' : '';
    return `${letter}${accChar}${mode === 'minor' ? 'm' : ''}`;
  }

  // Builds a full diatonic spelling for a key: which letter (and how many
  // sharps/flats off its natural pitch) each of the 12 pitch classes gets,
  // plus which letters the key SIGNATURE itself already alters — the two
  // together tell a caller both what to call a note and whether it needs
  // its own explicit accidental mark (only when it disagrees with the
  // signature) or none (when the signature already covers it).
  //
  // Method: place the 7 diatonic scale degrees on consecutive letters
  // starting at the tonic (the letter alphabet always cycles C-D-E-F-G-A-B
  // in scale order — that's just how staff notation works), each getting
  // whatever accidental reaches its actual pitch class from that letter's
  // natural one. The 5 remaining (chromatic) pitch classes each sit
  // exactly a half-step from two diatonic neighbors (a property of the
  // 7-note major/minor scale) — spelled as a sharp of the one below, or a
  // flat of the one above, depending on the key's sharp/flat preference.
  function buildSpelling(letter, accidental, mode) {
    const tonicPc = pitchClassFromLetterAccidental(letter, accidental);
    const steps = mode === 'minor' ? MINOR_SCALE_STEPS : MAJOR_SCALE_STEPS;
    const startLetterIndex = LETTER_CYCLE.indexOf(letter);
    const prefersFlats = prefersFlatsFor(letter, accidental, mode);

    // Key-signature accidentals come ONLY from these 7 true diatonic
    // degrees — computed here, before the chromatic fill-in below can
    // introduce a second (unrelated) spelling that happens to reuse the
    // same letter (e.g. a chromatic D# alongside a diatonic natural D)
    // and would otherwise overwrite the real signature entry for it.
    const pcToSpelling = {};
    const keySignatureAccidentals = {};
    for (let degree = 0; degree < 7; degree++) {
      const pc = (tonicPc + steps[degree]) % 12;
      const degreeLetter = LETTER_CYCLE[(startLetterIndex + degree) % 7];
      const naturalPc = LETTER_NATURAL_PC[degreeLetter];
      let offset = pc - naturalPc;
      while (offset > 6) offset -= 12;
      while (offset < -6) offset += 12;
      pcToSpelling[pc] = { letter: degreeLetter, accidentalOffset: offset };
      if (offset !== 0) keySignatureAccidentals[degreeLetter] = offset;
    }

    for (let pc = 0; pc < 12; pc++) {
      if (pcToSpelling[pc]) continue;
      const below = pcToSpelling[(pc - 1 + 12) % 12]; // always diatonic — see comment above
      const above = pcToSpelling[(pc + 1) % 12]; // always diatonic
      const sharpOfBelow = { letter: below.letter, accidentalOffset: below.accidentalOffset + 1 };
      const flatOfAbove = { letter: above.letter, accidentalOffset: above.accidentalOffset - 1 };
      // Usually both options land on a single sharp/flat (e.g. F# vs Gb)
      // and the key's preference breaks the tie. But when this chromatic
      // pitch is exactly a signature accidental undone — e.g. B-natural
      // in Eb major, where Bb is in the signature — "sharp of Bb" lands
      // on a plain natural (offset 0) while "flat of C" would land on the
      // unidiomatic Cb (offset -1). Simpler spelling wins regardless of
      // the key's general preference: a natural sign reading as "the
      // letter you'd expect, undone" beats a correct-but-confusing
      // enharmonic respelling.
      const absSharp = Math.abs(sharpOfBelow.accidentalOffset);
      const absFlat = Math.abs(flatOfAbove.accidentalOffset);
      if (absSharp !== absFlat) {
        pcToSpelling[pc] = absSharp < absFlat ? sharpOfBelow : flatOfAbove;
      } else {
        pcToSpelling[pc] = prefersFlats ? flatOfAbove : sharpOfBelow;
      }
    }

    return { pcToSpelling, keySignatureAccidentals, prefersFlats };
  }

  function detect(notes) {
    const histogram = new Array(12).fill(0);
    for (const n of notes) {
      const duration = Math.max(0, n.end - n.start);
      histogram[((n.pitch % 12) + 12) % 12] += duration;
    }

    let best = { score: -Infinity, tonicPc: 0, mode: 'major' };
    for (const [mode, profile] of [['major', MAJOR_PROFILE], ['minor', MINOR_PROFILE]]) {
      for (let pc = 0; pc < 12; pc++) {
        const shifted = new Array(12);
        for (let i = 0; i < 12; i++) shifted[i] = profile[(((i - pc) % 12) + 12) % 12];
        const score = correlate(histogram, shifted);
        if (score > best.score) best = { score, tonicPc: pc, mode };
      }
    }

    const effectivePc = best.mode === 'minor' ? (best.tonicPc + 3) % 12 : best.tonicPc;
    const prefersFlats = FLAT_MAJOR_PITCH_CLASSES.has(effectivePc);
    const name = (prefersFlats
      ? ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
      : ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])[best.tonicPc];
    const letter = name[0];
    const accidental = name.includes('#') ? 'sharp' : name.includes('b') ? 'flat' : 'natural';

    return {
      tonicPc: best.tonicPc,
      mode: best.mode,
      letter,
      accidental,
      prefersFlats,
    };
  }

  return { detect, buildSpelling, keySpecString, prefersFlatsFor };
})();

// Explicit assignment: a top-level `const` does not become a `window`
// property on its own (see the same note in notation.js).
window.KeyDetection = KeyDetection;
