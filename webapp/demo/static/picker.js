// How the YouTube demo chooses a track: a search box that takes a query
// or a pasted URL, with a dropdown of already-transcribed items under it.
// The player itself lives in /shared/app.js — see its "Picker API" block.

(function () {
  const { transcribe, setStatus, loadLibrary, formatTime, escapeHtml, picker, exampleInfoById, currentMeta } = window.ClassicPitch;

  const searchInput = document.getElementById('search-input');
  const resultsEl = document.getElementById('results');
  const librarySuggestions = document.getElementById('library-suggestions');
  const checkpointSelect = document.getElementById('checkpoint-select');
  const uploadBtn = document.getElementById('upload-btn');
  const uploadInput = document.getElementById('upload-input');

  let libraryItems = [];

  // A transcription is cached per recording *and* model, under
  // "<recording>~<model>" (see ITEM_ID_RE in app.py).
  const cacheId = (recordingId) => `${recordingId}~${checkpointSelect.value}`;
  picker.itemId = cacheId;

  picker.clearResults = () => {
    // A pending or in-flight search must not paint results over a track
    // that is being loaded.
    clearTimeout(searchTimer);
    searchSeq++;
    resultsEl.innerHTML = '';
    librarySuggestions.hidden = true;
    // Hand focus back to the page once a track is on its way. Picking a
    // suggestion leaves the caret in the search box, and the transport
    // keys deliberately don't fire while a text field has focus — so
    // without this the first space bar press after loading a track types
    // a space instead of starting playback.
    searchInput.blur();
  };
  picker.setBusy = (busy) => {
    searchInput.disabled = busy;
    uploadBtn.disabled = busy;
    checkpointSelect.disabled = busy;
  };
  picker.onLibrary = (items) => { libraryItems = items; };

  function isProbablyUrl(text) {
    return /^https?:\/\//i.test(text.trim());
  }

  // Searches run on their own once typing pauses. Responses can arrive
  // out of order, so only the latest one is allowed to show.
  let searchSeq = 0;
  let searchTimer = null;
  const SEARCH_DEBOUNCE_MS = 700;
  const MIN_QUERY_LENGTH = 3;

  // A URL is only worth sending once it holds a video id, otherwise a
  // half-typed one would flash an error; Enter still sends whatever is there.
  function looksComplete(q) {
    if (isProbablyUrl(q)) return /(?:v=|\/videos\/|embed\/|youtu\.be\/|\/shorts\/|\/live\/)[\w-]{11}/.test(q);
    return q.length >= MIN_QUERY_LENGTH;
  }

  function scheduleSearch() {
    clearTimeout(searchTimer);
    const q = searchInput.value.trim();
    if (!looksComplete(q)) {
      searchSeq++;
      resultsEl.innerHTML = '';
      setStatus('');
      return;
    }
    searchTimer = setTimeout(doSearch, SEARCH_DEBOUNCE_MS);
  }

  async function doSearch() {
    clearTimeout(searchTimer);
    const q = searchInput.value.trim();
    if (!q) return;
    const seq = ++searchSeq;
    setStatus('Searching…');
    try {
      const res = await fetch('/api/search?q=' + encodeURIComponent(q));
      const data = await res.json();
      if (seq !== searchSeq) return;
      if (!res.ok) throw new Error(data.error || 'search failed');
      resultsEl.innerHTML = '';
      const results = data.results || [];
      if (results.length === 0) {
        setStatus('No results found.', true);
      } else if (isProbablyUrl(q) && results.length === 1) {
        setStatus('');
        transcribe(results[0].id, results[0].title);
      } else {
        setStatus('');
        renderResults(results);
      }
    } catch (err) {
      if (seq === searchSeq) setStatus(String(err.message || err), true);
    }
  }

  function renderResults(results) {
    resultsEl.innerHTML = '';
    for (const r of results) {
      const item = document.createElement('div');
      item.className = 'result-item';
      // Only the start of a long video is transcribed, so say so up front.
      const truncated = r.duration && r.transcribed_seconds && r.duration > r.transcribed_seconds + 1;
      const durText = r.duration
        ? formatTime(r.duration) + (truncated ? ` (first ${formatTime(r.transcribed_seconds)} transcribed)` : '')
        : '';
      item.innerHTML = `
        ${r.thumbnail ? `<img src="${r.thumbnail}" alt="">` : ''}
        <div>
          <div class="result-title">${escapeHtml(r.title || r.id)}</div>
          <div class="result-meta">${escapeHtml(r.uploader || '')}${durText ? ' · ' + durText : ''}</div>
        </div>
      `;
      item.addEventListener('click', () => transcribe(r.id, r.title));
      resultsEl.appendChild(item);
    }
  }

  function truncate(s, n) {
    if (!s) return '';
    return s.length > n ? s.slice(0, n - 1).trimEnd() + '…' : s;
  }

  const RECENT_COUNT = 3;

  // What the box offers: the transcriptions already made with the
  // selected model (newest first — that is the order the library comes
  // in), and the curated examples from the info panel, which are offered
  // whether or not they have been transcribed with this model yet.
  // Untouched, that is the last three transcriptions followed by the
  // examples; typing filters both by title.
  function suggestionEntries() {
    const q = searchInput.value.trim().toLowerCase();
    const suffix = `~${checkpointSelect.value}`;
    const matchesQuery = (title) => !q || (title || '').toLowerCase().includes(q);

    const cached = libraryItems
      .filter((it) => it.item_id.endsWith(suffix) && matchesQuery(it.title))
      .map((it) => ({
        recordingId: it.item_id.slice(0, -suffix.length),
        title: it.title,
        duration: it.duration,
        cached: true,
      }));
    const entries = q ? cached : cached.slice(0, RECENT_COUNT);

    const seen = new Set(entries.map((e) => e.recordingId));
    for (const [recordingId, example] of Object.entries(exampleInfoById)) {
      if (seen.has(recordingId) || !matchesQuery(example.title)) continue;
      entries.push({ recordingId, title: example.title, cached: false });
    }
    return entries;
  }

  function renderLibrarySuggestions() {
    const entries = suggestionEntries();

    if (entries.length === 0) {
      librarySuggestions.hidden = true;
      librarySuggestions.innerHTML = '';
      return;
    }

    librarySuggestions.innerHTML = '';
    for (const entry of entries) {
      const example = exampleInfoById[entry.recordingId];
      const el = document.createElement('div');
      el.className = 'suggestion-item';
      const durText = entry.duration ? formatTime(entry.duration) : '';
      const status = entry.cached ? `already transcribed${durText ? ' · ' + durText : ''}` : '';
      const metaLine = example && example.captionText
        ? `<div class="suggestion-meta suggestion-example">★ example — ${escapeHtml(truncate(example.captionText, 80))}</div>`
        : `<div class="suggestion-meta">${status}</div>`;
      el.innerHTML = `<div>${escapeHtml(entry.title || entry.recordingId)}</div>${metaLine}`;
      // mousedown (not click) fires before the input's blur hides the list.
      el.addEventListener('mousedown', (e) => {
        e.preventDefault();
        librarySuggestions.hidden = true;
        transcribe(entry.recordingId, entry.title, example ? example.captionHtml : undefined);
      });
      librarySuggestions.appendChild(el);
    }
    librarySuggestions.hidden = false;
  }

  async function uploadFile(file) {
    setStatus(`Reading "${file.name}"…`, false, true);
    picker.setBusy(true);
    try {
      const body = new FormData();
      body.append('file', file);
      const res = await fetch('/api/upload', { method: 'POST', body });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'upload failed');
      setStatus('');
      picker.setBusy(false);
      transcribe(data.id, data.title);
    } catch (err) {
      picker.setBusy(false);
      setStatus(String(err.message || err), true);
    }
  }

  // Changing the model re-runs whatever is loaded with the new one (a
  // transcription is cached per model, so going back is instant).
  checkpointSelect.addEventListener('change', () => {
    const meta = currentMeta();
    if (!meta) return;
    const recordingId = meta.item_id.split('~')[0];
    const example = exampleInfoById[recordingId];
    transcribe(recordingId, meta.title, example ? example.captionHtml : undefined);
  });

  uploadBtn.addEventListener('click', () => uploadInput.click());
  uploadInput.addEventListener('change', () => {
    const file = uploadInput.files[0];
    uploadInput.value = '';
    if (file) uploadFile(file);
  });

  async function loadCheckpoints() {
    try {
      const res = await fetch('/api/models');
      const data = await res.json();
      checkpointSelect.innerHTML = '';
      for (const c of data.checkpoints || []) {
        const opt = document.createElement('option');
        opt.value = c.id;
        opt.textContent = c.label;
        checkpointSelect.appendChild(opt);
      }
      if ((data.checkpoints || []).some((c) => c.id === data.default_checkpoint)) {
        checkpointSelect.value = data.default_checkpoint;
      }
    } catch (err) {
      // Left empty; the status below says so.
    }
    if (checkpointSelect.options.length === 0) {
      setStatus('No checkpoints found in checkpoints/.', true);
      searchInput.disabled = true;
      uploadBtn.disabled = true;
    }
  }

  searchInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') doSearch(); });
  searchInput.addEventListener('focus', renderLibrarySuggestions);
  searchInput.addEventListener('input', () => { renderLibrarySuggestions(); scheduleSearch(); });
  searchInput.addEventListener('blur', () => {
    // Delayed so a mousedown on a suggestion registers before it's hidden.
    setTimeout(() => { librarySuggestions.hidden = true; }, 150);
  });

  loadCheckpoints();
  loadLibrary();
})();
