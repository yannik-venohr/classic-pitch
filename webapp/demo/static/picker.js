// How the YouTube demo chooses a track: a search box that takes a query
// or a pasted URL, with a dropdown of already-transcribed items under it.
// The player itself lives in /shared/app.js — see its "Picker API" block.

(function () {
  const { transcribe, setStatus, loadLibrary, formatTime, escapeHtml, picker, exampleInfoById } = window.ClassicPitch;

  const searchInput = document.getElementById('search-input');
  const searchBtn = document.getElementById('search-btn');
  const resultsEl = document.getElementById('results');
  const librarySuggestions = document.getElementById('library-suggestions');

  let libraryItems = [];

  picker.clearResults = () => {
    resultsEl.innerHTML = '';
    librarySuggestions.hidden = true;
    // Hand focus back to the page once a track is on its way. Picking a
    // suggestion leaves the caret in the search box, and the transport
    // keys deliberately don't fire while a text field has focus — so
    // without this the first space bar press after loading a track types
    // a space instead of starting playback.
    searchInput.blur();
  };
  picker.setBusy = (busy) => { searchBtn.disabled = busy; };
  picker.onLibrary = (items) => { libraryItems = items; };

  function isProbablyUrl(text) {
    return /^https?:\/\//i.test(text.trim());
  }

  async function doSearch() {
    const q = searchInput.value.trim();
    if (!q) return;
    setStatus('Searching…');
    resultsEl.innerHTML = '';
    searchBtn.disabled = true;
    try {
      const res = await fetch('/api/search?q=' + encodeURIComponent(q));
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'search failed');
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
      setStatus(String(err.message || err), true);
    } finally {
      searchBtn.disabled = false;
    }
  }

  function renderResults(results) {
    resultsEl.innerHTML = '';
    for (const r of results) {
      const item = document.createElement('div');
      item.className = 'result-item';
      const durText = r.duration ? formatTime(r.duration) : '';
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

  function renderLibrarySuggestions() {
    const q = searchInput.value.trim().toLowerCase();
    const matches = q
      ? libraryItems.filter((it) => (it.title || '').toLowerCase().includes(q))
      : libraryItems;

    if (matches.length === 0) {
      librarySuggestions.hidden = true;
      librarySuggestions.innerHTML = '';
      return;
    }

    librarySuggestions.innerHTML = '';
    for (const it of matches.slice(0, 8)) {
      const example = exampleInfoById[it.item_id];
      const el = document.createElement('div');
      el.className = 'suggestion-item';
      const durText = it.duration ? formatTime(it.duration) : '';
      const metaLine = example && example.captionText
        ? `<div class="suggestion-meta suggestion-example">★ example — ${escapeHtml(truncate(example.captionText, 80))}</div>`
        : `<div class="suggestion-meta">already transcribed${durText ? ' · ' + durText : ''}</div>`;
      el.innerHTML = `<div>${escapeHtml(it.title || it.item_id)}</div>${metaLine}`;
      // mousedown (not click) fires before the input's blur hides the list.
      el.addEventListener('mousedown', (e) => {
        e.preventDefault();
        librarySuggestions.hidden = true;
        transcribe(it.item_id, it.title, example ? example.captionHtml : undefined);
      });
      librarySuggestions.appendChild(el);
    }
    librarySuggestions.hidden = false;
  }

  searchBtn.addEventListener('click', doSearch);
  searchInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') doSearch(); });
  searchInput.addEventListener('focus', renderLibrarySuggestions);
  searchInput.addEventListener('input', renderLibrarySuggestions);
  searchInput.addEventListener('blur', () => {
    // Delayed so a mousedown on a suggestion registers before it's hidden.
    setTimeout(() => { librarySuggestions.hidden = true; }, 150);
  });

  loadLibrary();
})();
