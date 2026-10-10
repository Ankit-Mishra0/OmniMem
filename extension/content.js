/**
 * OmniMem content script.
 *
 * Deliberately thin. It knows how to read the DOM (via platforms.js), how to
 * strip secrets (via redact.js) and how to talk to the background worker. It
 * holds no token, makes no network calls, and contains no site selectors.
 *
 * Capture covers *both* roles. Assistant-only capture was the original design's
 * central mistake: what the user is trying to achieve is stated in the user's
 * own turns, and that is precisely the thing another tool needs in order to
 * continue the work.
 */

(function () {
  'use strict';

  var DEBOUNCE_MS = 600;
  var MIN_CHARS = 12;        // below this a turn cannot carry a usable fact
  var MAX_CHARS = 20000;     // don't ship an entire pasted repo

  // The brief carries this marker (render.py BRIEF_HEADER). Any turn containing
  // it is text we injected ourselves; capturing it would feed the store its own
  // output and inflate every fact's sighting count on each new chat.
  var ECHO_MARKERS = ['OmniMem brief', 'OmniMem session summary', '```omnimem'];

  var adapter = window.OmniMemPlatforms.forHost(location.hostname);
  var redact = window.OmniMemRedact.redact;
  var wouldRedact = window.OmniMemRedact.wouldRedact;

  var settings = { capture: false, redact: true, project: '', online: false };
  var sent = new Set();          // hashes of turns already shipped, this page
  var prevTail = '';             // last tick's text for the final turn
  var sessionId = makeSessionId();
  var timer = null;
  var settleTimer = null;
  var observer = null;

  // ------------------------------------------------------------------ util

  function send(msg) {
    return new Promise(function (resolve) {
      try {
        chrome.runtime.sendMessage(msg, function (reply) {
          // Reading lastError suppresses the "unchecked runtime.lastError"
          // console noise when the worker is asleep or was just reloaded.
          void chrome.runtime.lastError;
          resolve(reply || { ok: false });
        });
      } catch (_) {
        resolve({ ok: false });   // extension context invalidated (reload)
      }
    });
  }

  function hash32(s) {
    var h = 0x811c9dc5;
    for (var i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = (h * 0x01000193) >>> 0;
    }
    return h.toString(36);
  }

  function makeSessionId() {
    var conv = '';
    try { conv = adapter.session(location) || ''; } catch (_) { conv = ''; }
    return adapter.id + ':' + (conv || 'tab-' + hash32(location.pathname + document.title));
  }

  function firstMatch(selectors) {
    for (var i = 0; i < selectors.length; i++) {
      var found;
      try { found = document.querySelectorAll(selectors[i]); } catch (_) { continue; }
      if (found && found.length) return Array.prototype.slice.call(found);
    }
    return [];
  }

  function textOf(el) {
    var raw = el.innerText || el.textContent || '';
    // \u00a0 spelled out rather than pasted: a literal non-breaking space here
    // is invisible and any whitespace-normalising tool would silently drop it.
    return raw.replace(/\u00a0/g, ' ').replace(/\n{3,}/g, '\n\n').trim();
  }

  // ------------------------------------------------------------------ scan

  /**
   * All turns on the page, in document order.
   *
   * Uses the first selector in each chain that matches anything, so a UI change
   * falls back to an older selector rather than capturing nothing. Elements
   * nested inside another collected element are dropped -- otherwise a role
   * selector that happens to match inside the opposite role's container would
   * report the same text twice under two roles.
   */
  function collectTurns() {
    var out = [];
    firstMatch(adapter.user).forEach(function (el) { out.push({ role: 'user', el: el }); });
    firstMatch(adapter.assistant).forEach(function (el) { out.push({ role: 'assistant', el: el }); });
    if (out.length < 2) return out;

    var kept = out.filter(function (a) {
      return !out.some(function (b) { return b.el !== a.el && b.el.contains(a.el); });
    });
    kept.sort(function (a, b) {
      var rel = a.el.compareDocumentPosition(b.el);
      if (rel & Node.DOCUMENT_POSITION_FOLLOWING) return -1;
      if (rel & Node.DOCUMENT_POSITION_PRECEDING) return 1;
      return 0;
    });
    return kept;
  }

  function isStreaming() {
    return firstMatch(adapter.stop).length > 0;
  }

  function scan() {
    if (!settings.capture) return;

    var turns = collectTurns();
    if (!turns.length) return;

    var texts = turns.map(function (t) { return { role: t.role, text: textOf(t.el) }; });
    var tail = texts[texts.length - 1];
    var streaming = isStreaming();
    var tailDeferred = false;

    for (var i = 0; i < texts.length; i++) {
      var turn = texts[i];
      var isTail = i === texts.length - 1;

      // Every turn but the last is complete by construction -- a newer turn
      // exists after it. The last one needs two independent signals: no stop
      // button on screen, and text unchanged since the previous tick. Either
      // alone is unreliable (the stop button can lag a beat, and a slow token
      // stream can look settled between ticks).
      if (isTail && (streaming || turn.text !== prevTail)) { tailDeferred = true; continue; }
      if (turn.text.length < MIN_CHARS) continue;
      if (ECHO_MARKERS.some(function (m) { return turn.text.indexOf(m) !== -1; })) {
        if (turn.role === 'assistant') maybeSession(turn.text);
        continue;
      }

      var key = turn.role + ':' + hash32(turn.text);
      if (sent.has(key)) continue;
      sent.add(key);
      ship(turn.role, turn.text);
    }

    prevTail = tail ? tail.text : '';

    // Re-arm ourselves if the final turn was held back.
    //
    // This is not an optimisation, it is required: MutationObserver stops firing
    // the moment the DOM settles, so the very last reply of a conversation would
    // never get a second tick to compare against and would never be captured.
    // scan() is idempotent (the `sent` set), so a spare pass costs nothing.
    clearTimeout(settleTimer);
    if (tailDeferred) settleTimer = setTimeout(scan, DEBOUNCE_MS * 2);
  }

  function ship(role, text) {
    var body = text.length > MAX_CHARS
      ? text.slice(0, MAX_CHARS) + '\n[…truncated by OmniMem]'
      : text;
    var scrubbed = settings.redact ? redact(body) : body;

    send({
      t: 'capture',
      site: adapter.id,
      tool: adapter.id,
      role: role,
      text: scrubbed,
      redacted: settings.redact && scrubbed !== body,
      sessionId: sessionId,
      url: location.origin + location.pathname,
      project: settings.project || null
    }).then(function (res) {
      if (res && res.queued) setDot('queued');
      else if (res && res.ok) setDot('saved');
    });
  }

  /**
   * The assistant answered a summary request. Route the structured block to
   * /api/session instead of /api/capture.
   *
   * This is the browser equivalent of an agent calling omnimem_save_session:
   * the model in the page does the summarising, so the facts arrive at
   * agent-structured authority and cost nothing.
   */
  function maybeSession(text) {
    var m = text.match(/```(?:omnimem|json)?\s*(\{[\s\S]*?\})\s*```/);
    if (!m) return;
    var payload;
    try { payload = JSON.parse(m[1]); } catch (_) { return; }
    if (!payload || (!payload.facts && !payload.summary)) return;

    var key = 'session:' + hash32(m[1]);
    if (sent.has(key)) return;
    sent.add(key);

    send({
      t: 'session',
      site: adapter.id,
      tool: adapter.id,
      facts: Array.isArray(payload.facts) ? payload.facts : null,
      summary: payload.summary || null,
      next_step: payload.next_step || payload.nextStep || null,
      sessionId: sessionId,
      project: settings.project || null
    }).then(function () { setDot('saved'); flash('Session summary saved'); });
  }

  // ------------------------------------------------------------------ input

  function inputEl() {
    var els = firstMatch(adapter.input);
    return els.length ? els[els.length - 1] : null;
  }

  /**
   * Put text in the composer.
   *
   * React and ProseMirror both ignore a plain `.value =` or `.innerText =`,
   * because their state lives outside the DOM. For inputs we go through the
   * prototype's native setter (which React's onChange listener does observe);
   * for contenteditable we use execCommand('insertText') -- deprecated, but the
   * only thing that reliably produces a real beforeinput/input pair.
   */
  function insertText(el, text) {
    el.focus();
    if (el.tagName === 'TEXTAREA' || el.tagName === 'INPUT') {
      var proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement : HTMLInputElement;
      var setter = Object.getOwnPropertyDescriptor(proto.prototype, 'value').set;
      var joined = text + (el.value ? '\n\n' + el.value : '');
      setter.call(el, joined);
      el.dispatchEvent(new Event('input', { bubbles: true }));
      return true;
    }
    if (el.isContentEditable) {
      var ok = false;
      try { ok = document.execCommand('insertText', false, text + '\n\n'); } catch (_) { ok = false; }
      if (!ok) {
        el.textContent = text + '\n\n' + (el.textContent || '');
        el.dispatchEvent(new Event('input', { bubbles: true }));
      }
      return true;
    }
    return false;
  }

  function copyFallback(text) {
    navigator.clipboard.writeText(text).then(
      function () { flash('Composer not found — brief copied to clipboard'); },
      function () { flash('Could not inject or copy'); }
    );
  }

  var SUMMARY_PROMPT = [
    'OmniMem session summary request. Reply with ONLY this fenced block, nothing else:',
    '',
    '```omnimem',
    '{"summary": "2-3 sentences on what this conversation established",',
    ' "next_step": "the one sentence a fresh session should start from",',
    ' "facts": [{"kind": "goal", "text": "..."}]}',
    '```',
    '',
    'kind is one of: goal, decision, constraint, convention, stack, entity,',
    'gotcha, task, open_question. One atomic claim per entry, under 180',
    'characters, no pleasantries. Omit anything you are not sure about.'
  ].join('\n');

  // ------------------------------------------------------------------ ui

  var dot = null;

  function setDot(state) {
    if (!dot) return;
    dot.className = 'omnimem-dot omnimem-dot-' + state;
    dot.title = {
      off: 'OmniMem: capture paused on this site',
      offline: 'OmniMem: local engine not running',
      idle: 'OmniMem: watching this conversation',
      saved: 'OmniMem: saved',
      queued: 'OmniMem: engine offline — queued for later'
    }[state] || 'OmniMem';
    if (state === 'saved' || state === 'queued') {
      setTimeout(function () {
        if (dot) setDot(settings.capture ? 'idle' : 'off');
      }, 1800);
    }
  }

  function flash(text) {
    var el = document.getElementById('omnimem-flash');
    if (!el) return;
    el.textContent = text;
    el.classList.add('omnimem-flash-show');
    setTimeout(function () { el.classList.remove('omnimem-flash-show'); }, 2600);
  }

  /** Fetch the brief and put it in the composer. Shared by the bar and the popup. */
  function injectBrief(button) {
    if (button) { button.disabled = true; button.textContent = 'Loading…'; }
    return send({ t: 'resume', project: settings.project || null }).then(function (res) {
      if (button) { button.disabled = false; button.textContent = 'Inject brief'; }
      if (!res || !res.ok) {
        setDot('offline');
        flash('Local engine not running — start it with `omnimem serve`');
        return;
      }
      var text = res.brief +
        '\n\nContinue this project from the state above. Do not ask me to ' +
        're-explain anything already in it.';
      var el = inputEl();
      if (!el || !insertText(el, text)) copyFallback(text);
      else flash('Brief injected for "' + (res.project || 'project') + '"');
    });
  }

  /**
   * A fixed-position bar, not a button spliced into the composer.
   *
   * The original build inserted its button into the chat form, where React owns
   * the subtree and rips out foreign children on the next render -- so the
   * button vanished mid-conversation. Nothing here touches the page's own tree.
   */
  function buildBar() {
    if (document.getElementById('omnimem-bar')) return;

    var bar = document.createElement('div');
    bar.id = 'omnimem-bar';
    bar.className = 'omnimem-bar';

    dot = document.createElement('span');
    dot.className = 'omnimem-dot omnimem-dot-idle';

    var inject = document.createElement('button');
    inject.type = 'button';
    inject.className = 'omnimem-btn omnimem-btn-primary';
    inject.textContent = 'Inject brief';
    inject.addEventListener('click', function () { injectBrief(inject); });

    var save = document.createElement('button');
    save.type = 'button';
    save.className = 'omnimem-btn';
    save.textContent = 'Save session';
    save.title = 'Ask this model to summarise the conversation into OmniMem';
    save.addEventListener('click', function () {
      var el = inputEl();
      if (!el || !insertText(el, SUMMARY_PROMPT)) copyFallback(SUMMARY_PROMPT);
      else flash('Send this prompt — the reply is saved automatically');
    });

    var hide = document.createElement('button');
    hide.type = 'button';
    hide.className = 'omnimem-btn omnimem-btn-icon';
    hide.textContent = '×';
    hide.title = 'Hide until reload';
    hide.addEventListener('click', function () { bar.remove(); });

    bar.appendChild(dot);
    bar.appendChild(inject);
    bar.appendChild(save);
    bar.appendChild(hide);

    var flashEl = document.createElement('div');
    flashEl.id = 'omnimem-flash';
    flashEl.className = 'omnimem-flash';

    document.body.appendChild(bar);
    document.body.appendChild(flashEl);
    setDot(settings.capture ? 'idle' : 'off');
  }

  // ------------------------------------------------------------------ boot

  function onMutation() {
    clearTimeout(timer);
    timer = setTimeout(scan, DEBOUNCE_MS);
  }

  function startObserving() {
    if (observer) return;
    observer = new MutationObserver(onMutation);
    observer.observe(document.body, {
      childList: true,
      subtree: true,
      // Streaming replaces text inside an existing node as often as it appends
      // one, so characterData is required to notice the stream at all. The
      // debounce is what keeps this cheap: bursts collapse to one scan.
      characterData: true
    });
  }

  function stopObserving() {
    if (!observer) return;
    observer.disconnect();
    observer = null;
  }

  /**
   * Single-page navigation between conversations.
   *
   * Without this, moving from one chat to another in the same tab keeps the old
   * session id and the old `sent` set, so the new conversation's turns would be
   * filed under the previous chat.
   */
  function watchNavigation() {
    var last = location.pathname;
    var check = function () {
      if (location.pathname === last) return;
      last = location.pathname;
      sessionId = makeSessionId();
      sent = new Set();
      prevTail = '';
    };
    ['pushState', 'replaceState'].forEach(function (name) {
      var orig = history[name];
      history[name] = function () {
        var out = orig.apply(this, arguments);
        setTimeout(check, 0);
        return out;
      };
    });
    window.addEventListener('popstate', check);
  }

  chrome.storage.onChanged.addListener(function (changes) {
    if (changes.captureEnabled || changes.pausedSites || changes.redactEnabled) {
      send({ t: 'hello', site: adapter.id }).then(applySettings);
    }
  });

  // The popup cannot reach the page's DOM, so composer insertion is delegated
  // here. Also re-runs a scan on demand, which is the manual escape hatch when
  // a site changes its markup and the observer misses a turn.
  chrome.runtime.onMessage.addListener(function (msg, _sender, respond) {
    if (msg && msg.t === 'inject') {
      injectBrief();
      respond({ ok: true });
    } else if (msg && msg.t === 'scan') {
      prevTail = '';
      scan();
      respond({ ok: true });
    }
    return false;
  });

  function applySettings(reply) {
    settings = {
      capture: !!(reply && reply.capture) && adapter.user.length > 0,
      redact: reply ? reply.redact !== false : true,
      project: (reply && reply.project) || '',
      online: !!(reply && reply.online)
    };
    if (settings.capture) startObserving(); else stopObserving();
    if (dot) setDot(!settings.online ? 'offline' : (settings.capture ? 'idle' : 'off'));
  }

  send({ t: 'hello', site: adapter.id }).then(function (reply) {
    applySettings(reply);
    buildBar();
    watchNavigation();
    // One pass after the initial render, so opening an existing conversation
    // records its history rather than only what happens from now on.
    setTimeout(function () { prevTail = ''; scan(); }, 1200);
    setTimeout(scan, 2600);
  });
})();
