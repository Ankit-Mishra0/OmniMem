/**
 * OmniMem background service worker.
 *
 * The worker owns *all* state and *all* network I/O; the content script is a
 * thin DOM scraper that posts messages here. Two reasons this split is not
 * optional:
 *
 * 1. Origin. A fetch from a content script carries the *page's* origin
 *    (https://chatgpt.com), which /api/handshake deliberately rejects. A fetch
 *    from the worker carries chrome-extension://<id>, which it accepts. So the
 *    token can only ever be obtained from here.
 * 2. Survivability. A tab reload wipes the content script mid-conversation. The
 *    worker keeps the offline queue in chrome.storage, so a capture taken while
 *    the server was down still lands after the next successful ping.
 *
 * MV3 terminates this worker whenever it is idle, so nothing durable lives in a
 * module-level variable -- every read goes through chrome.storage, with only a
 * throw-away memory cache on top.
 */

'use strict';

importScripts('redact.js');

const DEFAULTS = {
  port: 8000,
  token: '',
  captureEnabled: true,
  redactEnabled: true,
  pausedSites: [],
  defaultProject: '',
  queue: [],
  stats: { captured: 0, redacted: 0, queued: 0, lastAt: 0, lastError: '' }
};

const QUEUE_MAX = 300;      // ~a few hundred KB; bounded so a long outage
                            // cannot grow storage without limit
const PING_TIMEOUT_MS = 1500;
const API_TIMEOUT_MS = 8000;

let cache = null;

async function getState() {
  if (cache) return cache;
  const got = await chrome.storage.local.get(Object.keys(DEFAULTS));
  cache = Object.assign({}, DEFAULTS, got);
  return cache;
}

async function setState(patch) {
  const state = await getState();
  Object.assign(state, patch);
  cache = state;
  await chrome.storage.local.set(patch);
  return state;
}

function base(state) {
  // 127.0.0.1 rather than localhost: on some systems localhost resolves to ::1
  // first, uvicorn binds 127.0.0.1, and the request fails with no useful error.
  return 'http://127.0.0.1:' + (state.port || 8000);
}

function withTimeout(ms) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), ms);
  return { signal: ctl.signal, done: () => clearTimeout(timer) };
}

// ---------------------------------------------------------------- transport

async function ping() {
  const state = await getState();
  const t = withTimeout(PING_TIMEOUT_MS);
  try {
    const res = await fetch(base(state) + '/health', { signal: t.signal });
    if (!res.ok) return null;
    return await res.json();
  } catch (_) {
    return null;
  } finally {
    t.done();
  }
}

async function fetchToken() {
  const state = await getState();
  const t = withTimeout(PING_TIMEOUT_MS);
  try {
    const res = await fetch(base(state) + '/api/handshake', { signal: t.signal });
    if (!res.ok) return '';
    const data = await res.json();
    if (data && data.token) {
      await setState({ token: data.token });
      return data.token;
    }
    return '';
  } catch (_) {
    return '';
  } finally {
    t.done();
  }
}

/**
 * Call the local API. Returns {ok, status, data}.
 *
 * Never throws: a memory tool that surfaces errors into the page is worse than
 * one that quietly stops recording, so every failure comes back as ok:false and
 * the caller decides whether to queue.
 */
async function api(path, opts) {
  opts = opts || {};
  let state = await getState();
  let token = state.token || (await fetchToken());

  const send = async (tok) => {
    const t = withTimeout(API_TIMEOUT_MS);
    try {
      const res = await fetch(base(state) + path, {
        method: opts.method || 'GET',
        headers: Object.assign(
          { 'X-OmniMem-Token': tok },
          opts.body ? { 'Content-Type': 'application/json' } : {}
        ),
        body: opts.body ? JSON.stringify(opts.body) : undefined,
        signal: t.signal
      });
      let data = null;
      try { data = await res.json(); } catch (_) { /* empty or non-JSON body */ }
      return { ok: res.ok, status: res.status, data: data };
    } catch (err) {
      return { ok: false, status: 0, data: null, error: String(err && err.message || err) };
    } finally {
      t.done();
    }
  };

  let out = await send(token);
  if (out.status === 401 || out.status === 503) {
    // The store was re-provisioned (or this is a first run against a server
    // that had not generated a token yet). Re-handshake once, then give up.
    const fresh = await fetchToken();
    if (fresh && fresh !== token) out = await send(fresh);
  }
  if (!out.ok) {
    await setState({ stats: Object.assign({}, state.stats, {
      lastError: out.error || ('HTTP ' + out.status)
    }) });
  }
  return out;
}

// ---------------------------------------------------------------- queue

async function enqueue(item) {
  const state = await getState();
  const queue = state.queue.slice();
  queue.push(item);
  while (queue.length > QUEUE_MAX) queue.shift();
  await setState({
    queue: queue,
    stats: Object.assign({}, state.stats, { queued: queue.length })
  });
}

/**
 * Drain the queue oldest-first, stopping at the first failure.
 *
 * Stopping (rather than skipping) preserves conversation order, which matters:
 * ingest treats the newest sighting of a fact as its recency, so replaying a
 * later turn before an earlier one would age the earlier one wrongly.
 */
async function flush() {
  const state = await getState();
  if (!state.queue.length) return 0;
  if (!(await ping())) return 0;

  const queue = state.queue.slice();
  let sent = 0;
  while (queue.length) {
    const item = queue[0];
    const out = await api(item.path, { method: 'POST', body: item.body });
    if (!out.ok) break;
    queue.shift();
    sent++;
  }
  const fresh = await getState();
  await setState({
    queue: queue,
    stats: Object.assign({}, fresh.stats, { queued: queue.length })
  });
  return sent;
}

async function post(path, body) {
  const out = await api(path, { method: 'POST', body: body });
  if (!out.ok) {
    await enqueue({ path: path, body: body });
    return { ok: false, queued: true };
  }
  return { ok: true, data: out.data };
}

// ---------------------------------------------------------------- per-tab project

// storage.session is cleared when the browser closes, which is the right
// lifetime for "this tab is working on project X".
const sessionStore = chrome.storage.session || chrome.storage.local;

async function tabProject(tabId) {
  if (tabId == null) return '';
  const got = await sessionStore.get(['tabProjects']);
  const map = got.tabProjects || {};
  return map[String(tabId)] || '';
}

async function setTabProject(tabId, project) {
  const got = await sessionStore.get(['tabProjects']);
  const map = got.tabProjects || {};
  if (project) map[String(tabId)] = project;
  else delete map[String(tabId)];
  await sessionStore.set({ tabProjects: map });
}

async function resolveProject(tabId, hint) {
  if (hint) return hint;
  const perTab = await tabProject(tabId);
  if (perTab) return perTab;
  const state = await getState();
  return state.defaultProject || '';
}

// ---------------------------------------------------------------- badge

async function paintBadge(tabId, paused) {
  if (tabId == null) return;
  try {
    await chrome.action.setBadgeBackgroundColor({ tabId: tabId, color: paused ? '#64748b' : '#6366f1' });
    await chrome.action.setBadgeText({ tabId: tabId, text: paused ? '' : '●' });
  } catch (_) { /* tab closed between capture and paint */ }
}

// ---------------------------------------------------------------- messages

async function handle(msg, sender) {
  const tabId = sender && sender.tab ? sender.tab.id : null;
  const state = await getState();

  switch (msg.t) {
    case 'hello': {
      // Content script boot: tell it whether to observe at all, so a paused
      // site never even runs the MutationObserver.
      const paused = state.pausedSites.indexOf(msg.site) !== -1;
      await paintBadge(tabId, paused || !state.captureEnabled);
      return {
        capture: state.captureEnabled && !paused,
        redact: state.redactEnabled,
        project: await resolveProject(tabId, ''),
        online: !!(await ping())
      };
    }

    case 'capture': {
      const paused = state.pausedSites.indexOf(msg.site) !== -1;
      if (!state.captureEnabled || paused) return { ok: false, reason: 'paused' };
      const project = await resolveProject(tabId, msg.project);
      const body = {
        project: project || null,
        tool: msg.tool || 'browser',
        role: msg.role,
        text: msg.text,
        session_id: msg.sessionId || null,
        url: msg.url || null
      };
      const res = await post('/api/capture', body);
      const stats = Object.assign({}, state.stats, {
        captured: state.stats.captured + 1,
        redacted: state.stats.redacted + (msg.redacted ? 1 : 0),
        lastAt: Date.now()
      });
      await setState({ stats: stats });
      await paintBadge(tabId, false);
      return res;
    }

    case 'session': {
      const project = await resolveProject(tabId, msg.project);
      return await post('/api/session', {
        project: project || null,
        tool: msg.tool || 'browser',
        facts: msg.facts || null,
        summary: msg.summary || null,
        next_step: msg.next_step || null,
        session_id: msg.sessionId || null
      });
    }

    case 'resume': {
      const project = await resolveProject(tabId, msg.project);
      const qs = project ? '?project=' + encodeURIComponent(project) : '';
      const out = await api('/api/resume' + qs);
      if (!out.ok) return { ok: false, error: out.error || ('HTTP ' + out.status) };
      return { ok: true, project: out.data.project, brief: out.data.brief };
    }

    case 'projects': {
      const out = await api('/api/overview');
      return { ok: out.ok, projects: (out.data && out.data.projects) || [] };
    }

    case 'status': {
      const health = await ping();
      const fresh = await getState();
      return {
        online: !!health,
        version: health ? health.version : null,
        port: fresh.port,
        hasToken: !!fresh.token,
        captureEnabled: fresh.captureEnabled,
        redactEnabled: fresh.redactEnabled,
        pausedSites: fresh.pausedSites,
        defaultProject: fresh.defaultProject,
        stats: fresh.stats,
        queued: fresh.queue.length
      };
    }

    case 'settings': {
      const patch = {};
      ['port', 'captureEnabled', 'redactEnabled', 'pausedSites', 'defaultProject'].forEach((k) => {
        if (k in msg) patch[k] = msg[k];
      });
      if ('port' in patch) patch.token = '';   // different server, different token
      await setState(patch);
      // Mirror the toggles into ~/.omnimem/config.json so `omnimem doctor` and
      // any other client see the same state the popup shows.
      if ('captureEnabled' in patch || 'redactEnabled' in patch || 'pausedSites' in patch) {
        const fresh = await getState();
        await api('/api/config', { method: 'POST', body: {
          capture: fresh.captureEnabled,
          redact: fresh.redactEnabled,
          paused_sites: fresh.pausedSites
        } });
      }
      await flush();
      return { ok: true };
    }

    case 'setTabProject': {
      await setTabProject(msg.tabId != null ? msg.tabId : tabId, msg.project);
      return { ok: true };
    }

    case 'flush':
      return { ok: true, sent: await flush() };

    default:
      return { ok: false, error: 'unknown message ' + msg.t };
  }
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  handle(msg, sender).then(sendResponse, (err) => {
    sendResponse({ ok: false, error: String(err && err.message || err) });
  });
  return true;   // keep the channel open for the async reply
});

chrome.runtime.onInstalled.addListener(() => { fetchToken().then(flush); });
chrome.runtime.onStartup.addListener(() => { fetchToken().then(flush); });
chrome.tabs.onRemoved.addListener((tabId) => { setTabProject(tabId, ''); });
