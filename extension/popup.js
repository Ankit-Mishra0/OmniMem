/**
 * OmniMem popup.
 *
 * Pure view layer: it reads state from the background worker and writes changes
 * back through the same message channel. It never touches the network or
 * chrome.storage directly, so the worker stays the single writer and the popup
 * cannot drift out of sync with what a content script believes.
 */

'use strict';

var $ = function (id) { return document.getElementById(id); };

var tab = null;
var adapter = null;
var state = null;

function send(msg) {
  return new Promise(function (resolve) {
    chrome.runtime.sendMessage(msg, function (reply) {
      void chrome.runtime.lastError;
      resolve(reply || { ok: false });
    });
  });
}

function siteId() {
  if (!tab || !tab.url) return '';
  try {
    return window.OmniMemPlatforms.forHost(new URL(tab.url).hostname).id;
  } catch (_) {
    return '';
  }
}

function siteLabel() {
  if (!adapter) return 'this site';
  return adapter.label;
}

// ------------------------------------------------------------------ render

function paint() {
  var online = state.online;

  $('status-dot').className = 'dot ' + (online ? 'online' : 'offline');
  $('status-text').textContent = online
    ? 'engine v' + (state.version || '?')
    : 'engine offline';
  $('engine-off').hidden = online;

  $('capture').checked = !!state.captureEnabled;
  $('redact').checked = !!state.redactEnabled;
  $('port').value = state.port;

  var site = siteId();
  $('site-name').textContent = siteLabel();
  $('pause-site').checked = site ? state.pausedSites.indexOf(site) !== -1 : false;
  $('pause-site').disabled = !site;
  $('paused-count').textContent = state.pausedSites.length
    ? state.pausedSites.length + ' site' + (state.pausedSites.length > 1 ? 's' : '') + ' paused'
    : 'No sites paused';

  $('stat-captured').textContent = state.stats.captured;
  $('stat-redacted').textContent = state.stats.redacted;
  $('stat-queued').textContent = state.queued;
  $('flush').hidden = !state.queued || !online;

  $('err-line').textContent = state.stats.lastError
    ? 'Last error: ' + state.stats.lastError
    : '';

  $('serve-cmd').textContent = state.port === 8000
    ? 'omnimem serve'
    : 'omnimem serve --port ' + state.port;

  $('inject').disabled = !online || !tab;
}

async function refresh() {
  state = await send({ t: 'status' });
  paint();
}

// ------------------------------------------------------------------ actions

async function saveSettings(patch) {
  await send(Object.assign({ t: 'settings' }, patch));
  await refresh();
}

function toggleSite(on) {
  var site = siteId();
  if (!site) return;
  var list = state.pausedSites.filter(function (s) { return s !== site; });
  if (on) list.push(site);
  saveSettings({ pausedSites: list });
}

async function loadProjects() {
  var res = await send({ t: 'projects' });
  var list = $('project-list');
  list.textContent = '';
  (res.projects || []).forEach(function (p) {
    var opt = document.createElement('option');
    opt.value = p.project;
    // The datalist label is where "which project is this?" actually gets
    // answered -- a slug alone is ambiguous once you have a dozen of them.
    opt.label = (p.facts || 0) + ' facts · last in ' + (p.last_tool || 'unknown');
    list.appendChild(opt);
  });
  if (res.projects && res.projects.length) {
    $('project-hint').textContent = res.projects.length +
      ' project' + (res.projects.length > 1 ? 's' : '') + ' tracked. Leave blank to let OmniMem infer.';
  }
}

// ------------------------------------------------------------------ wiring

document.addEventListener('DOMContentLoaded', async function () {
  var tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  tab = tabs && tabs[0] ? tabs[0] : null;
  if (tab && tab.url) {
    try {
      adapter = window.OmniMemPlatforms.forHost(new URL(tab.url).hostname);
    } catch (_) { adapter = null; }
  }

  await refresh();
  await loadProjects();

  $('project').value = state.defaultProject || '';

  $('capture').addEventListener('change', function (e) {
    saveSettings({ captureEnabled: e.target.checked });
  });
  $('redact').addEventListener('change', function (e) {
    saveSettings({ redactEnabled: e.target.checked });
  });
  $('pause-site').addEventListener('change', function (e) {
    toggleSite(e.target.checked);
  });

  // Commit the project on blur and on Enter rather than per keystroke: each
  // save round-trips to the engine to mirror config, and doing that on every
  // character would hammer it while you type a name.
  var commitProject = function () {
    var value = $('project').value.trim();
    if (value !== (state.defaultProject || '')) saveSettings({ defaultProject: value });
  };
  $('project').addEventListener('blur', commitProject);
  $('project').addEventListener('keydown', function (e) {
    if (e.key === 'Enter') commitProject();
  });

  $('this-tab').addEventListener('click', async function () {
    if (!tab) return;
    await send({ t: 'setTabProject', tabId: tab.id, project: $('project').value.trim() });
    $('project-hint').textContent = 'This tab now files into "' +
      ($('project').value.trim() || 'inferred') + '".';
  });

  $('inject').addEventListener('click', async function () {
    if (!tab) return;
    // The content script owns composer insertion -- it is the only context that
    // can see the page's DOM.
    chrome.tabs.sendMessage(tab.id, { t: 'inject' }, function () {
      void chrome.runtime.lastError;
      window.close();
    });
  });

  $('flush').addEventListener('click', async function () {
    $('flush').disabled = true;
    await send({ t: 'flush' });
    $('flush').disabled = false;
    await refresh();
  });

  $('port').addEventListener('change', function (e) {
    var port = parseInt(e.target.value, 10);
    if (port > 0 && port < 65536) saveSettings({ port: port });
  });

  // The engine serves the landing page at /docs, so the setup guide works
  // offline and always matches the installed version.
  $('docs').addEventListener('click', function (e) {
    e.preventDefault();
    chrome.tabs.create({ url: 'http://127.0.0.1:' + state.port + '/docs' });
  });
});
