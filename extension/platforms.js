/**
 * OmniMem — per-site DOM adapters.
 *
 * Every selector in the extension lives here. Chat UIs rewrite their markup
 * constantly, so when capture breaks on one site this is the only file to touch
 * -- the content script itself contains no site knowledge at all.
 *
 * Each adapter is:
 *   id          stable name recorded as the `tool` on captured turns
 *   label       shown in the popup
 *   match(host) true when this adapter owns the page
 *   user[]      selectors for user turns, tried in order
 *   assistant[] selectors for assistant turns, tried in order
 *   input[]     selectors for the prompt box, tried in order
 *   stop[]      selectors for the "stop generating" control; its presence is
 *               treated as "the last assistant turn is still streaming"
 *   session(loc) conversation id from the URL, so one tab that moves between
 *               chats does not merge them into a single session
 *
 * Selector lists are fallback chains rather than single values: the first one
 * that matches anything wins. That way a UI change usually degrades to an older
 * selector instead of silently capturing nothing.
 */

(function (root) {
  'use strict';

  function lastPath(loc) {
    var parts = (loc.pathname || '').split('/').filter(Boolean);
    return parts.length ? parts[parts.length - 1] : '';
  }

  var PLATFORMS = [
    {
      id: 'chatgpt',
      label: 'ChatGPT',
      match: function (h) { return /(^|\.)chatgpt\.com$|(^|\.)openai\.com$/.test(h); },
      // ChatGPT tags both roles with one attribute, which is the most stable
      // hook it has ever had; the class-based entries are older builds.
      user: ['[data-message-author-role="user"]', '.whitespace-pre-wrap.break-words'],
      assistant: ['[data-message-author-role="assistant"]', '.agent-turn .markdown'],
      input: ['#prompt-textarea', 'div[contenteditable="true"]#prompt-textarea',
        'form div[contenteditable="true"]', 'form textarea'],
      stop: ['button[data-testid="stop-button"]', 'button[aria-label*="Stop"]'],
      session: function (loc) { return /\/c\//.test(loc.pathname) ? lastPath(loc) : ''; }
    },
    {
      id: 'claude_web',
      label: 'Claude.ai',
      match: function (h) { return /(^|\.)claude\.ai$/.test(h); },
      user: ['[data-testid="user-message"]', '.font-user-message'],
      assistant: ['.font-claude-message', '[data-is-streaming="false"] .prose'],
      input: ['div[contenteditable="true"].ProseMirror', 'div[contenteditable="true"]'],
      stop: ['button[aria-label*="Stop"]', '[data-is-streaming="true"]'],
      session: function (loc) { return /\/chat\//.test(loc.pathname) ? lastPath(loc) : ''; }
    },
    {
      id: 'gemini',
      label: 'Gemini',
      match: function (h) { return /(^|\.)gemini\.google\.com$/.test(h); },
      user: ['user-query .query-text', 'user-query'],
      assistant: ['model-response message-content', 'model-response .markdown',
        '.model-response-text'],
      input: ['rich-textarea div[contenteditable="true"]', 'div[contenteditable="true"]'],
      stop: ['button[aria-label*="Stop"]', 'mat-icon[fonticon="stop"]'],
      session: function (loc) { return /\/app\//.test(loc.pathname) ? lastPath(loc) : ''; }
    },
    {
      id: 'aistudio',
      label: 'AI Studio',
      match: function (h) { return /(^|\.)aistudio\.google\.com$/.test(h); },
      user: ['ms-chat-turn[data-turn-role="User"]', '.user-prompt-container'],
      assistant: ['ms-chat-turn[data-turn-role="Model"]', '.model-prompt-container'],
      input: ['textarea[aria-label*="prompt"]', 'ms-autosize-textarea textarea', 'textarea'],
      stop: ['button[aria-label*="Stop"]'],
      session: function (loc) { return lastPath(loc); }
    },
    {
      id: 'perplexity',
      label: 'Perplexity',
      match: function (h) { return /(^|\.)perplexity\.ai$/.test(h); },
      user: ['[data-testid="user-query"]', 'div.group\\/query', 'h1.group\\/query'],
      assistant: ['[data-testid="answer"]', 'div[id^="markdown-content"]', '.prose'],
      input: ['textarea[placeholder]', 'div[contenteditable="true"]'],
      stop: ['button[aria-label*="Stop"]'],
      session: function (loc) { return /\/search\//.test(loc.pathname) ? lastPath(loc) : ''; }
    },
    {
      id: 'deepseek',
      label: 'DeepSeek',
      match: function (h) { return /(^|\.)deepseek\.com$/.test(h); },
      // DeepSeek ships hashed class names, so these are the least durable
      // selectors in the file. The role-attribute fallbacks are what usually
      // survive a redeploy.
      user: ['[class*="_user"] [class*="markdown"]', 'div[class*="fbb737a4"]'],
      assistant: ['[class*="ds-markdown"]', 'div[class*="f9bf7997"]'],
      input: ['textarea#chat-input', 'textarea'],
      stop: ['div[role="button"][aria-label*="Stop"]'],
      session: function (loc) { return /\/a\/chat\//.test(loc.pathname) ? lastPath(loc) : ''; }
    },
    {
      id: 'grok',
      label: 'Grok',
      match: function (h) { return /(^|\.)grok\.com$|(^|\.)x\.com$/.test(h); },
      user: ['.message-bubble.items-end', 'div[class*="justify-end"] .message-bubble'],
      assistant: ['.message-bubble:not(.items-end)', '.response-content-markdown'],
      input: ['textarea[aria-label*="Ask"]', 'div[contenteditable="true"]', 'textarea'],
      stop: ['button[aria-label*="Stop"]'],
      session: function (loc) { return /\/c\//.test(loc.pathname) ? lastPath(loc) : ''; }
    },
    {
      id: 'copilot',
      label: 'Copilot',
      match: function (h) { return /(^|\.)copilot\.microsoft\.com$/.test(h); },
      user: ['[data-content="user-message"]', 'div[class*="userMessage"]'],
      assistant: ['[data-content="ai-message"]', 'div[class*="aiMessage"]'],
      input: ['textarea#userInput', 'textarea'],
      stop: ['button[title*="Stop"]'],
      session: function (loc) { return lastPath(loc); }
    },
    {
      id: 'mistral',
      label: 'Le Chat',
      match: function (h) { return /(^|\.)mistral\.ai$/.test(h); },
      user: ['div[class*="justify-end"] .prose', '[data-message-author-role="user"]'],
      assistant: ['div[class*="justify-start"] .prose', '[data-message-author-role="assistant"]'],
      input: ['textarea', 'div[contenteditable="true"]'],
      stop: ['button[aria-label*="Stop"]'],
      session: function (loc) { return /\/chat\//.test(loc.pathname) ? lastPath(loc) : ''; }
    }
  ];

  /**
   * Generic fallback for a site with no adapter.
   *
   * Captures nothing -- guessing roles from unknown markup produces garbage
   * facts, and a memory store poisoned with garbage is worse than an empty one.
   * The Inject button still works, so an unlisted site is read-only.
   */
  var GENERIC = {
    id: 'web',
    label: 'Web',
    match: function () { return true; },
    user: [],
    assistant: [],
    input: ['div[contenteditable="true"]', 'textarea'],
    stop: [],
    session: function () { return ''; }
  };

  function forHost(host) {
    for (var i = 0; i < PLATFORMS.length; i++) {
      if (PLATFORMS[i].match(host)) return PLATFORMS[i];
    }
    return GENERIC;
  }

  root.OmniMemPlatforms = {
    all: PLATFORMS,
    generic: GENERIC,
    forHost: forHost
  };
})(typeof self !== 'undefined' ? self : this);
