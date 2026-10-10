/**
 * OmniMem — outbound redaction.
 *
 * Runs in the *content script*, before any captured text leaves the page. That
 * placement is deliberate: redacting in the background worker would mean the
 * secret had already crossed a process boundary and could show up in a crash
 * dump or an extension log. Here, the raw string never exists outside the tab.
 *
 * Loaded as a plain content script and via importScripts() in the worker, so it
 * assigns onto globalThis rather than using module syntax.
 */

(function (root) {
  'use strict';

  // Ordered most-specific first: a private key block is matched and removed
  // whole before the generic "assignment" rule can nibble at one of its lines.
  var RULES = [
    // Whole PEM blocks. [\s\S] because the payload spans newlines.
    [/-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g,
      '[redacted:private-key]'],

    // JWTs. Three base64url segments; the eyJ prefix is the decoded '{"' so it
    // anchors the match and keeps this off ordinary dotted identifiers.
    [/\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b/g, '[redacted:jwt]'],

    // Vendor-prefixed keys, which are self-identifying and safe to match hard.
    [/\bsk-(?:proj-|ant-|live-|test-)?[A-Za-z0-9_-]{16,}\b/g, '[redacted:api-key]'],
    [/\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b/g, '[redacted:github-token]'],
    [/\bgithub_pat_[A-Za-z0-9_]{30,}\b/g, '[redacted:github-token]'],
    [/\bAKIA[0-9A-Z]{16}\b/g, '[redacted:aws-key]'],
    [/\bASIA[0-9A-Z]{16}\b/g, '[redacted:aws-key]'],
    [/\bAIza[0-9A-Za-z_-]{35}\b/g, '[redacted:google-key]'],
    [/\bya29\.[0-9A-Za-z_-]{20,}\b/g, '[redacted:google-token]'],
    [/\bxox[baprs]-[0-9A-Za-z-]{10,}\b/g, '[redacted:slack-token]'],
    [/\bhf_[A-Za-z0-9]{30,}\b/g, '[redacted:hf-token]'],
    [/\bglpat-[A-Za-z0-9_-]{16,}\b/g, '[redacted:gitlab-token]'],
    [/\bnpm_[A-Za-z0-9]{30,}\b/g, '[redacted:npm-token]'],
    [/\bdop_v1_[a-f0-9]{60,}\b/g, '[redacted:do-token]'],
    [/\bsk_live_[A-Za-z0-9]{20,}\b/g, '[redacted:stripe-key]'],

    // Authorization headers, in curl or in prose.
    [/\b(?:Bearer|Basic|Token)\s+[A-Za-z0-9._~+/=-]{12,}/gi, 'Bearer [redacted]'],

    // Anything that names itself a secret and then assigns a value. Bounded to
    // one line so it cannot swallow the rest of a code block.
    [/\b(api[_-]?key|secret[_-]?key|secret|password|passwd|access[_-]?token|refresh[_-]?token|auth[_-]?token|private[_-]?key|client[_-]?secret)(\s*[:=]\s*|"\s*:\s*)["']?[^\s"',;)]{6,}["']?/gi,
      function (m, name, sep) { return name + (sep.indexOf('=') >= 0 ? '=' : ': ') + '[redacted]'; }],

    // Postgres/Mongo/Redis URLs with inline credentials. Keep the host: knowing
    // "we are on Postgres" is exactly the kind of fact worth remembering, the
    // password is not.
    [/\b([a-z][a-z0-9+.-]*:\/\/)([^\s:/@]+):([^\s:/@]+)@/gi,
      function (m, scheme, user) { return scheme + user + ':[redacted]@'; }],

    // Email addresses.
    [/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g, '[redacted:email]'],

    // Card-shaped digit runs (4x4 with optional separators). Narrow on purpose:
    // a bare 16-digit run is left alone because too many legitimate ids look
    // like one, and over-redaction quietly destroys memory quality.
    [/\b(?:\d[ -]?){3}\d(?:[ -]\d{4}){3}\b/g, '[redacted:card]']
  ];

  function redact(text) {
    if (!text) return text;
    var out = String(text);
    for (var i = 0; i < RULES.length; i++) {
      out = out.replace(RULES[i][0], RULES[i][1]);
    }
    return out;
  }

  /** True when redact() would change the text. Used only for the UI counter. */
  function wouldRedact(text) {
    return redact(text) !== text;
  }

  root.OmniMemRedact = { redact: redact, wouldRedact: wouldRedact, RULES: RULES };
})(typeof self !== 'undefined' ? self : this);
