// Python sent inside a command (arbor/dashboard/server.mjs).
//
// Both of the server's modes run a Python file from this checkout on the far
// host, inside the command: the share's gateway takes commands, not stdin, and a
// cluster's own checkout is not ours to trust. Compressed, so that even Git for
// Windows' ssh (8,000 characters) passes it.
//
// The realtime collector also goes minified: its module docstring, and its
// comment-only and blank lines outside strings, removed before compressing.
// They were most of what pushed it over that limit (7,874 characters in
// October 2026), and the directions checklist needed room. Nothing else
// changes: arbor/tests/test_dashboard.py checks that the minified file parses
// to the same AST, apart from the module docstring. share_query.py is not
// minified: its usage message is its docstring.

import { readFileSync } from "node:fs";
import { deflateSync } from "node:zlib";

const TRIPLE = /"""|'''/g;

export function minified(text) {
  const out = [];
  let open = null;        // the delimiter of a triple-quoted string open at the line's end
  let first = true;       // no statement yet: a string here is the module docstring
  let dropping = null;    // inside the module docstring
  for (const line of text.split("\n")) {
    if (dropping) {
      if (line.includes(dropping)) dropping = null;
      continue;
    }
    if (!open && (/^\s*$/.test(line) || /^\s*#/.test(line))) continue;
    if (first) {
      first = false;
      const m = /^\s*[rRuU]?("""|''')/.exec(line);
      if (m) {
        const n = line.split(m[1]).length - 1;
        if (n % 2 === 1) dropping = m[1];
        continue;
      }
    }
    out.push(line);
    for (const d of line.match(TRIPLE) || []) {
      if (open === null) open = d;
      else if (open === d) open = null;
    }
  }
  return out.join("\n") + "\n";
}

export function shipped(file, { minify = false } = {}) {
  let text = readFileSync(file, "utf8");
  if (minify) text = minified(text);
  return `import base64,zlib;exec(zlib.decompress(base64.b64decode("${
    deflateSync(Buffer.from(text, "utf8")).toString("base64")}")))`;
}
