// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Build plugin: writes THIRD_PARTY_LICENSES.txt next to index.html, listing every npm package whose code ends up in
// the bundle with its licence text. The build fails on a package without a licence file or with a licence outside
// ALLOWED, so a new dependency can't ship unnoticed.

import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { Plugin } from "vite";

export const ALLOWED = new Set(["MIT", "ISC", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "0BSD"]);

/** The package directory of a bundled module id, or null for our own sources and virtual modules. */
export function packageDir(id: string): string | null {
  const path = id.split("?")[0].replaceAll("\\", "/");
  const at = path.lastIndexOf("/node_modules/");
  if (path.startsWith("\0") || at < 0) return null;
  const rest = path.slice(at + "/node_modules/".length).split("/");
  const depth = rest[0].startsWith("@") ? 2 : 1;
  if (rest.length <= depth) return null;
  return path.slice(0, at) + "/node_modules/" + rest.slice(0, depth).join("/");
}

function licenseText(dir: string): string | null {
  const file = readdirSync(dir).find((f) => /^(licen[cs]e|copying)(\.(md|txt))?$/i.test(f));
  return file ? readFileSync(join(dir, file), "utf-8").trim() : null;
}

export function thirdPartyLicenses(): Plugin {
  return {
    name: "hyperloom-third-party-licenses",
    apply: "build",
    generateBundle(_options, bundle) {
      const dirs = new Set<string>();
      for (const chunk of Object.values(bundle)) {
        if (chunk.type !== "chunk") continue;
        for (const [id, mod] of Object.entries(chunk.modules)) {
          const dir = packageDir(id);
          if (dir && mod.renderedLength > 0) dirs.add(dir);
        }
      }
      const entries: string[] = [];
      for (const dir of [...dirs].sort()) {
        const pkg = JSON.parse(readFileSync(join(dir, "package.json"), "utf-8"));
        const text = existsSync(dir) ? licenseText(dir) : null;
        if (!ALLOWED.has(pkg.license)) this.error(`${pkg.name}: licence ${JSON.stringify(pkg.license)} is not allowed`);
        if (!text) this.error(`${pkg.name}: no licence file in ${dir}`);
        entries.push(`${pkg.name}@${pkg.version} (${pkg.license})\n${pkg.homepage ?? ""}\n\n${text}`);
      }
      const header =
        "Third-party code bundled in the Hyperloom Inspector's modern UI (static/modern/assets/).\n" +
        "Everything else in the bundle is Hyperloom's, under the MIT licence.";
      this.emitFile({
        type: "asset",
        fileName: "THIRD_PARTY_LICENSES.txt",
        source: [header, ...entries].join("\n\n" + "-".repeat(80) + "\n\n") + "\n",
      });
    },
  };
}
