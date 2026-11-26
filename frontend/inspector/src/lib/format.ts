// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Display formatting.

// toLocale*String builds an Intl formatter on every call, which cost a replay at x600 a tenth of its frame time;
// the formatters are built once and reused.
const numFormats = new Map<number, Intl.NumberFormat>();
const timeFormat = new Intl.DateTimeFormat([], {
  hour12: false,
  hour: "numeric",
  minute: "numeric",
  second: "numeric",
});
const dateFormat = new Intl.DateTimeFormat([], { month: "2-digit", day: "2-digit" });

export function fmtNum(n: number | null | undefined, digits = 1): string {
  if (n == null) return "–";
  let f = numFormats.get(digits);
  if (!f) numFormats.set(digits, (f = new Intl.NumberFormat(undefined, { maximumFractionDigits: digits })));
  return f.format(Number(n));
}

const compactFormat = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 });

/** Short counts for axes and cards: 6.9M, 657.5K. */
export function fmtCompact(n: number | null | undefined): string {
  return n == null || !isFinite(n) ? "–" : compactFormat.format(n);
}

/** `1,234 artifacts`; once the index stopped at its cap, `20,000+ artifacts (list cut at 20,000)`. */
export function fmtArtifactCount(n: number, cutAt: number | null | undefined): string {
  const count = `${fmtNum(n, 0)}${cutAt != null && n >= cutAt ? "+" : ""} artifacts`;
  return cutAt == null ? count : `${count} (list cut at ${fmtNum(cutAt, 0)})`;
}

export function fmtBytes(b: number | null | undefined): string {
  if (b == null) return "–";
  if (b > 2 ** 30) return (b / 2 ** 30).toFixed(1) + " GiB";
  if (b > 2 ** 20) return (b / 2 ** 20).toFixed(1) + " MiB";
  if (b > 1024) return (b / 1024).toFixed(1) + " KiB";
  return b + " B";
}

/** Local time of day; the date is added when *t* is more than a day from *wall*. */
export function fmtTime(t: number | null | undefined, wall = Date.now() / 1000): string {
  if (t == null) return "–";
  const d = new Date(t * 1000);
  const time = timeFormat.format(d);
  return Math.abs(wall - t) > 86400 ? `${time} ${dateFormat.format(d)}` : time;
}

export function fmtDur(sec: number | null | undefined): string {
  if (sec == null || !isFinite(sec)) return "–";
  const s = Math.max(0, Math.round(sec));
  const hh = Math.floor(s / 3600);
  const mm = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  if (hh) return `${hh}h${String(mm).padStart(2, "0")}m`;
  if (mm) return `${mm}m${String(ss).padStart(2, "0")}s`;
  return `${ss}s`;
}

export function preview(v: unknown, n = 160): string {
  const t = typeof v === "string" ? v : JSON.stringify(v);
  if (t == null) return "";
  return t.length > n ? t.slice(0, n) + "…" : t;
}
