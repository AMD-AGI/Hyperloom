// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Hover hints: an element with `data-hint` shows that text the moment the pointer is on it or it takes the keyboard
// focus, where a native title waits about a second. `data-hint-head` (optional) is a figure shown above it in bold.
// Each line of the text is a paragraph of its own.
// One tooltip for the page, delegated from the document, so views only set the attributes.

/** Space between the hint and its element, and kept to the viewport's edges. */
export const GAP = 6;

/**
 * The hint's top-left corner: below its element and left-aligned with it; above it when it would run past the bottom
 * edge and fits there; always inside the viewport.
 */
export function placeHint(
  r: { left: number; top: number; bottom: number },
  w: number,
  h: number,
  vw: number,
  vh: number,
): { left: number; top: number } {
  const below = r.bottom + GAP;
  const above = r.top - GAP - h;
  const top = below + h > vh - GAP && above >= GAP ? above : below;
  return {
    left: Math.max(GAP, Math.min(r.left, vw - GAP - w)),
    top: Math.max(GAP, Math.min(top, vh - GAP - h)),
  };
}

/** Shows the hints of every `[data-hint]` element in *doc*; returns the uninstall. */
export function installHints(doc: Document = document): () => void {
  const tip = doc.createElement("div");
  tip.className = "hint-tip";
  tip.id = "hint-tip";
  tip.setAttribute("role", "tooltip");
  tip.hidden = true;
  doc.body.append(tip);
  let current: HTMLElement | null = null;
  let byPointer = false;
  let inside = false;
  let px = 0;
  let py = 0;

  const hintOf = (node: EventTarget | null) =>
    node instanceof Element ? node.closest<HTMLElement>("[data-hint]") : null;
  const hide = () => {
    current?.removeAttribute("aria-describedby");
    current = null;
    tip.hidden = true;
  };
  const show = (el: HTMLElement) => {
    const text = el.dataset.hint;
    if (!text) return hide();
    if (current !== el) current?.removeAttribute("aria-describedby");
    current = el;
    const head = el.dataset.hintHead;
    const bold = doc.createElement("b");
    bold.textContent = head ?? "";
    const lines = text.split("\n").map((line) => {
      const p = doc.createElement("div");
      p.textContent = line;
      return p;
    });
    tip.replaceChildren(...(head ? [bold] : []), ...lines);
    // Measured at the corner: near the right edge a fixed box shrinks to the room left of it.
    tip.style.left = "0px";
    tip.style.top = "0px";
    tip.hidden = false;
    const at = placeHint(el.getBoundingClientRect(), tip.offsetWidth, tip.offsetHeight, innerWidth, innerHeight);
    tip.style.left = `${at.left}px`;
    tip.style.top = `${at.top}px`;
    el.setAttribute("aria-describedby", tip.id);
  };
  // The page re-renders or scrolls under a still pointer (live figures, a rebuilt tab): show what is under it now,
  // where it is now.
  let frame = 0;
  const refresh = () => {
    frame = 0;
    if (!current) return;
    const el = !byPointer && current.isConnected ? current : inside ? hintOf(doc.elementFromPoint(px, py)) : null;
    if (el) show(el);
    else hide();
  };
  const later = () => {
    if (current && !frame) frame = requestAnimationFrame(refresh);
  };
  const observer = new MutationObserver((records) => {
    if (records.some((r) => !tip.contains(r.target))) later();
  });
  observer.observe(doc.body, {
    subtree: true,
    childList: true,
    attributes: true,
    attributeFilter: ["data-hint", "data-hint-head"],
  });

  const over = (e: PointerEvent) => {
    px = e.clientX;
    py = e.clientY;
    byPointer = true;
    inside = true;
    const el = hintOf(e.target);
    if (el) show(el);
    else if (current) hide();
  };
  const out = (e: PointerEvent) => {
    inside = e.relatedTarget != null;
    if (current && !(e.relatedTarget instanceof Node && current.contains(e.relatedTarget))) hide();
  };
  const move = (e: PointerEvent) => {
    px = e.clientX;
    py = e.clientY;
  };
  const focus = (e: FocusEvent) => {
    const el = hintOf(e.target);
    if (!el) return;
    byPointer = false;
    show(el);
  };
  const blur = () => {
    if (!byPointer) hide();
  };
  const key = (e: KeyboardEvent) => {
    if (e.key === "Escape") hide();
  };
  const listeners: [string, EventListener][] = [
    ["pointerover", over as EventListener],
    ["pointerout", out as EventListener],
    ["pointermove", move as EventListener],
    ["focusin", focus as EventListener],
    ["focusout", blur],
    ["keydown", key as EventListener],
  ];
  for (const [type, fn] of listeners) doc.addEventListener(type, fn);
  doc.addEventListener("scroll", later, true);
  return () => {
    for (const [type, fn] of listeners) doc.removeEventListener(type, fn);
    doc.removeEventListener("scroll", later, true);
    observer.disconnect();
    cancelAnimationFrame(frame);
    tip.remove();
  };
}
