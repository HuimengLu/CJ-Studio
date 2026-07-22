"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { ArrowsOutSimpleIcon, DownloadSimpleIcon, XIcon } from "@/components/icons";

/* Library — the last 50 exported photos, saved automatically on every Export
   (deduped by content; oldest evicted beyond the cap; survives restarts).
   Grid mirrors the platform console's Images page: hairline wireframe that is
   present even with no photos; 4 columns wide, stepping down to 2. Items are
   grouped by day once they span multiple dates. */

type Item = {
  id: string;
  name: string;
  kind: string;      // "after" (white-plate) | "cover"
  ratio: string;
  w: number;
  h: number;
  created: number;
};

const fmtDate = (ts: number) =>
  new Date(ts * 1000).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });

/* "Today" / "Yesterday" / "Jul 21, 2026" */
const dayLabel = (ts: number) => {
  const d = new Date(ts * 1000);
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(today.getDate() - 1);
  if (d.toDateString() === today.toDateString()) return "Today";
  if (d.toDateString() === yesterday.toDateString()) return "Yesterday";
  return fmtDate(ts);
};

export default function LibraryPage() {
  const [items, setItems] = useState<Item[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [sel, setSel] = useState<Item | null>(null);
  const [confirmDel, setConfirmDel] = useState<Item | null>(null);
  const [cols, setCols] = useState(4);
  // Paging animation: which side the incoming image slides from (+1 = next,
  // -1 = prev, 0 = plain fade), and whether it has loaded (the slide plays on
  // load, not on keypress — animating a blank frame helps no one).
  const [pageDir, setPageDir] = useState(0);
  const [imgLoaded, setImgLoaded] = useState(false);

  const openDetail = useCallback((item: Item, dir = 0) => {
    setPageDir(dir);
    setImgLoaded(false);
    setSel(item);
  }, []);

  useEffect(() => {
    fetch("/api/library")
      .then((r) => r.json())
      .then((j) => setItems(j.items ?? []))
      .catch(() => {})
      .finally(() => setLoaded(true));
  }, []);

  // Track the active column count so each day group can pad its last row
  // with wireframe blanks (must match the grid's media queries).
  useEffect(() => {
    const mq3 = window.matchMedia("(max-width: 1400px)");
    const mq2 = window.matchMedia("(max-width: 1000px)");
    const update = () => setCols(mq2.matches ? 2 : mq3.matches ? 3 : 4);
    update();
    mq3.addEventListener("change", update);
    mq2.addEventListener("change", update);
    return () => {
      mq3.removeEventListener("change", update);
      mq2.removeEventListener("change", update);
    };
  }, []);

  // Keyboard: Esc closes (confirm first, then detail); ←/→ browse in detail.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (confirmDel) setConfirmDel(null);
        else if (sel) setSel(null);
        return;
      }
      if ((e.key === "ArrowRight" || e.key === "ArrowLeft") && sel && !confirmDel) {
        const dir = e.key === "ArrowRight" ? 1 : -1;
        const idx = items.findIndex((i) => i.id === sel.id);
        const next = idx + dir;
        if (next >= 0 && next < items.length) openDetail(items[next], dir);
        e.preventDefault();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [sel, confirmDel, items, openDetail]);

  // Preload the neighbours of the open item so arrow-paging feels instant —
  // the real latency here is the multi-MB PNG, not the animation.
  useEffect(() => {
    if (!sel) return;
    const idx = items.findIndex((i) => i.id === sel.id);
    for (const n of [idx - 1, idx + 1]) {
      if (n >= 0 && n < items.length) {
        const img = new window.Image();
        img.src = `/api/library/${items[n].id}/image`;
      }
    }
  }, [sel, items]);

  const download = useCallback((it: Item) => {
    const a = document.createElement("a");
    a.href = `/api/library/${it.id}/image`;
    a.download = it.name;
    a.click();
  }, []);

  const remove = useCallback(async (it: Item) => {
    await fetch(`/api/library/${it.id}`, { method: "DELETE" }).catch(() => {});
    setItems((xs) => xs.filter((x) => x.id !== it.id));
    setConfirmDel(null);
    setSel((s) => (s?.id === it.id ? null : s));
  }, []);

  // Items grouped by calendar day, newest day first (items arrive newest-first).
  const groups = useMemo(() => {
    const m = new Map<string, Item[]>();
    for (const it of items) {
      const key = new Date(it.created * 1000).toDateString();
      const g = m.get(key);
      if (g) g.push(it);
      else m.set(key, [it]);
    }
    return [...m.values()];
  }, [items]);

  const pad = (n: number) => (cols - (n % cols)) % cols;

  const cell = (it: Item) => (
    <div key={it.id} className="cj-lib-cell" onDoubleClick={() => openDetail(it)}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img className="cj-lib-img" src={`/api/library/${it.id}/thumb`} alt={it.name} loading="lazy" />
      <div className="cj-lib-actions">
        <button className="cj-lib-iconbtn tl" title="Delete" aria-label={`Delete ${it.name}`}
          onClick={() => setConfirmDel(it)}>
          <XIcon size={20} />
        </button>
        <button className="cj-lib-iconbtn bl" title="Full screen" aria-label={`View ${it.name}`}
          onClick={() => openDetail(it)}>
          <ArrowsOutSimpleIcon size={20} />
        </button>
        <button className="cj-lib-iconbtn br" title="Download" aria-label={`Download ${it.name}`}
          onClick={() => download(it)}>
          <DownloadSimpleIcon size={20} />
        </button>
      </div>
    </div>
  );

  const blankRow = (count: number, keyPrefix: string) =>
    Array.from({ length: count }, (_, i) => (
      <div key={`${keyPrefix}-${i}`} className="cj-lib-cell blank" />
    ));

  return (
    <div className="cj-lib">
      {items.length > 0 ? (
        groups.map((g, gi) => (
          <div key={gi}>
            {/* Day headers only once exports span multiple days. */}
            {groups.length > 1 && <div className="cj-lib-day">{dayLabel(g[0].created)}</div>}
            <div className="cj-lib-grid">
              {g.map(cell)}
              {blankRow(pad(g.length), `pad-${gi}`)}
            </div>
          </div>
        ))
      ) : (
        <div className="cj-lib-grid">{blankRow(cols * 6, "empty")}</div>
      )}

      {loaded && items.length === 0 && (
        <div className="cj-lib-empty">Photos you export will be saved here automatically.</div>
      )}

      {/* ── full-screen detail (double-click or the corners icon; ←/→ browse) ── */}
      {sel && (
        <div className="cj-lib-detail" role="dialog" aria-label={sel.name}>
          <div className="cj-lib-detail-stage">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              key={sel.id}
              src={`/api/library/${sel.id}/image`}
              alt={sel.name}
              className={imgLoaded ? "in" : ""}
              style={{ "--page-from": `${pageDir * 12}px` } as React.CSSProperties}
              onLoad={() => setImgLoaded(true)}
              onError={() => setImgLoaded(true)}
            />
          </div>
          <div className="cj-lib-detail-side">
            <div className="cj-lib-detail-meta">
              <span>
                {sel.kind === "cover" ? "cover scene"
                  : sel.kind === "social" ? "social graphic" : "white backdrop"}
              </span>
              <span>ratio: {sel.ratio}</span>
              <span>size: {sel.w}x{sel.h}</span>
              <span>saved: {fmtDate(sel.created)}</span>
            </div>
            <button className="cj-lib-download" onClick={() => download(sel)}>
              <DownloadSimpleIcon size={16} />
              <span>Download</span>
            </button>
          </div>
          <button className="cj-lib-close" aria-label="Close" onClick={() => setSel(null)}>
            ✕
          </button>
        </div>
      )}

      {/* ── delete confirm (shared dialog design) ── */}
      {confirmDel && (
        <div className="cj-modal-scrim" onClick={() => setConfirmDel(null)}>
          <div className="cj-modal small" onClick={(e) => e.stopPropagation()}>
            <div className="cj-modal-head"><h2>Delete photo</h2></div>
            <div className="cj-modal-text">
              Remove <b>{confirmDel.name}</b> from the library? This cannot be undone.
            </div>
            <div className="cj-modal-foot">
              <button className="cj-modal-cancel" onClick={() => setConfirmDel(null)}>Cancel</button>
              <button className="cj-modal-go" onClick={() => remove(confirmDel)}>Delete</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
