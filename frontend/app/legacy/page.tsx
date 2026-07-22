"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import CompareSlider from "@/components/CompareSlider";
import DimensionOverlay, { type Dimension } from "@/components/DimensionOverlay";
import { ArrowClockwiseIcon, CameraPlusIcon, CheckCircleIcon, CircleIcon } from "@/components/icons";
import OptionCard from "@/components/OptionCard";

type Ratio = "1:1" | "4:3" | "4:5";
const RATIOS: { id: Ratio; label: string }[] = [
  { id: "1:1", label: "EBAY" },
  { id: "4:3", label: "WEBSITE" },
  { id: "4:5", label: "INSTAGRAM" },
];
/** Numeric width/height per ratio — drives the preview box's aspect-ratio. */
const RATIO_AR: Record<Ratio, number> = { "1:1": 1, "4:3": 4 / 3, "4:5": 4 / 5 };

/** One processed photo; edit state is per-photo so edits never leak across. */
type Photo = {
  id: string;
  name: string;
  ratio: Ratio;
  origBg: boolean;        // 25% original photo between backdrop and subject
  dimensions: Dimension[];       // measurement annotations (normalized coords)
};

type ProcRow = {
  name: string;
  thumb: string | null;   // object URL
  state: "waiting" | "active" | "done" | "error";
  warn?: string;
};

/* A failed upload kept in the filmstrip so it can be retried in place —
   the File handle is retained for the retry request. */
type FailedRow = {
  name: string;
  thumb: string | null;
  file: File;
  warn?: string;
  busy?: boolean;
};

const imgUrl = (p: Photo, kind: "after" | "before", w?: number) =>
  `/api/photos/${p.id}/image?kind=${kind}&ratio=${encodeURIComponent(p.ratio)}` +
  `&origbg=${p.origBg ? 1 : 0}` +
  (w ? `&w=${w}` : "");

async function processFile(f: File): Promise<{ id?: string; warn?: string }> {
  // one retry: the dev proxy occasionally reuses a keep-alive connection the
  // backend already closed (ECONNRESET) — a fresh attempt gets a new socket
  for (let attempt = 0; attempt < 2; attempt++) {
    const fd = new FormData();
    fd.append("file", f);
    try {
      const res = await fetch("/api/photos", { method: "POST", body: fd });
      const j = await res.json();
      if (!res.ok || !j.id) return { warn: j.warn ?? `${f.name}: processing failed` };
      return { id: j.id };
    } catch {
      if (attempt === 1) return { warn: `${f.name}: network error` };
    }
  }
  return { warn: `${f.name}: network error` };
}

async function downloadExport(items: object[], fallback: string) {
  const res = await fetch("/api/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items }),
  });
  const blob = await res.blob();
  const m = (res.headers.get("Content-Disposition") ?? "").match(/filename="([^"]+)"/);
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = m?.[1] ?? fallback;
  a.click();
  URL.revokeObjectURL(a.href);
}

function StepsStrip() {
  return (
    <div className="cj-strip">
      <div className="cj-step"><span className="cj-num">1</span><span className="cj-steplbl">Upload</span></div>
      <span className="cj-dash">—</span>
      <div className="cj-step"><span className="cj-num">2</span><span className="cj-steplbl">Algorithm Optimize</span></div>
      <span className="cj-dash">—</span>
      <div className="cj-step"><span className="cj-num">3</span><span className="cj-steplbl">Export</span></div>
    </div>
  );
}

export default function LegacyListingPage() {
  const [phase, setPhase] = useState<"home" | "processing" | "result">("home");
  const [rows, setRows] = useState<ProcRow[]>([]);
  const [photos, setPhotos] = useState<Photo[]>([]);
  const [active, setActive] = useState(0);
  const [warn, setWarn] = useState<string | null>(null);
  const [animate, setAnimate] = useState(false);
  const [imgLoading, setImgLoading] = useState(false);
  const [pending, setPending] = useState<ProcRow[]>([]);
  const [confirmDelete, setConfirmDelete] = useState<number | null>(null);
  const [exportOpen, setExportOpen] = useState(false);
  const [exportSel, setExportSel] = useState<Set<number>>(new Set());
  const [exporting, setExporting] = useState(false);
  const [dragover, setDragover] = useState(false);
  const [dimMode, setDimMode] = useState(false);
  const [failed, setFailed] = useState<FailedRow[]>([]);
  const sliderCtl = useRef<{ toLeft: () => void } | null>(null);
  const cancelled = useRef(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const moreRef = useRef<HTMLInputElement>(null);

  const photo = photos[active];

  const mutatePhoto = useCallback((patch: Partial<Photo>) => {
    setImgLoading(true);
    setPhotos((ps) => ps.map((p, i) => (i === active ? { ...p, ...patch } : p)));
  }, [active]);

  /* Return to the empty upload screen. Fired by the sidebar's Legacy Listing
     item — clicking it while already on /legacy can't remount this page, so it
     dispatches a "cj:reset-legacy" event that we handle here. */
  const resetToHome = useCallback(() => {
    cancelled.current = true;
    setPhase("home");
    setRows([]);
    setPhotos([]);
    setPending([]);
    setActive(0);
    setWarn(null);
    setDimMode(false);
    setExportOpen(false);
    setConfirmDelete(null);
    setFailed([]);
  }, []);

  useEffect(() => {
    window.addEventListener("cj:reset-legacy", resetToHome);
    return () => window.removeEventListener("cj:reset-legacy", resetToHome);
  }, [resetToHome]);

  // Esc leaves dimension-placing mode (matches the Library's Esc habit).
  useEffect(() => {
    if (!dimMode) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setDimMode(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [dimMode]);

  /* ── initial batch: one row per file, processed sequentially ── */
  const startBatch = useCallback(async (files: File[]) => {
    files = files.filter((f) => f.type.startsWith("image/"));
    if (!files.length) return;
    cancelled.current = false;
    setRows(files.map((f) => ({ name: f.name, thumb: URL.createObjectURL(f), state: "waiting" })));
    setPhase("processing");

    const got: Photo[] = [];
    const fails: FailedRow[] = [];
    const warns: string[] = [];
    for (let i = 0; i < files.length; i++) {
      if (cancelled.current) return;
      setRows((r) => r.map((row, j) => (j === i ? { ...row, state: "active" } : row)));
      const out = await processFile(files[i]);
      if (cancelled.current) return;
      if (out.id) {
        got.push({ id: out.id, name: files[i].name, ratio: "1:1", origBg: false, dimensions: [] });
        setRows((r) => r.map((row, j) => (j === i ? { ...row, state: "done" } : row)));
      } else {
        fails.push({ name: files[i].name, thumb: URL.createObjectURL(files[i]), file: files[i], warn: out.warn });
        warns.push(`${files[i].name}: ${out.warn ?? "failed"}`);
        setRows((r) => r.map((row, j) => (j === i ? { ...row, state: "error", warn: out.warn } : row)));
      }
    }
    setFailed(fails);
    setWarn(warns.length ? warns.join("  ") : null);
    if (got.length) {
      setPhotos(got);
      setActive(0);
      setAnimate(true);
      setImgLoading(false);
      setPhase("result");
    } else {
      setPhase("home");
    }
  }, []);

  /* ── add more from the filmstrip: spinner tiles walk across in place ── */
  const addMore = useCallback(async (files: File[]) => {
    files = files.filter((f) => f.type.startsWith("image/"));
    if (!files.length) return;
    setPending(files.map((f) => ({ name: f.name, thumb: URL.createObjectURL(f), state: "waiting" })));
    const warns: string[] = [];
    for (let i = 0; i < files.length; i++) {
      setPending((t) => t.map((x, j) => (j === i ? { ...x, state: "active" } : x)));
      const out = await processFile(files[i]);
      if (out.id) {
        setPhotos((ps) => [...ps, {
          id: out.id!, name: files[i].name, ratio: "1:1", origBg: false, dimensions: [],
        }]);
      } else {
        setFailed((fs) => [...fs, {
          name: files[i].name, thumb: URL.createObjectURL(files[i]), file: files[i], warn: out.warn,
        }]);
        warns.push(`${files[i].name}: ${out.warn ?? "failed"}`);
      }
      setPending((t) => t.map((x, j) => (j === i ? { ...x, state: "done" } : x)));
    }
    setPending([]);
    setWarn(warns.length ? warns.join("  ") : null);
  }, []);

  /* Retry a failed upload in place — the tile spins, then either becomes a
     real photo or stays with an updated message. */
  const retryFailed = useCallback(async (idx: number) => {
    const row = failed[idx];
    if (!row || row.busy) return;
    setFailed((fs) => fs.map((f, j) => (j === idx ? { ...f, busy: true } : f)));
    const out = await processFile(row.file);
    if (out.id) {
      setPhotos((ps) => [...ps, { id: out.id!, name: row.name, ratio: "1:1", origBg: false, dimensions: [] }]);
      setFailed((fs) => fs.filter((f) => f.file !== row.file));
      setWarn(null);
    } else {
      setFailed((fs) => fs.map((f, j) => (j === idx ? { ...f, busy: false, warn: out.warn } : f)));
      setWarn(`${row.name}: ${out.warn ?? "failed"}`);
    }
  }, [failed]);

  const deletePhoto = useCallback((i: number) => {
    const target = photos[i];
    if (target) fetch(`/api/photos/${target.id}`, { method: "DELETE" }).catch(() => {});
    const next = photos.filter((_, j) => j !== i);
    setPhotos(next);
    if (!next.length) {
      setPhase("home");
      setActive(0);
    } else {
      setActive((a) => Math.max(0, Math.min(i < a ? a - 1 : a, next.length - 1)));
    }
    setConfirmDelete(null);
  }, [photos]);

  const exportSelected = useCallback(async () => {
    const items = photos
      .filter((_, i) => exportSel.has(i))
      .map((p) => ({ id: p.id, ratio: p.ratio, orig_bg: p.origBg, dimensions: p.dimensions }));
    if (!items.length) return;
    setExporting(true);
    try {
      await downloadExport(items, items.length > 1 ? "cj_photos.zip" : "cj_photo.png");
      setExportOpen(false);
    } finally {
      setExporting(false);
    }
  }, [photos, exportSel]);

  const exportSingle = useCallback(() => {
    if (!photo) return;
    void downloadExport(
      [{ id: photo.id, ratio: photo.ratio, orig_bg: photo.origBg, dimensions: photo.dimensions }],
      `cj_listing_${photo.ratio.replace(":", "x")}.png`,
    );
  }, [photo]);

  const multi = photos.length > 1;
  const donePct = useMemo(() => {
    const done = rows.filter((r) => r.state === "done" || r.state === "error").length;
    return rows.length ? Math.round((done / rows.length) * 100) : 0;
  }, [rows]);

  /* ═══ home ═══ */
  if (phase === "home") {
    return (
      <div className="cj-home">
        <div className="cj-home-top" />
        {warn && <div className="cj-warn">&#9888; {warn}</div>}
        <div
          className={`cj-drop${dragover ? " dragover" : ""}`}
          onClick={() => fileRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragover(true); }}
          onDragLeave={() => setDragover(false)}
          onDrop={(e) => { e.preventDefault(); setDragover(false); startBatch(Array.from(e.dataTransfer.files)); }}
        >
          <div className="cj-drop-icon"><CameraPlusIcon size={40} /></div>
          <div className="cj-drop-title">Click or drop files</div>
          <div className="cj-drop-hint">JPG · PNG · WEBP / MAX 200MB</div>
          <input
            ref={fileRef} type="file" multiple hidden
            accept="image/jpeg,image/png,image/webp"
            onChange={(e) => { startBatch(Array.from(e.target.files ?? [])); e.target.value = ""; }}
          />
        </div>
        <StepsStrip />
      </div>
    );
  }

  /* ═══ processing ═══ */
  if (phase === "processing") {
    return (
      <div className="cj-home">
        <div className="cj-home-top" />
        <div className="cj-proc">
          {rows.length > 1 && (
            <div>
              <div className="cj-proc-head">
                <span className="cj-proc-head-label">Batch Progress</span>
                <span className="cj-proc-head-pct">{donePct}%</span>
              </div>
              <div className="cj-proc-head-track">
                <div className="cj-proc-head-fill" style={{ width: `${donePct}%` }} />
              </div>
            </div>
          )}
          <div className="cj-proc-list">
            {rows.map((r, i) => (
              <div key={i} className={`cj-proc-row ${r.state}`}>
                {r.thumb
                  ? /* eslint-disable-next-line @next/next/no-img-element */
                    <img className="cj-proc-thumb" src={r.thumb} alt="" />
                  : <div className="cj-proc-thumb" style={{ background: "var(--line-soft)" }} />}
                <div className="cj-proc-body">
                  <div className="cj-proc-top">
                    <span className="cj-proc-name">{r.name}</span>
                    <span className="cj-proc-status">
                      {{ waiting: "Waiting…", active: "Optimizing…", done: "Done", error: "Failed" }[r.state]}
                    </span>
                  </div>
                  <div className="cj-proc-track"><div className="cj-proc-fill" /></div>
                </div>
              </div>
            ))}
          </div>
          <button className="cj-proc-cancel" onClick={() => { cancelled.current = true; setPhase("home"); }}>
            Cancel
          </button>
        </div>
        <StepsStrip />
      </div>
    );
  }

  /* ═══ result ═══ */
  if (!photo) return null;

  return (
    <div className="cj-result">
      {/* Global notices float over the stage — panel-buried warnings get missed. */}
      {warn && <div className="cj-warn cj-warn-float">&#9888; {warn}</div>}
      <div className="cj-canvas-col">
        <div className="cj-stage">
          {/* key on photo.id only: switching photos remounts (fresh slider +
              reveal), but editing the same photo — ratio / text / caption —
              keeps the component so the divider stays where the user left it;
              only the image srcs update. */}
          <CompareSlider
            key={photo.id}
            beforeSrc={imgUrl(photo, "before", 1400)}
            afterSrc={imgUrl(photo, "after", 1400)}
            aspectRatio={RATIO_AR[photo.ratio]}
            animate={animate}
            onAfterLoaded={() => { setAnimate(false); setImgLoading(false); }}
            controlRef={sliderCtl}
            overlay={
              (dimMode || photo.dimensions.length > 0) && (
                <DimensionOverlay
                  key={photo.id}
                  adding={dimMode}
                  dimensions={photo.dimensions}
                  onChange={(ds) => setPhotos((ps) => ps.map((p, i) => (i === active ? { ...p, dimensions: ds } : p)))}
                  onExitAdding={() => setDimMode(false)}
                  onInteract={() => sliderCtl.current?.toLeft()}
                />
              )
            }
          />
          {imgLoading && <div className="cj-stage-loading"><div className="cj-spin" /></div>}
        </div>

        {/* Always present — the "+" tile is the only way to add more photos. */}
        {(
          <div className="cj-film">
            {photos.map((p, i) => (
              <div key={p.id} className="cj-tw">
                <button
                  className={`cj-thumb${i === active ? " active" : ""}`}
                  style={{ backgroundImage: `url(/api/photos/${p.id}/thumb)` }}
                  onClick={() => { if (i !== active) { setActive(i); setImgLoading(true); setDimMode(false); } }}
                  aria-label={p.name}
                />
                <button className="cj-delbtn" onClick={() => setConfirmDelete(i)} aria-label={`Delete ${p.name}`}>
                  ✕
                </button>
              </div>
            ))}
            {pending.map((t, i) => (
              <div
                key={`pend-${i}`}
                className={`cj-pend${t.state === "active" ? " spin" : t.state === "waiting" ? " dim" : ""}`}
                style={t.thumb ? { backgroundImage: `url(${t.thumb})` } : undefined}
              />
            ))}
            {failed.map((f, i) => (
              <button
                key={`fail-${i}`}
                className={`cj-fail${f.busy ? " busy" : ""}`}
                style={f.thumb ? { backgroundImage: `url(${f.thumb})` } : undefined}
                title={`${f.name} failed — click to retry${f.warn ? `\n${f.warn}` : ""}`}
                onClick={() => retryFailed(i)}
              >
                {!f.busy && <ArrowClockwiseIcon size={22} />}
              </button>
            ))}
            <button className="cj-addtile" onClick={() => moreRef.current?.click()} aria-label="Add photos">
              +
            </button>
          </div>
        )}
        <input
          ref={moreRef} type="file" multiple hidden
          accept="image/jpeg,image/png,image/webp"
          onChange={(e) => { addMore(Array.from(e.target.files ?? [])); e.target.value = ""; }}
        />
      </div>

      {/* ── right edit panel (Figma 182:544 / 182:605) ── */}
      <aside className="cj-panel">
        <div className="cj-panel-body">
          <div className="cj-grp">
            <div className="cj-field">
              <span className="cj-label">Output Ratio</span>
              <div className="cj-ratio-row">
                {RATIOS.map((r) => (
                  <OptionCard
                    key={r.id}
                    active={photo.ratio === r.id}
                    title={r.id}
                    label={r.label}
                    onClick={() => photo.ratio !== r.id && mutatePhoto({ ratio: r.id })}
                  />
                ))}
              </div>
            </div>
            <div className="cj-field">
              <div className="cj-toggle-row">
                <span className="cj-label">Original Background</span>
                <button
                  className={`cj-switch${photo.origBg ? " on" : ""}`}
                  onClick={() => mutatePhoto({ origBg: !photo.origBg })}
                  aria-label="Toggle original background"
                />
              </div>
              <p className="cj-hint">Blends the original photo into the backdrop at 25% opacity</p>
            </div>
          </div>
          <div className="cj-grp cj-actions">
            <button
              className={`cj-btn-outline${dimMode ? " active" : ""}`}
              onClick={() => {
                if (dimMode) setDimMode(false);
                else { setDimMode(true); sliderCtl.current?.toLeft(); }
              }}
            >
              {dimMode ? "Placing points — Esc to exit" : "Add Dimensions"}
            </button>
          </div>
        </div>
        <div className="cj-foot">
          {multi ? (
            <button
              className="cj-btn-primary"
              onClick={() => { setExportSel(new Set(photos.map((_, i) => i))); setExportOpen(true); }}
            >
              Export
            </button>
          ) : (
            <button className="cj-btn-primary" onClick={exportSingle}>Export image</button>
          )}
        </div>
      </aside>

      {/* ── export selection modal (all selected by default) ── */}
      {exportOpen && (
        <div className="cj-modal-scrim" onClick={() => setExportOpen(false)}>
          <div className="cj-modal" onClick={(e) => e.stopPropagation()}>
            <div className="cj-modal-head">
              <h2>Export Selection</h2>
            </div>
            <div className="cj-modal-body">
              <div className="cj-modal-grid">
                {photos.map((p, i) => (
                  <button
                    key={p.id}
                    className={`cj-ex-item${exportSel.has(i) ? " sel" : ""}`}
                    onClick={() =>
                      setExportSel((s) => {
                        const n = new Set(s);
                        if (n.has(i)) n.delete(i); else n.add(i);
                        return n;
                      })
                    }
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={`/api/photos/${p.id}/thumb?s=320`} alt={p.name} />
                    <span className="cj-ex-badge">
                      {exportSel.has(i) ? <CheckCircleIcon size={15} /> : <CircleIcon size={15} />}
                    </span>
                    {p.dimensions.length > 0 && (
                      <span className="cj-ex-flags">
                        <span className="cj-ex-flag">
                          {p.dimensions.length} dim{p.dimensions.length > 1 ? "s" : ""}
                        </span>
                      </span>
                    )}
                  </button>
                ))}
              </div>
            </div>
            <div className="cj-modal-foot">
              <button className="cj-modal-cancel" onClick={() => setExportOpen(false)}>Cancel</button>
              <button className="cj-modal-go" disabled={exportSel.size === 0 || exporting} onClick={exportSelected}>
                {exporting ? "Exporting…" : `Export Selected (${exportSel.size})`}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── delete confirm ── */}
      {confirmDelete !== null && photos[confirmDelete] && (
        <div className="cj-modal-scrim" onClick={() => setConfirmDelete(null)}>
          <div className="cj-modal small" onClick={(e) => e.stopPropagation()}>
            <div className="cj-modal-head"><h2>Delete photo</h2></div>
            <div className="cj-modal-text">
              Remove <b>{photos[confirmDelete].name}</b> from this batch? Its edits will be lost.
            </div>
            <div className="cj-modal-foot">
              <button className="cj-modal-cancel" onClick={() => setConfirmDelete(null)}>Cancel</button>
              <button className="cj-modal-go" onClick={() => deletePhoto(confirmDelete)}>Delete</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
