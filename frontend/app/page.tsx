"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import CompareSlider from "@/components/CompareSlider";
import DimensionOverlay, { type Dimension } from "@/components/DimensionOverlay";

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
  textMode: boolean;
  origBg: boolean;        // 25% original photo between backdrop and subject
  caption: string;        // input draft
  applied: string;        // caption actually rendered (committed via Apply)
  dimensions: Dimension[];       // measurement annotations (normalized coords)
};

type ProcRow = {
  name: string;
  thumb: string | null;   // object URL
  state: "waiting" | "active" | "done" | "error";
  warn?: string;
};

const imgUrl = (p: Photo, kind: "after" | "before", w?: number) =>
  `/api/photos/${p.id}/image?kind=${kind}&ratio=${encodeURIComponent(p.ratio)}` +
  `&text=${p.textMode ? 1 : 0}&caption=${encodeURIComponent(p.textMode ? p.applied : "")}` +
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

export default function ListingPage() {
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
  const sliderCtl = useRef<{ toLeft: () => void } | null>(null);
  const cancelled = useRef(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const moreRef = useRef<HTMLInputElement>(null);

  const photo = photos[active];

  const mutatePhoto = useCallback((patch: Partial<Photo>) => {
    setImgLoading(true);
    setPhotos((ps) => ps.map((p, i) => (i === active ? { ...p, ...patch } : p)));
  }, [active]);

  /* Return to the empty upload screen. Fired by the sidebar's Listing item —
     clicking it while already on "/" can't remount this page, so it dispatches
     a "cj:reset-listing" event that we handle here. */
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
  }, []);

  useEffect(() => {
    window.addEventListener("cj:reset-listing", resetToHome);
    return () => window.removeEventListener("cj:reset-listing", resetToHome);
  }, [resetToHome]);

  /* ── initial batch: one row per file, processed sequentially ── */
  const startBatch = useCallback(async (files: File[]) => {
    files = files.filter((f) => f.type.startsWith("image/"));
    if (!files.length) return;
    cancelled.current = false;
    setRows(files.map((f) => ({ name: f.name, thumb: URL.createObjectURL(f), state: "waiting" })));
    setPhase("processing");

    const got: Photo[] = [];
    const warns: string[] = [];
    for (let i = 0; i < files.length; i++) {
      if (cancelled.current) return;
      setRows((r) => r.map((row, j) => (j === i ? { ...row, state: "active" } : row)));
      const out = await processFile(files[i]);
      if (cancelled.current) return;
      if (out.id) {
        got.push({ id: out.id, name: files[i].name, ratio: "1:1", textMode: false, origBg: false, caption: "", applied: "", dimensions: [] });
        setRows((r) => r.map((row, j) => (j === i ? { ...row, state: "done" } : row)));
      } else {
        warns.push(out.warn ?? files[i].name);
        setRows((r) => r.map((row, j) => (j === i ? { ...row, state: "error", warn: out.warn } : row)));
      }
    }
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
          id: out.id!, name: files[i].name, ratio: "1:1", textMode: false, origBg: false, caption: "", applied: "", dimensions: [],
        }]);
      } else {
        warns.push(out.warn ?? files[i].name);
      }
      setPending((t) => t.map((x, j) => (j === i ? { ...x, state: "done" } : x)));
    }
    setPending([]);
    setWarn(warns.length ? warns.join("  ") : null);
  }, []);

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
      .map((p) => ({ id: p.id, ratio: p.ratio, text_mode: p.textMode, caption: p.textMode ? p.applied : "", orig_bg: p.origBg }));
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
      [{ id: photo.id, ratio: photo.ratio, text_mode: photo.textMode, caption: photo.textMode ? photo.applied : "", orig_bg: photo.origBg }],
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
          <div className="cj-drop-icon"><span className="ms">add_photo_alternate</span></div>
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
  const showApply = photo.caption.trim() !== "" && photo.caption !== photo.applied;

  return (
    <div className="cj-result">
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

        {(multi || pending.length > 0) && (
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
            <p className="cj-sec">Output Ratio</p>
            <div className="cj-ratio-row">
              {RATIOS.map((r) => (
                <button
                  key={r.id}
                  className={`cj-ratio-card${photo.ratio === r.id ? " active" : ""}`}
                  onClick={() => photo.ratio !== r.id && mutatePhoto({ ratio: r.id })}
                >
                  {r.id}
                  <small>{r.label}</small>
                </button>
              ))}
            </div>
          </div>
          <div className="cj-grp">
            <div className="cj-toggle-row">
              <p className="cj-sec">Text Overlay</p>
              <button
                className={`cj-switch${photo.textMode ? " on" : ""}`}
                onClick={() => mutatePhoto({ textMode: !photo.textMode })}
                aria-label="Toggle text overlay"
              />
            </div>
            <div className={`cj-desc-row${photo.textMode ? " open" : ""}`}>
              <input
                className="cj-desc-input"
                placeholder="Description for bottom-right overlay..."
                value={photo.caption}
                onChange={(e) =>
                  setPhotos((ps) => ps.map((p, i) => (i === active ? { ...p, caption: e.target.value } : p)))
                }
                onKeyDown={(e) => { if (e.key === "Enter" && showApply) mutatePhoto({ applied: photo.caption }); }}
              />
              {showApply && (
                <button className="cj-apply" onClick={() => mutatePhoto({ applied: photo.caption })}>
                  Apply
                </button>
              )}
            </div>
          </div>
          <div className="cj-grp">
            <div className="cj-toggle-row">
              <p className="cj-sec">Original Background</p>
              <button
                className={`cj-switch${photo.origBg ? " on" : ""}`}
                onClick={() => mutatePhoto({ origBg: !photo.origBg })}
                aria-label="Toggle original background"
              />
            </div>
          </div>
          {warn && <div className="cj-warn">&#9888; {warn}</div>}
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
          <button
            className="cj-btn-outline"
            onClick={() => { setDimMode(true); sliderCtl.current?.toLeft(); }}
          >
            Add Dimensions
          </button>
        </div>
      </aside>

      {/* ── export selection modal (all selected by default) ── */}
      {exportOpen && (
        <div className="cj-modal-scrim" onClick={() => setExportOpen(false)}>
          <div className="cj-modal" onClick={(e) => e.stopPropagation()}>
            <div className="cj-modal-head">
              <h2>Export Selection</h2>
              <button className="cj-modal-x" onClick={() => setExportOpen(false)}>
                <span className="ms">close</span>
              </button>
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
                      <span className="ms" style={{ fontSize: 15 }}>
                        {exportSel.has(i) ? "check_circle" : "circle"}
                      </span>
                    </span>
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
            <div className="cj-modal-head"><h2 style={{ fontSize: 24, lineHeight: "32px" }}>Delete photo</h2></div>
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
