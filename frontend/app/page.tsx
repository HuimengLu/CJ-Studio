"use client";

import { Fragment, useEffect, useRef, useState, type ReactNode } from "react";
import CompareSlider from "@/components/CompareSlider";
import DimensionOverlay from "@/components/DimensionOverlay";
import { ArrowClockwiseIcon, CameraPlusIcon, CheckIcon } from "@/components/icons";
import OptionCard from "@/components/OptionCard";
import {
  RATIOS, RATIO_AR, SESSION_EXPIRED, entryKey, imgUrl, stageW, thumbUrl, useListing,
  type Photo,
} from "./listing-state";

/* New Listing — the primary photo-processing pipeline: gpt-image-2 redraws the
   product on a white background with a studio shadow, multiply-blended onto the
   CJ backdrop. (The pre-AI local pipeline lives on at /legacy.) The backend
   routes keep their historical /api/testing2 prefix.

   Cover mode: Generate Cover classifies the product (gpt-5.4) and
   composes it into that category's scene (gpt-image-2 + static/cover bg). The
   scene lands as an EXTRA filmstrip entry (variant "cover") sharing the same
   backend id as the white-plate entry; its "before" is the original photo.

   All processing state lives in listing-state.tsx (a layout-level provider),
   so batches keep running and results survive tab switches. This component
   is the view: DOM refs and hover/drag visuals only. */

/* Filmstrip thumb: stays a flat gray tile until its image has actually
   downloaded (tracked via an off-DOM preload of the same URL). */
function ThumbButton({ p, active, onClick }: {
  p: Photo; active: boolean; onClick: () => void;
}) {
  const url = thumbUrl(p);
  const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    setLoaded(false);
    const img = new window.Image();
    img.onload = () => setLoaded(true);
    img.src = url;
    return () => { img.onload = null; };
  }, [url]);
  return (
    <button
      className={`cj-thumb${active ? " active" : ""}`}
      style={loaded ? { backgroundImage: `url(${url})` } : undefined}
      onClick={onClick}
      aria-label={p.name}
    />
  );
}

/* Modal wrapper that stays mounted through its exit animation: flipping
   `open` false plays the scrim + panel fade-down (the library detail's
   motion language at a lighter tempo) before unmounting. Prop changes are
   adapted during render (store-previous-props pattern); reduced motion
   unmounts on the next tick instead. */
function ModalShell({ open, onClose, children }: {
  open: boolean; onClose: () => void; children: ReactNode;
}) {
  const [render, setRender] = useState(open);
  const [closing, setClosing] = useState(false);
  const [prevOpen, setPrevOpen] = useState(open);
  if (open !== prevOpen) {
    setPrevOpen(open);
    if (open) { setRender(true); setClosing(false); }
    else if (render) setClosing(true);
  }
  useEffect(() => {
    if (!closing) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const t = setTimeout(() => { setRender(false); setClosing(false); }, reduce ? 0 : 380);
    return () => clearTimeout(t);
  }, [closing]);
  if (!render) return null;
  return (
    <div className={`cj-modal-scrim${closing ? " out" : ""}`} onClick={onClose}>
      {children}
    </div>
  );
}

function StepsStrip({ current = 0 }: { current?: number }) {
  const steps = ["Upload", "Enhance", "Export"];
  return (
    <div className="cj-strip">
      {steps.map((label, i) => (
        <span key={label} style={{ display: "contents" }}>
          {i > 0 && <span className="cj-dash" aria-hidden="true" />}
          <div className={`cj-step${current === i + 1 ? " active" : ""}`}>
            <span className="cj-num">{i + 1}</span>
            <span className="cj-steplbl">{label}</span>
          </div>
        </span>
      ))}
    </div>
  );
}

export default function ListingPage() {
  const {
    phase, rows, photos, active, warn, animate, imgLoading, quietLoad, pending,
    confirmDelete, exportOpen, exportSel, exporting, dimMode, coverBusy,
    failed, savedNote, newCovers, undoDelete, photo, donePct, sliderCtl,
    setPhotos, setActive, setWarn, setAnimate, setImgLoading, setQuietLoad, setDimMode,
    setConfirmDelete, setExportOpen, setExportSel,
    startBatch, cancelBatch, addMore, retryFailed, toggleCover,
    deletePhoto, undoRemove, exportSelected, exportSingle, mutatePhoto,
  } = useListing();

  const [dragover, setDragover] = useState(false);
  const [dropHover, setDropHover] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const moreRef = useRef<HTMLInputElement>(null);

  const multi = photos.length > 1;
  const coverOn = !!photo && photos.some((p) => p.id === photo.id && p.variant === "cover");
  const busy = !!photo && coverBusy.has(photo.id);

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
          onMouseEnter={() => setDropHover(true)}
          onMouseLeave={() => setDropHover(false)}
        >
          <div className="cj-drop-icon"><CameraPlusIcon size={40} /></div>
          <div className="cj-drop-title">Click or drop files</div>
          <div className="cj-drop-hint">JPG · PNG · WEBP · max 200 MB</div>
          <input
            ref={fileRef} type="file" multiple hidden
            accept="image/jpeg,image/png,image/webp"
            onChange={(e) => { startBatch(Array.from(e.target.files ?? [])); e.target.value = ""; }}
          />
        </div>
        {/* The second sentence wraps as a unit — a mid-phrase break on narrow
            screens orphans "a time."; a hard <br> would force two lines even
            where one fits. */}
        <div className="cj-drop-note">
          Got multiple photos? Upload them all.{" "}
          <span className="cj-nowrap-unit">We&apos;ll process 3 at a time.</span>
        </div>
        <StepsStrip current={dragover || dropHover ? 1 : 0} />
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
              {/* One estimate for the whole batch — photos run 3 in parallel,
                  so ≤3 photos take about one generation (~15s) total, larger
                  batches one wave per started group of 3. Per-row text stays
                  terse to avoid repeating this on every line. */}
              <div className="cj-proc-head-sub">
                Takes about {15 * Math.ceil(rows.length / 3)} sec
              </div>
              <div className="cj-proc-head-track">
                <div className="cj-proc-head-fill" style={{ width: `${donePct}%` }} />
              </div>
            </div>
          )}
          {/* Multi-photo batches: progress lives ONLY in the batch bar above;
              rows are a status checklist (spinner / green check / Waiting) —
              parallel items animate near-identically, so per-row bars were
              four moving elements telling one story. A single photo has no
              batch bar, so it keeps its own bar as the sole indicator. */}
          <div className={`cj-proc-list${rows.length > 1 ? " checklist" : ""}`}>
            {rows.map((r, i) => (
              <div key={i} className={`cj-proc-row ${r.state}`}>
                {r.thumb
                  ? /* eslint-disable-next-line @next/next/no-img-element */
                    <img className="cj-proc-thumb" src={r.thumb} alt="" />
                  : <div className="cj-proc-thumb" style={{ background: "var(--line-soft)" }} />}
                {rows.length > 1 ? (
                  <>
                    <span className="cj-proc-name">{r.name}</span>
                    <span
                      className="cj-proc-status"
                      aria-label={{ waiting: undefined, active: "Enhancing photo", done: "Done", error: undefined }[r.state]}
                    >
                      {{
                        waiting: "Waiting…",
                        active: <span className="cj-proc-spin" />,
                        done: <CheckIcon size={15} />, error: "Failed",
                      }[r.state]}
                    </span>
                  </>
                ) : (
                  <div className="cj-proc-body">
                    <div className="cj-proc-top">
                      <span className="cj-proc-name">{r.name}</span>
                      <span
                        className="cj-proc-status"
                        aria-label={r.state === "done" ? "Done" : undefined}
                      >
                        {{
                          waiting: "Waiting…",
                          active: "Enhancing photo… about 15 seconds",
                          done: <CheckIcon size={15} />, error: "Failed",
                        }[r.state]}
                      </span>
                    </div>
                    <div className="cj-proc-track"><div className="cj-proc-fill" /></div>
                  </div>
                )}
              </div>
            ))}
          </div>
          <button className="cj-proc-cancel" onClick={cancelBatch}>
            Cancel Processing
          </button>
        </div>
        <StepsStrip current={2} />
      </div>
    );
  }

  /* ═══ result ═══ */
  if (!photo) return null;

  return (
    <div className="cj-result">
      {/* Global notices float over the stage — panel-buried warnings get missed. */}
      {warn && <div className="cj-warn cj-warn-float">&#9888; {warn}</div>}
      {undoDelete && (
        <div className="cj-undo">
          <span>Photo removed</span>
          <button onClick={undoRemove}>Undo</button>
        </div>
      )}
      <div className="cj-canvas-col">
        {/* is-loading lets mobile CSS hold the stage at the finished frame's
            aspect ratio while pixels are still downloading. */}
        <div className={`cj-stage${imgLoading ? " is-loading" : ""}`}>
          <CompareSlider
            key={entryKey(photo)}
            beforeSrc={imgUrl(photo, "before", stageW())}
            afterSrc={imgUrl(photo, "after", stageW())}
            aspectRatio={RATIO_AR[photo.ratio]}
            /* Mirrors backend pipeline._TESTING2_ZOOM: 1:1 composes 1.25x,
               4:5 composes 1.2x. Keep the two in sync. */
            contentZoom={(ar) => (ar === 1 ? 1.25 : ar < 1 ? 1.2 : 1)}
            animate={animate}
            restX={0}
            onAfterLoaded={() => { setAnimate(false); setImgLoading(false); }}
            onAfterError={() => {
              setImgLoading(false);
              setWarn(SESSION_EXPIRED);
            }}
            controlRef={sliderCtl}
            overlay={
              (dimMode || photo.dimensions.length > 0) && (
                <DimensionOverlay
                  key={entryKey(photo)}
                  adding={dimMode}
                  dimensions={photo.dimensions}
                  onChange={(ds) => setPhotos((ps) => ps.map((p, i) => (i === active ? { ...p, dimensions: ds } : p)))}
                  onExitAdding={() => setDimMode(false)}
                  onInteract={() => sliderCtl.current?.toLeft()}
                />
              )
            }
          />
          {/* Single loading signal: the spinner over the flat is-loading
              base (which also holds the finished frame's size). Quiet loads
              (ratio switches, prefetched) skip the veil so the frame morph
              runs unobscured. */}
          {imgLoading && !quietLoad && <div className="cj-stage-loading"><div className="cj-spin" /></div>}
        </div>

        {/* Always present — the "+" tile is the only way to add more photos. */}
        {(
          <div className="cj-film">
            {photos.map((p, i) => (
              <Fragment key={entryKey(p)}>
                <div className="cj-tw">
                  <ThumbButton
                    p={p}
                    active={i === active}
                    onClick={() => { if (i !== active) { setActive(i); setQuietLoad(false); setImgLoading(true); setDimMode(false); setAnimate(true); } }}
                  />
                  <button className="cj-delbtn" onClick={() => setConfirmDelete(i)} aria-label={`Delete ${p.name}`}>
                    ✕
                  </button>
                  {p.variant === "cover" && newCovers.has(p.id) && (
                    <span className="cj-newdot" title="New cover. Click to view" />
                  )}
                </div>
                {/* A cover being generated shows up immediately as a loading
                    tile where the finished cover will land (right after its
                    base photo), styled like an uploading photo. The tile's
                    image is the cover's "before" (the original upload). */}
                {!p.variant && coverBusy.has(p.id) &&
                  !photos.some((q) => q.id === p.id && q.variant === "cover") && (
                    <div
                      className="cj-pend spin"
                      style={{ backgroundImage: `url(/api/testing2/${p.id}/image?kind=before&ratio=1:1&w=160)` }}
                      title="Generating cover…"
                      aria-label="Generating cover"
                    />
                  )}
              </Fragment>
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
                title={`${f.name} failed. Click to retry${f.warn ? `\n${f.warn}` : ""}`}
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
          </div>
          <div className="cj-grp cj-actions">
            <button
              className={`cj-btn-outline${dimMode ? " active" : ""}`}
              onClick={() => {
                if (dimMode) setDimMode(false);
                else { setDimMode(true); sliderCtl.current?.toLeft(); }
              }}
            >
              {dimMode ? "Placing Points · Esc to Exit" : "Add Dimensions"}
            </button>
            <div className="cj-field">
              <button
                className="cj-btn-outline"
                disabled={busy}
                onClick={toggleCover}
                aria-busy={busy}
              >
                {busy ? "Generating Cover…" : coverOn ? "Remove Cover" : "Generate Cover"}
              </button>
              <p className="cj-hint" role="status">
                {busy
                  ? "Hang tight! Your cover will appear below in about 20 sec."
                  : coverOn
                    ? "Removing a cover cannot be undone."
                    : "Takes about 20 sec. Best for featured items."}
              </p>
            </div>
          </div>
        </div>
        <div className="cj-foot cj-lst-foot">
          {multi ? (
            <button
              className="cj-btn-primary"
              onClick={() => { setExportSel(new Set(photos.map((_, i) => i))); setExportOpen(true); }}
            >
              Export
            </button>
          ) : (
            <button className="cj-btn-primary" onClick={exportSingle}>Export Image</button>
          )}
          {savedNote && <p className="cj-saved">✓ Saved to Library</p>}
        </div>
      </aside>

      <ModalShell open={exportOpen} onClose={() => setExportOpen(false)}>
        <div className="cj-modal" onClick={(e) => e.stopPropagation()}>
            <div className="cj-modal-head">
              <h2>Export Selection</h2>
            </div>
            <div className="cj-modal-body">
              <div className="cj-modal-grid">
                {photos.map((p, i) => (
                  <button
                    key={entryKey(p)}
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
                    <img src={thumbUrl(p, 320)} alt={p.name} />
                    <span className="cj-ex-badge">
                      {exportSel.has(i) && <CheckIcon size={13} />}
                    </span>
                    {/* what this tile actually contains — batch exports pick the right version */}
                    <span className="cj-ex-flags">
                      {p.variant === "cover" && <span className="cj-ex-flag">Cover</span>}
                      {p.dimensions.length > 0 && (
                        <span className="cj-ex-flag">
                          {p.dimensions.length} dim{p.dimensions.length > 1 ? "s" : ""}
                        </span>
                      )}
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
      </ModalShell>

      {confirmDelete !== null && photos[confirmDelete] && (
        <div className="cj-modal-scrim" onClick={() => setConfirmDelete(null)}>
          <div className="cj-modal small" onClick={(e) => e.stopPropagation()}>
            <div className="cj-modal-head"><h2>Delete Photo</h2></div>
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
