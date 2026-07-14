"use client";

import { useEffect, useRef, useState } from "react";

/* Product dimension annotations (Figma 324:674) — multiple per image.
   Coordinates are normalized (0-1) relative to the displayed image, so the
   annotations survive canvas resizing / zoom. Rendered inside CompareSlider's
   `.cj-split`, which is sized exactly to the image.

   Interactions (always live, not only while adding):
   - hover/click a line       → red trash button appears (click = delete)
   - drag the line's middle   → move the whole dimension
   - hover an endpoint        → that endpoint's handle appears; drag to move it
   - click the line           → both handles + the value input appear
   - click the value text     → inline input to edit it                      */

export type Pt = { x: number; y: number };
export type Dimension = { start: Pt; end: Pt; value: string };

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
const STROKE = "#5d5e66";

type Drag = {
  i: number;
  mode: "start" | "end" | "mid";
  lastX: number;             // client px of the previous move event
  lastY: number;
  moved: boolean;            // true once the pointer travelled > 3px
};

export default function DimensionOverlay({
  adding,
  dimensions,
  onChange,
  onExitAdding,
  onInteract,
}: {
  adding: boolean;
  dimensions: Dimension[];
  onChange: (d: Dimension[]) => void;
  onExitAdding: () => void;
  /** Fired when the user starts editing a dimension (drag / select / retype) —
   *  the parent glides the compare divider out of the way. */
  onInteract?: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const hoverT = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });

  // draft while adding a new dimension
  const [dStart, setDStart] = useState<Pt | null>(null);
  const [dEnd, setDEnd] = useState<Pt | null>(null);
  const [dValue, setDValue] = useState("");
  const [cursor, setCursor] = useState<Pt | null>(null);

  // interactions on saved dimensions
  const [hover, setHover] = useState<number | null>(null);
  const [hoverEnd, setHoverEnd] = useState<{ i: number; w: "start" | "end" } | null>(null);
  const [sel, setSel] = useState<number | null>(null);
  const [editIdx, setEditIdx] = useState<number | null>(null);
  const [editValue, setEditValue] = useState("");
  const [drag, setDrag] = useState<Drag | null>(null);

  // Track the layer's pixel size so normalized points map to px (resize-safe).
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const upd = () => setSize({ w: el.clientWidth, h: el.clientHeight });
    upd();
    const ro = new ResizeObserver(upd);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Fresh draft each time the Add Dimensions flow starts.
  useEffect(() => {
    if (!adding) return;
    setDStart(null);
    setDEnd(null);
    setDValue("");
    setCursor(null);
    setSel(null);
    setEditIdx(null);
  }, [adding]);

  const draftInput = adding && !!dStart && !!dEnd;
  useEffect(() => {
    if (draftInput || editIdx !== null)
      requestAnimationFrame(() => inputRef.current?.select());
  }, [draftInput, editIdx]);

  const toNorm = (cx: number, cy: number): Pt => {
    const r = ref.current!.getBoundingClientRect();
    return { x: clamp01((cx - r.left) / r.width), y: clamp01((cy - r.top) / r.height) };
  };
  const px = (p: Pt) => ({ x: p.x * size.w, y: p.y * size.h });

  /* geometry of one dimension in px: extended line, end caps, label + trash */
  const geom = (A: Pt, B: Pt) => {
    const a = px(A), b = px(B);
    const dx = b.x - a.x, dy = b.y - a.y;
    const len = Math.hypot(dx, dy) || 1;
    const ux = dx / len, uy = dy / len;
    const EXT = 8, CAP = 5, GAP = 22;
    const a2 = { x: a.x - ux * EXT, y: a.y - uy * EXT };
    const b2 = { x: b.x + ux * EXT, y: b.y + uy * EXT };
    const cvx = -uy * CAP, cvy = ux * CAP;
    let nx = -uy, ny = ux;
    if (ny > 0) { nx = -nx; ny = -ny; }          // perpendicular pointing "up"
    const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    return {
      line: [a2.x, a2.y, b2.x, b2.y] as const,
      capA: [a2.x - cvx, a2.y - cvy, a2.x + cvx, a2.y + cvy] as const,
      capB: [b2.x - cvx, b2.y - cvy, b2.x + cvx, b2.y + cvy] as const,
      label: { x: mid.x + nx * GAP, y: mid.y + ny * GAP },
      trash: { x: mid.x - nx * 26, y: mid.y - ny * 26 },
    };
  };

  /* ── hover bookkeeping (shared timeout so moving between the line, its
        handles and the trash button doesn't flicker) ── */
  const enterHover = (i: number) => {
    if (hoverT.current) clearTimeout(hoverT.current);
    setHover(i);
  };
  const leaveHover = () => {
    if (hoverT.current) clearTimeout(hoverT.current);
    hoverT.current = setTimeout(() => setHover(null), 160);
  };

  /* ── dragging (endpoint or whole line) ── */
  const startDrag = (i: number, mode: Drag["mode"]) => (e: React.PointerEvent) => {
    e.stopPropagation();
    onInteract?.();
    try {
      (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
    } catch {
      /* unknown pointerId (synthetic events) — drag still works via bubbling */
    }
    setDrag({ i, mode, lastX: e.clientX, lastY: e.clientY, moved: false });
  };
  const dragMove = (e: React.PointerEvent) => {
    if (!drag) return;
    const moved = drag.moved || Math.hypot(e.clientX - drag.lastX, e.clientY - drag.lastY) > 3;
    if (!moved) return;
    const r = ref.current!.getBoundingClientRect();
    const d = dimensions[drag.i];
    if (!d) return;
    let next: Dimension;
    if (drag.mode === "mid") {
      const ddx = (e.clientX - drag.lastX) / r.width;
      const ddy = (e.clientY - drag.lastY) / r.height;
      next = {
        ...d,
        start: { x: clamp01(d.start.x + ddx), y: clamp01(d.start.y + ddy) },
        end: { x: clamp01(d.end.x + ddx), y: clamp01(d.end.y + ddy) },
      };
    } else {
      const p = toNorm(e.clientX, e.clientY);
      next = drag.mode === "start" ? { ...d, start: p } : { ...d, end: p };
    }
    onChange(dimensions.map((x, j) => (j === drag.i ? next : x)));
    setDrag({ ...drag, lastX: e.clientX, lastY: e.clientY, moved: true });
  };
  const dragUp = (e: React.PointerEvent) => {
    if (!drag) return;
    e.stopPropagation();
    if (!drag.moved && drag.mode === "mid") {
      // a plain click on the line: select it — both handles + value input
      setSel(drag.i);
      setEditIdx(drag.i);
      setEditValue(dimensions[drag.i]?.value ?? "");
    }
    setDrag(null);
  };

  const removeDim = (i: number) => {
    onChange(dimensions.filter((_, j) => j !== i));
    setHover(null);
    setHoverEnd(null);
    setSel(null);
    setEditIdx(null);
  };

  const commitEdit = (i: number) => {
    const v = editValue.trim();
    if (v) onChange(dimensions.map((d, j) => (j === i ? { ...d, value: v } : d)));
    setEditIdx(null);
    setSel(null);
  };

  /* ── adding-mode handlers on the capture layer ── */
  const layerDown = (e: React.PointerEvent) => {
    if (!adding || draftInput) return;
    e.stopPropagation();
    if (!dStart) setDStart(toNorm(e.clientX, e.clientY));
    else if (!dEnd) setDEnd(toNorm(e.clientX, e.clientY));
  };
  const layerMove = (e: React.PointerEvent) => {
    if (adding && dStart && !dEnd) setCursor(toNorm(e.clientX, e.clientY));
  };
  const commitDraft = () => {
    const v = dValue.trim();
    if (!v || !dStart || !dEnd) return;
    onChange([...dimensions, { start: dStart, end: dEnd, value: v }]);
    onExitAdding();
  };

  const phase = !dStart ? "start" : !dEnd ? "end" : "value";
  const interactive = !adding;    // saved dims are inert while placing points

  /* one dimension (or the draft) rendered as svg parts + html bits */
  const renderDim = (d: Dimension | { start: Pt; end: Pt; value: string }, i: number | null) => {
    if (!size.w) return null;
    const g = geom(d.start, d.end);
    const isDraft = i === null;
    const showTrash = !isDraft && interactive && (hover === i || sel === i);
    const showStartHandle = isDraft || (interactive &&
      (sel === i || (hoverEnd && hoverEnd.i === i && hoverEnd.w === "start") ||
       (drag && drag.i === i && drag.mode === "start")));
    const showEndHandle = isDraft || (interactive &&
      (sel === i || (hoverEnd && hoverEnd.i === i && hoverEnd.w === "end") ||
       (drag && drag.i === i && drag.mode === "end")));
    const showInput = isDraft ? draftInput : editIdx === i;
    const showLabel = !showInput && !!d.value;
    const a = px(d.start), b = px(d.end);

    return (
      <div key={isDraft ? "draft" : i}>
        <svg className="cj-dim-svg" width={size.w} height={size.h}>
          <line x1={g.line[0]} y1={g.line[1]} x2={g.line[2]} y2={g.line[3]}
            stroke={STROKE} strokeWidth={1.5} />
          <line x1={g.capA[0]} y1={g.capA[1]} x2={g.capA[2]} y2={g.capA[3]}
            stroke={STROKE} strokeWidth={1.5} />
          <line x1={g.capB[0]} y1={g.capB[1]} x2={g.capB[2]} y2={g.capB[3]}
            stroke={STROKE} strokeWidth={1.5} />
          {/* fat invisible hit line: hover → trash, drag middle → move, click → select */}
          {!isDraft && interactive && (
            <line
              x1={g.line[0]} y1={g.line[1]} x2={g.line[2]} y2={g.line[3]}
              stroke="rgba(0,0,0,0)" strokeWidth={18}
              style={{ pointerEvents: "stroke", cursor: "move" }}
              onPointerEnter={() => enterHover(i!)}
              onPointerLeave={leaveHover}
              onPointerDown={startDrag(i!, "mid")}
              onPointerMove={dragMove}
              onPointerUp={dragUp}
            />
          )}
        </svg>

        {/* endpoint hit areas + emphasized handles */}
        {(["start", "end"] as const).map((w) => {
          const p = w === "start" ? a : b;
          const shown = w === "start" ? showStartHandle : showEndHandle;
          return (
            <div key={w}>
              {!isDraft && interactive && (
                <div
                  className="cj-dim-pt"
                  style={{ left: p.x, top: p.y }}
                  onPointerEnter={() => { enterHover(i!); setHoverEnd({ i: i!, w }); }}
                  onPointerLeave={() => { leaveHover(); setHoverEnd(null); }}
                  onPointerDown={startDrag(i!, w)}
                  onPointerMove={dragMove}
                  onPointerUp={dragUp}
                />
              )}
              {shown && <div className="cj-dim-handle" style={{ left: p.x, top: p.y }} />}
            </div>
          );
        })}

        {/* value label / inline input, centered above the line */}
        {showLabel && (
          <div
            className="cj-dim-label"
            style={{ left: g.label.x, top: g.label.y, pointerEvents: interactive ? "auto" : "none" }}
            onPointerDown={(e) => {
              if (!interactive || isDraft) return;
              e.stopPropagation();
              onInteract?.();
              setEditIdx(i!);
              setEditValue(d.value);
            }}
          >
            {d.value}
          </div>
        )}
        {showInput && (
          <input
            ref={inputRef}
            className="cj-dim-input"
            style={{ left: g.label.x, top: g.label.y }}
            placeholder="Enter the Dimension"
            value={isDraft ? dValue : editValue}
            onPointerDown={(e) => e.stopPropagation()}
            onChange={(e) => (isDraft ? setDValue(e.target.value) : setEditValue(e.target.value))}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                if (isDraft) commitDraft();
                else commitEdit(i!);
              }
              if (e.key === "Escape" && !isDraft) { setEditIdx(null); setSel(null); }
            }}
            onBlur={() => { if (!isDraft) commitEdit(i!); }}
          />
        )}
      </div>
    );
  };

  return (
    <div
      ref={ref}
      className="cj-dim-layer"
      style={{
        pointerEvents: adding ? "auto" : "none",
        cursor: adding && phase !== "value" ? "crosshair" : "default",
      }}
      onPointerDown={layerDown}
      onPointerMove={layerMove}
    >
      {/* saved dimensions */}
      {dimensions.map((d, i) => renderDim(d, i))}

      {/* draft: dashed preview → solid line + input */}
      {adding && dStart && !dEnd && (
        <svg className="cj-dim-svg" width={size.w} height={size.h}>
          {cursor && (
            <line
              x1={px(dStart).x} y1={px(dStart).y}
              x2={px(cursor).x} y2={px(cursor).y}
              stroke={STROKE} strokeWidth={1.5} strokeDasharray="5 4"
            />
          )}
        </svg>
      )}
      {adding && dStart && !dEnd && (
        <div className="cj-dim-handle" style={{ left: px(dStart).x, top: px(dStart).y }} />
      )}
      {adding && dStart && dEnd &&
        renderDim({ start: dStart, end: dEnd, value: dValue }, null)}

      {/* trash buttons (rendered after everything so they sit on top) */}
      {interactive && dimensions.map((d, i) => {
        if (!size.w || !(hover === i || sel === i)) return null;
        const g = geom(d.start, d.end);
        return (
          <button
            key={`trash-${i}`}
            className="cj-dim-trash"
            style={{ left: g.trash.x, top: g.trash.y }}
            aria-label="Delete dimension"
            onPointerEnter={() => enterHover(i)}
            onPointerLeave={leaveHover}
            onPointerDown={(e) => e.stopPropagation()}
            onClick={(e) => { e.stopPropagation(); removeDim(i); }}
          >
            <span className="ms">delete</span>
          </button>
        );
      })}

      {/* floating instruction panel — only while picking the two points
          (Figma 353:113: text · Revert Selection · X in one row) */}
      {adding && phase !== "value" && (
        <div className="cj-dim-panel" onPointerDown={(e) => e.stopPropagation()}>
          <div className="cj-dim-instr">
            {phase === "start" ? "Select a Starting Point" : "Select an End Point"}
          </div>
          {phase !== "start" && (
            <button
              className="cj-dim-revert"
              onClick={() => { setDStart(null); setDEnd(null); setDValue(""); setCursor(null); }}
            >
              Revert Selection
            </button>
          )}
          <button className="cj-dim-x" onClick={onExitAdding} aria-label="Exit dimension mode">
            <span className="ms">close</span>
          </button>
        </div>
      )}
    </div>
  );
}
