"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/** Before/after compare slider — port of the Streamlit canvas iframe.
 *  The after image is in-flow and sizes the card; the before image + divider
 *  overlay it, clipped to the divider position. Drag anywhere to scrub;
 *  the corner tags animate fully to either side. */
export default function CompareSlider({
  beforeSrc,
  afterSrc,
  aspectRatio,
  animate,
  onAfterLoaded,
  onAfterError,
  overlay,
  controlRef,
  restX = 50,
  contentZoom,
}: {
  beforeSrc: string;
  afterSrc: string;
  /** Numeric width/height of the output ratio (1, 4/3, 4/5). Drives the box's
   *  aspect-ratio so the preview fits the stage without clipping (see .cj-split
   *  container-query sizing in globals.css). */
  aspectRatio: number;
  animate: boolean;
  onAfterLoaded?: () => void;
  /** Fired when the after image fails to load — e.g. the in-memory backend
   *  restarted and this photo's record is gone. */
  onAfterError?: () => void;
  overlay?: React.ReactNode;
  /** Imperative handle: lets the parent glide the divider (e.g. fully left
   *  while dimension annotations are being added/edited). */
  controlRef?: React.MutableRefObject<{ toLeft: () => void } | null>;
  /** Divider start position (%) when NOT running the reveal animation — e.g. 0
   *  to mount already showing the "after" image. Defaults to 50 (half/half). */
  restX?: number;
  /** Per-ratio product zoom applied by the backend composition (e.g. the CJ
   *  pipeline enlarges 4:5 renders 1.2x). When provided, ratio switches
   *  interpolate the product's on-screen scale smoothly instead of letting
   *  the size difference pop in with the cross-fade. Omit for pipelines
   *  without per-ratio zoom (legacy). */
  contentZoom?: (aspectRatio: number) => number;
}) {
  const boxRef = useRef<HTMLDivElement>(null);
  const raf = useRef<number | null>(null);
  const interacted = useRef(false);
  const dragging = useRef(false);
  // capture once: the reveal decision belongs to this mount only, so a parent
  // re-render flipping the prop mid-sweep can't cancel the animation
  const reveal = useRef(animate).current;
  // The reveal sweep must not start until BOTH images have decoded — kicked
  // off on mount it runs invisibly behind the loading state, so the sweep is
  // already over by the time pixels appear. (Error counts as "ready" so a
  // missing before image can't block the after side forever.)
  const [beforeReady, setBeforeReady] = useState(false);
  const [afterReady, setAfterReady] = useState(false);
  const [x, setX] = useState(reveal ? 100 : restX);
  const xRef = useRef(x);
  xRef.current = x;

  /* ── Ratio switch: blend at matched size, settle WITH the morph ─────────
     Key physics: with object-fit COVER, any two renders shown in the SAME
     box display the product at the same size, apart from a fixed factor m
     from zoom/aspect (a narrower target width-fills: m = zNew/zOld; a wider
     target height-fills: m = arNew·zNew/arOld·zOld — for the CJ ratio set
     m stays within ~0.96-1.2, so the overlay's start scale never strays far
     from 1). Earlier versions compensated for m as a SEPARATE stage
     (scale old up, then morph), and whenever the morph pulled size back the
     other way the product visibly grew then shrank. The fix is one clock:

       load  — box held at the old ratio; incoming pair decodes hidden
       blend — 120ms fade to the new render inside the OLD box, pre-scaled
               by 1/m so its product matches the old one exactly. A start
               scale above 1 just crops; below 1 the full-bleed base remains
               beneath it, so no box background can ever show
       morph — box morphs to the new ratio while the overlay's transform
               settles 1/m → 1 on the SAME 300ms clock; cover-tracking and
               the settle compose into one monotonic size glide
       committed — base src swaps UNDER the overlay; the overlay lifts only
               after the base <img> confirms the new bitmap painted

     The base layer is never transformed at all. */
  type FadeStage = "load" | "blend" | "morph" | "committed";
  type Fade = {
    before: string; after: string;
    arOld: number; ar: number;
    mInv: number;                 // overlay's starting scale (1/m)
    stage: FadeStage;
    bOk: boolean; aOk: boolean;   // incoming pair decoded (error counts)
  };
  const [shown, setShown] = useState({ before: beforeSrc, after: afterSrc });
  const [fade, setFade] = useState<Fade | null>(null);
  // The box aspect-ratio the DOM actually shows; lags the prop until "morph".
  const [shownAr, setShownAr] = useState(aspectRatio);
  const [prevSrc, setPrevSrc] = useState({ beforeSrc, afterSrc, aspectRatio });
  if (prevSrc.beforeSrc !== beforeSrc || prevSrc.afterSrc !== afterSrc) {
    const z = contentZoom ?? (() => 1);
    const m = aspectRatio <= prevSrc.aspectRatio
      ? z(aspectRatio) / z(prevSrc.aspectRatio)
      : (aspectRatio * z(aspectRatio)) / (prevSrc.aspectRatio * z(prevSrc.aspectRatio));
    setFade({
      before: beforeSrc, after: afterSrc,
      arOld: prevSrc.aspectRatio, ar: aspectRatio,
      mInv: 1 / m, stage: "load", bOk: false, aOk: false,
    });
    setPrevSrc({ beforeSrc, afterSrc, aspectRatio });
  }
  // Mirror for DOM event handlers on the base layer (main img onLoad).
  const fadeRef = useRef(fade);
  useEffect(() => { fadeRef.current = fade; });

  /* load → blend: both incoming images decoded → two frames so everything
     has painted in its start state (a cached image "loads" before first
     paint; launching immediately collapses a transition into an instant
     swap) → launch the fade at matched size. */
  useEffect(() => {
    if (!fade || fade.stage !== "load" || !fade.bOk || !fade.aOk) return;
    let r2 = 0;
    const r1 = requestAnimationFrame(() => {
      r2 = requestAnimationFrame(() => {
        setFade((f) => (f && f.stage === "load" ? { ...f, stage: "blend" } : f));
      });
    });
    return () => { cancelAnimationFrame(r1); cancelAnimationFrame(r2); };
  }, [fade]);

  /* Stage timers, slightly past each CSS duration (timers rather than
     transitionend so prefers-reduced-motion, where transitions are off,
     still sequences through instantly-applied states).
     blend 120ms → morph 300ms (box + overlay settle, one clock) → committed. */
  useEffect(() => {
    if (!fade) return;
    if (fade.stage === "blend") {
      const t = setTimeout(() => {
        setShownAr(fade.ar);
        setFade((f) => (f && f.stage === "blend" ? { ...f, stage: "morph" } : f));
      }, 170);
      return () => clearTimeout(t);
    }
    if (fade.stage === "morph") {
      const t = setTimeout(() => {
        setShown({ before: fade.before, after: fade.after });
        setFade((f) => (f && f.stage === "morph" ? { ...f, stage: "committed" } : f));
      }, 310);
      return () => clearTimeout(t);
    }
  }, [fade]);

  /* committed → done: lift the overlay only after the base <img> has PAINTED
     the new bitmap (onLoad + two frames; see the base img handler). The
     timeout is a safety net in case that onLoad never fires. */
  const finishFade = useCallback(() => {
    requestAnimationFrame(() => requestAnimationFrame(() => {
      setFade((f) => (f && f.stage === "committed" ? null : f));
    }));
  }, []);
  useEffect(() => {
    if (!fade || fade.stage !== "committed") return;
    const t = setTimeout(finishFade, 400);
    return () => clearTimeout(t);
  }, [fade, finishFade]);

  /* Error escape hatch: a failed incoming after-image jumps straight to the
     final state and reports up (session-expired handling lives there). */
  const abortFade = useCallback((f: Fade) => {
    setShown({ before: f.before, after: f.after });
    setShownAr(f.ar);
    setFade(null);
  }, []);

  /* Overlay classes: .on fades it in (blend); .settle sends its transform
     from scale(mInv) to scale(1) on the morph clock. Both persist through
     "committed" so the overlay keeps covering the base swap. */
  const xfadeClass = fade
    ? `cj-xfade${fade.stage !== "load" ? " on" : ""}${fade.stage === "morph" || fade.stage === "committed" ? " settle" : ""}`
    : "cj-xfade";

  const clamp = (v: number) => Math.max(0, Math.min(100, v));
  const ease = (t: number) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);

  const animateTo = useCallback((target: number) => {
    interacted.current = true;
    if (raf.current) cancelAnimationFrame(raf.current);
    const from = xRef.current;
    let t0: number | null = null;
    const dur = 450;
    const tick = (ts: number) => {
      if (t0 === null) t0 = ts;
      const p = Math.min(1, (ts - t0) / dur);
      setX(clamp(from + (target - from) * ease(p)));
      if (p < 1) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
  }, []);

  // parent control: glide the divider fully left (eased, no jump)
  useEffect(() => {
    if (!controlRef) return;
    controlRef.current = { toLeft: () => animateTo(0) };
    return () => { controlRef.current = null; };
  }, [controlRef, animateTo]);

  // initial reveal: sweep 100 → 0, once per mount, only after both images
  // decoded (loading animation first, THEN the before/after reveal).
  const swept = useRef(false);
  useEffect(() => {
    if (!reveal || swept.current || !beforeReady || !afterReady) return;
    swept.current = true;
    let t0: number | null = null;
    const dur = 1100;
    const step = (ts: number) => {
      if (interacted.current) return;
      if (t0 === null) t0 = ts;
      const p = Math.min(1, (ts - t0) / dur);
      setX(100 * (1 - ease(p)));
      if (p < 1) raf.current = requestAnimationFrame(step);
    };
    raf.current = requestAnimationFrame(step);
    return () => {
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, [reveal, beforeReady, afterReady]);

  const fromEvent = (clientX: number) => {
    const r = boxRef.current?.getBoundingClientRect();
    if (!r) return;
    setX(clamp(((clientX - r.left) / r.width) * 100));
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if ((e.target as HTMLElement).closest(".cj-tag")) return;
    dragging.current = true;
    interacted.current = true;
    if (raf.current) cancelAnimationFrame(raf.current);
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    fromEvent(e.clientX);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (dragging.current) fromEvent(e.clientX);
  };
  const onPointerUp = () => {
    dragging.current = false;
  };

  return (
    <div
      ref={boxRef}
      className="cj-split"
      style={{ "--cj-ar": shownAr } as React.CSSProperties}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        className="cj-after-img"
        src={shown.after}
        alt="Enhanced"
        decoding="sync"
        onLoad={() => {
          setAfterReady(true);
          // Committed src swap painted → safe to lift the overlay.
          if (fadeRef.current?.stage === "committed") finishFade();
          onAfterLoaded?.();
        }}
        onError={onAfterError}
      />
      <div
        className="cj-before-wrap"
        style={{ clipPath: `polygon(0 0, ${x}% 0, ${x}% 100%, 0 100%)` }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={shown.before}
          alt="Original"
          decoding="sync"
          onLoad={() => setBeforeReady(true)}
          onError={() => setBeforeReady(true)}
        />
      </div>
      {/* Incoming pair: invisible while it decodes (load), fades in at
          "blend" pre-scaled to match the old render's geometry, settles to
          scale(1) with the morph, and stays up as cover through
          "committed". */}
      {fade && (
        <div className={xfadeClass} style={{ "--xf-mi": fade.mInv } as React.CSSProperties}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            className="cj-xfade-img"
            src={fade.after}
            alt=""
            decoding="sync"
            onLoad={() => {
              setFade((f) => (f && f.stage === "load" ? { ...f, aOk: true } : f));
              onAfterLoaded?.();
            }}
            onError={() => { abortFade(fade); onAfterError?.(); }}
          />
          <div
            className="cj-xfade-clip"
            style={{ clipPath: `polygon(0 0, ${x}% 0, ${x}% 100%, 0 100%)` }}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              className="cj-xfade-img"
              src={fade.before}
              alt=""
              decoding="sync"
              onLoad={() => setFade((f) => (f && f.stage === "load" ? { ...f, bOk: true } : f))}
              onError={() => setFade((f) => (f && f.stage === "load" ? { ...f, bOk: true } : f))}
            />
          </div>
        </div>
      )}
      <div className="cj-slider" style={{ left: `${x}%` }} />
      <button className="cj-tag cj-tag-l" onClick={() => animateTo(100)}>
        Original
      </button>
      <button className="cj-tag cj-tag-r" onClick={() => animateTo(0)}>
        Enhanced
      </button>
      {/* Overlays (dimension annotations) belong to the AFTER image only, so
          they are clipped to the divider's right side — the before layer must
          never show them. Clipping also removes pointer hits on the hidden
          part, which is correct: you can't edit what you can't see. */}
      {overlay && (
        <div className="cj-overlay-clip" style={{ clipPath: `inset(0 0 0 ${x}%)` }}>
          {overlay}
        </div>
      )}
    </div>
  );
}
