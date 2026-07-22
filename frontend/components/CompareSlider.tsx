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
}) {
  const boxRef = useRef<HTMLDivElement>(null);
  const raf = useRef<number | null>(null);
  const interacted = useRef(false);
  const dragging = useRef(false);
  // capture once: the reveal decision belongs to this mount only, so a parent
  // re-render flipping the prop mid-sweep can't cancel the animation
  const reveal = useRef(animate).current;
  const [x, setX] = useState(reveal ? 100 : restX);
  const xRef = useRef(x);
  xRef.current = x;

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

  // initial reveal: sweep 100 → 0 once per mount
  useEffect(() => {
    if (!reveal) return;
    let t0: number | null = null;
    const dur = 2000;
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
      style={{ "--cj-ar": aspectRatio } as React.CSSProperties}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img className="cj-after-img" src={afterSrc} alt="Enhanced" onLoad={onAfterLoaded} onError={onAfterError} />
      <div
        className="cj-before-wrap"
        style={{ clipPath: `polygon(0 0, ${x}% 0, ${x}% 100%, 0 100%)` }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={beforeSrc} alt="Original" />
      </div>
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
