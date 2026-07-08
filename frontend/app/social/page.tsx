"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

/* Social Media Generator — 4-step wizard (PRD V1).
   Step 1  choose ratio + upload (auto-advances)
   Step 2  template gallery (previews use the uploaded photo)
   Step 3  edit title/subtitle with live preview (debounced)
   Step 4  review + download PNG                                            */

type RatioId = "1:1" | "4:5" | "9:16";
const RATIO_CARDS: { id: RatioId; name: string; w: number; h: number }[] = [
  { id: "1:1", name: "Facebook Post", w: 400, h: 400 },
  { id: "4:5", name: "Instagram Post", w: 360, h: 444 },
  { id: "9:16", name: "Instagram Story", w: 270, h: 480 },
];
const RATIO_NAME: Record<RatioId, string> = {
  "1:1": "Facebook Post", "4:5": "Instagram Post", "9:16": "Instagram Story",
};

type Slot = { key: string; label: string; placeholder: string };
type Template = { id: string; name: string; slots: Slot[] };

// Sample copy shown in previews before the user edits text.
const SAMPLES: Record<string, string> = {
  title: "Your Title Here",
  subtitle: "A short description of your post goes here",
};

const renderUrl = (
  img: string, template: string, ratio: RatioId,
  texts: Record<string, string>, slots: Slot[], w: number, samples: boolean,
) => {
  const params = new URLSearchParams({ img, template, ratio, w: String(w) });
  for (const s of slots) {
    const v = (texts[s.key] ?? "").trim();
    params.set(s.key, v || (samples ? SAMPLES[s.key] ?? "" : ""));
  }
  return `/api/social/render?${params.toString()}`;
};

/* Arc progress indicator (from the approved carousel reference). */
function Arc({ n, active }: { n: number; active: number }) {
  return (
    <svg id="cj-arc" width="100" height="400" viewBox="0 0 100 400">
      <path d="M 0,0 C 80,100 80,300 0,400" />
      {Array.from({ length: n }, (_, i) => (
        <circle
          key={i}
          cx={i === active ? 60 : 28}
          cy={(400 * (i + 1)) / (n + 1)}
          r={i === active ? 6 : 4}
          fill={i === active ? "#1b1b1b" : "#d1d5db"}
        />
      ))}
    </svg>
  );
}

/* Scroll-snap carousel: tracks which card is nearest the viewport centre. */
function useCarousel(count: number) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [centered, setCentered] = useState(0);
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const upd = () => {
      const mid = el.getBoundingClientRect().top + el.clientHeight / 2;
      let best = 0, bd = Infinity;
      el.querySelectorAll<HTMLElement>(".cj-soc-card").forEach((c, i) => {
        const r = c.getBoundingClientRect();
        const d = Math.abs(r.top + r.height / 2 - mid);
        if (d < bd) { bd = d; best = i; }
      });
      setCentered(best);
    };
    el.addEventListener("scroll", upd, { passive: true });
    upd();
    return () => el.removeEventListener("scroll", upd);
  }, [count]);
  return { scrollRef, centered };
}

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

export default function SocialPage() {
  const [step, setStep] = useState(1);
  const [ratio, setRatio] = useState<RatioId | null>(null);
  const [imgId, setImgId] = useState<string | null>(null);
  const [templates, setTemplates] = useState<Template[]>([]);
  const [template, setTemplate] = useState<string | null>(null);
  const [texts, setTexts] = useState<Record<string, string>>({});
  const [uploading, setUploading] = useState<RatioId | null>(null);
  const [downloading, setDownloading] = useState(false);
  const inputRefs = useRef<Record<string, HTMLInputElement | null>>({});

  const tpl = templates.find((t) => t.id === template) ?? null;
  const debouncedTexts = useDebounced(texts, 300);

  /* step 1: click a ratio card → file picker → upload → step 2 */
  const upload = useCallback(async (r: RatioId, f: File) => {
    setUploading(r);
    try {
      const fd = new FormData();
      fd.append("file", f);
      const res = await fetch("/api/social/upload", { method: "POST", body: fd });
      if (!res.ok) return;
      const j = await res.json();
      setRatio(r);
      setImgId(j.id);
      const tRes = await fetch(`/api/social/templates?ratio=${encodeURIComponent(r)}`);
      const tj = await tRes.json();
      setTemplates(tj.templates);
      setTemplate((cur) => (tj.templates.some((t: Template) => t.id === cur) ? cur : null));
      setStep(2);
    } finally {
      setUploading(null);
    }
  }, []);

  const download = useCallback(async () => {
    if (!imgId || !ratio || !tpl) return;
    setDownloading(true);
    try {
      const url = renderUrl(imgId, tpl.id, ratio, texts, tpl.slots, 1080, false);
      const res = await fetch(url);
      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `cj_social_${tpl.id}_${ratio.replace(":", "x")}.png`;
      a.click();
      URL.revokeObjectURL(a.href);
    } finally {
      setDownloading(false);
    }
  }, [imgId, ratio, tpl, texts]);

  const titles = ["Upload an image", "Select A Layout", "Edit Content", "Review Your Post"];

  const gallery = useCarousel(templates.length);
  const ratios = useCarousel(RATIO_CARDS.length);

  const galleryW: Record<RatioId, number> = { "1:1": 400, "4:5": 360, "9:16": 270 };
  const previewW: Record<RatioId, number> = { "1:1": 470, "4:5": 440, "9:16": 330 };
  const reviewW: Record<RatioId, number> = { "1:1": 480, "4:5": 440, "9:16": 320 };

  const selIdx = useMemo(
    () => Math.max(0, templates.findIndex((t) => t.id === template)),
    [templates, template],
  );

  return (
    <div className={`cj-soc${step >= 3 ? " nosnap" : ""}`} ref={step === 1 ? ratios.scrollRef : step === 2 ? gallery.scrollRef : undefined}>
      <div className="cj-soc-head">
        <button
          className={`cj-soc-back${step === 1 ? " hidden" : ""}`}
          onClick={() => setStep((s) => Math.max(1, s - 1))}
          aria-label="Back"
        >
          ←
        </button>
        <span className="cj-soc-title">{titles[step - 1]}</span>
        <span className="cj-soc-step">Step {step} / 4</span>
      </div>

      {/* ═══ step 1: ratio + upload ═══ */}
      {step === 1 && (
        <>
          <Arc n={RATIO_CARDS.length} active={ratios.centered} />
          <div className="cj-snap-pad" />
          {RATIO_CARDS.map((r, i) => (
            <div key={r.id} className={`cj-soc-card${i === ratios.centered ? " active" : ""}`}>
              <div
                className="cj-ratio-tile"
                style={{ width: r.w, height: r.h }}
                onClick={() => inputRefs.current[r.id]?.click()}
              >
                {uploading === r.id
                  ? <div className="cj-spin" style={{ position: "relative", zIndex: 1 }} />
                  : <span className="ms">add_a_photo</span>}
              </div>
              <input
                ref={(el) => { inputRefs.current[r.id] = el; }}
                type="file" hidden accept="image/jpeg,image/png,image/webp"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) upload(r.id, f);
                  e.target.value = "";
                }}
              />
              <div className="cj-card-lbl"><h2>{r.name}</h2><p>{r.id}</p></div>
            </div>
          ))}
          <div className="cj-snap-pad" />
          {imgId && ratio && (
            <div className="cj-soc-actions">
              <button className="cj-next-btn" onClick={() => setStep(2)}>Continue&nbsp;&nbsp;→</button>
            </div>
          )}
        </>
      )}

      {/* ═══ step 2: template gallery ═══ */}
      {step === 2 && imgId && ratio && (
        <>
          <Arc n={templates.length} active={template ? selIdx : gallery.centered} />
          <div className="cj-snap-pad" />
          {templates.map((t, i) => (
            <div key={t.id} className={`cj-soc-card${i === gallery.centered ? " active" : ""}`}>
              <div
                className={`cj-tpl-frame${template === t.id ? " sel" : ""}`}
                style={{ width: galleryW[ratio] }}
                onClick={() => setTemplate(t.id)}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={renderUrl(imgId, t.id, ratio, texts, t.slots, 540, true)} alt={t.name} />
              </div>
              <div className="cj-card-lbl">
                <h2>{t.name}</h2>
                <p className={template === t.id ? "cj-sel-note" : ""}>
                  {RATIO_NAME[ratio]} · {ratio}
                </p>
              </div>
            </div>
          ))}
          <div className="cj-snap-pad" />
          <div className="cj-soc-actions">
            <button className="cj-next-btn" disabled={!template} onClick={() => setStep(3)}>
              Next&nbsp;&nbsp;→
            </button>
          </div>
        </>
      )}

      {/* ═══ step 3: edit content (live preview) ═══ */}
      {step === 3 && imgId && ratio && tpl && (
        <>
          <div className="cj-edit-row">
            <div className="cj-edit-preview">
              <div className="cj-preview-card" style={{ width: previewW[ratio] }}>
                {/* sample copy stands in for empty fields (matches the Figma
                    screens); the final render uses only real text */}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={renderUrl(imgId, tpl.id, ratio, debouncedTexts, tpl.slots, 720, true)} alt="Preview" />
              </div>
            </div>
            <div className="cj-edit-form">
              {tpl.slots.length ? (
                tpl.slots.map((s) => (
                  <div key={s.key}>
                    <p className="cj-field-lbl">{s.label}</p>
                    <input
                      className="cj-underline-input"
                      placeholder={s.placeholder}
                      value={texts[s.key] ?? ""}
                      onChange={(e) => setTexts((t) => ({ ...t, [s.key]: e.target.value }))}
                    />
                  </div>
                ))
              ) : (
                <p className="cj-noslots">
                  This template has no editable text — it lets your photo do the talking.
                </p>
              )}
            </div>
          </div>
          <div className="cj-soc-actions">
            <button className="cj-next-btn" onClick={() => setStep(4)}>Next&nbsp;&nbsp;→</button>
          </div>
        </>
      )}

      {/* ═══ step 4: review + download ═══ */}
      {step === 4 && imgId && ratio && tpl && (
        <>
          <div className="cj-review-col">
            <div>
              <div className="cj-preview-card" style={{ width: reviewW[ratio] }}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={renderUrl(imgId, tpl.id, ratio, texts, tpl.slots, 1080, false)} alt="Final" />
              </div>
              <p className="cj-review-cap">{RATIO_NAME[ratio]} {ratio}</p>
            </div>
          </div>
          <div className="cj-soc-actions">
            <button className="cj-next-btn" disabled={downloading} onClick={download}>
              {downloading ? "Preparing…" : "Download  ↓"}
            </button>
          </div>
        </>
      )}
    </div>
  );
}
