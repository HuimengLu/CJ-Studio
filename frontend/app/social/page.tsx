"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/* Social Media Generator — simplified 2-step flow.
   Step 1  upload one image + optional title / subtitle
   Step 2  the engine recommends 2-3 templates (not the whole library) —
           pick a platform (ratio), browse the shortlist, regenerate, download.
   Mirrors the Product Photo result screen: canvas left, control panel right. */

type RatioId = "1:1" | "4:5" | "9:16";
const PLATFORMS: { id: RatioId; label: string; small: string }[] = [
  { id: "1:1", label: "1:1", small: "Facebook Post" },
  { id: "4:5", label: "4:5", small: "Instagram Post" },
  { id: "9:16", label: "9:16", small: "Instagram Story" },
];
const RATIO_NAME: Record<RatioId, string> = {
  "1:1": "Facebook Post", "4:5": "Instagram Post", "9:16": "Instagram Story",
};
const RATIO_H: Record<RatioId, number> = { "1:1": 270, "4:5": 333, "9:16": 480 };

/* A recommendation is one concrete (template × theme) option the engine
   surfaced for the user's content — the variation (line count) is resolved by
   auto-fit at render time. */
type Rec = {
  template: string;
  name: string;
  theme: string;
  variation: string;
  variations: string[];
};

/* Title falls back to the brand placeholder server-side; an empty
   description simply doesn't appear — so texts pass through as-is. */
const renderUrl = (
  img: string, template: string, ratio: RatioId,
  texts: Record<string, string>, w: number, theme?: string,
) => {
  const params = new URLSearchParams({ img, template, ratio, w: String(w) });
  for (const k of ["title", "subtitle"]) {
    const v = (texts[k] ?? "").trim();
    if (v) params.set(k, v);
  }
  if (theme) params.set("theme", theme);
  return `/api/social/render?${params.toString()}`;
};

export default function SocialPage() {
  const [step, setStep] = useState(1);
  const [imgId, setImgId] = useState<string | null>(null);
  const [texts, setTexts] = useState<Record<string, string>>({});
  const [ratio, setRatio] = useState<RatioId>("1:1");
  const [recs, setRecs] = useState<Rec[]>([]);
  const [selIdx, setSelIdx] = useState(0);
  const [rollKey, setRollKey] = useState(0);   // bump to re-roll a fresh shortlist
  const [loadingRecs, setLoadingRecs] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragover, setDragover] = useState(false);
  const [stageLoading, setStageLoading] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  /* step 1: upload (stays on step 1 — user adds copy, then hits Next) */
  const upload = useCallback(async (f: File) => {
    setUploading(true);
    try {
      const fd = new FormData();
      fd.append("file", f);
      const res = await fetch("/api/social/upload", { method: "POST", body: fd });
      if (!res.ok) return;
      const j = await res.json();
      setImgId(j.id);
      setPreview((old) => {
        if (old) URL.revokeObjectURL(old);
        return URL.createObjectURL(f);
      });
    } finally {
      setUploading(false);
    }
  }, []);

  /* step 2: ask the engine for a fresh shortlist whenever the platform (ratio),
     the copy, or the re-roll key changes. No seed → server re-randomises. */
  useEffect(() => {
    if (step !== 2) return;
    let alive = true;
    setLoadingRecs(true);
    (async () => {
      const p = new URLSearchParams({ ratio });
      const t = (texts.title ?? "").trim();
      const s = (texts.subtitle ?? "").trim();
      if (t) p.set("title", t);
      if (s) p.set("subtitle", s);
      const res = await fetch(`/api/social/recommend?${p.toString()}`);
      const j = await res.json();
      if (!alive) return;
      const list: Rec[] = j.recommendations ?? [];
      setRecs(list);
      setSelIdx(0);
      setStageLoading(list.length > 0);
      setLoadingRecs(false);
    })();
    return () => { alive = false; };
  }, [step, ratio, rollKey, texts]);

  const sel = recs[selIdx] ?? null;

  const download = useCallback(async () => {
    if (!imgId || !sel) return;
    setDownloading(true);
    try {
      const res = await fetch(renderUrl(imgId, sel.template, ratio, texts, 1080, sel.theme));
      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `cj_social_${sel.template}_${sel.theme}_${ratio.replace(":", "x")}.png`;
      a.click();
      URL.revokeObjectURL(a.href);
    } finally {
      setDownloading(false);
    }
  }, [imgId, sel, ratio, texts]);

  return (
    <div className="cj-soc nosnap">
      <div className={`cj-soc-head${step === 2 ? " plain" : ""}`}>
        <button
          className={`cj-soc-back${step === 1 ? " hidden" : ""}`}
          onClick={() => setStep((s) => Math.max(1, s - 1))}
          aria-label="Back"
        >
          ←
        </button>
      </div>

      {/* ═══ step 1: upload + optional copy ═══ */}
      {step === 1 && (
        <>
          <div className="cj-soc-center">
            <div className="cj-soc-setup">
              <div
                className={`cj-drop${dragover ? " dragover" : ""}`}
                onClick={() => fileRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setDragover(true); }}
                onDragLeave={() => setDragover(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragover(false);
                  const f = e.dataTransfer.files?.[0];
                  if (f) upload(f);
                }}
              >
                {uploading ? (
                  <div className="cj-spin" />
                ) : preview ? (
                  <>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img className="cj-drop-preview" src={preview} alt="Uploaded" />
                    <div className="cj-drop-hint">Click or drop to replace</div>
                  </>
                ) : (
                  <>
                    <div className="cj-drop-icon"><span className="ms">add_photo_alternate</span></div>
                    <div className="cj-drop-title">Click or drop a photo</div>
                    <div className="cj-drop-hint">JPG · PNG · WEBP</div>
                  </>
                )}
                <input
                  ref={fileRef} type="file" hidden
                  accept="image/jpeg,image/png,image/webp"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) upload(f);
                    e.target.value = "";
                  }}
                />
              </div>
              <div className="cj-grp">
                <p className="cj-sec">Title</p>
                <input
                  className="cj-desc-input"
                  placeholder="Construction Junction"
                  value={texts.title ?? ""}
                  onChange={(e) => setTexts((t) => ({ ...t, title: e.target.value }))}
                />
              </div>
              <div className="cj-grp">
                <p className="cj-sec">Subtitle (Optional)</p>
                <input
                  className="cj-desc-input"
                  placeholder="Enter your short description here"
                  value={texts.subtitle ?? ""}
                  onChange={(e) => setTexts((t) => ({ ...t, subtitle: e.target.value }))}
                />
              </div>
            </div>
          </div>
          <div className="cj-soc-actions">
            <button
              className="cj-next-btn"
              disabled={!imgId || uploading}
              onClick={() => setStep(2)}
            >
              Next&nbsp;&nbsp;→
            </button>
          </div>
        </>
      )}

      {/* ═══ step 2: recommendations + platform + download ═══ */}
      {step === 2 && imgId && (
        <div className="cj-result">
          <div className="cj-canvas-col">
            <div className="cj-stage cj-soc-stage">
              {sel && (
                <div className="cj-stagecard">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    key={`${sel.template}-${sel.theme}-${ratio}`}
                    src={renderUrl(imgId, sel.template, ratio, texts, 720, sel.theme)}
                    alt={sel.name}
                    onLoad={() => setStageLoading(false)}
                    onError={() => setStageLoading(false)}
                  />
                </div>
              )}
              {!sel && !loadingRecs && (
                <div className="cj-soc-empty">
                  No template fits this platform for your content.<br />
                  Try another platform.
                </div>
              )}
              {(stageLoading || loadingRecs) && (
                <div className="cj-stage-loading"><div className="cj-spin" /></div>
              )}
            </div>
            {/* recommendation shortlist — the engine picks 2-3, not the full library */}
            <div className="cj-film">
              {recs.map((r, i) => (
                <button
                  key={`${r.template}-${r.theme}-${i}`}
                  title={`${r.name} · ${r.theme}`}
                  className={`cj-tpl-thumb${i === selIdx ? " active" : ""}`}
                  style={{ width: Math.round(96 * (270 / RATIO_H[ratio])) }}
                  onClick={() => { if (i !== selIdx) { setStageLoading(true); setSelIdx(i); } }}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={renderUrl(imgId, r.template, ratio, texts, 176, r.theme)} alt={r.name} />
                </button>
              ))}
            </div>
          </div>

          <aside className="cj-panel">
            <div className="cj-panel-head"><h3>Preview</h3></div>
            <div className="cj-panel-body">
              <div className="cj-grp">
                <p className="cj-sec">Platform</p>
                <div className="cj-ratio-row">
                  {PLATFORMS.map((p) => (
                    <button
                      key={p.id}
                      className={`cj-ratio-card${ratio === p.id ? " active" : ""}`}
                      onClick={() => { if (ratio !== p.id) { setStageLoading(true); setRatio(p.id); } }}
                    >
                      {p.label}
                      <small>{p.small}</small>
                    </button>
                  ))}
                </div>
              </div>
              <div className="cj-grp">
                <div className="cj-toggle-row">
                  <p className="cj-sec">Recommendations</p>
                  <button
                    className="cj-reroll"
                    disabled={loadingRecs}
                    onClick={() => setRollKey((k) => k + 1)}
                    title="Show a fresh set"
                  >
                    ↻ Regenerate
                  </button>
                </div>
                <p className="cj-soc-tplname">
                  {sel ? sel.name : "—"}
                  <span>
                    {sel ? ` · ${sel.theme}` : ""} · {RATIO_NAME[ratio]}
                  </span>
                </p>
              </div>
            </div>
            <div className="cj-foot">
              <button className="cj-btn-primary" disabled={downloading || !sel} onClick={download}>
                {downloading ? "Preparing…" : "Download"}
              </button>
            </div>
          </aside>
        </div>
      )}
    </div>
  );
}
