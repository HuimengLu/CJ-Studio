"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/* Social Media Generator — simplified 3-step flow.
   Step 1  upload one image
   Step 2  optional title / description
   Step 3  live previews: pick a platform (ratio) + template, download PNG.
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

type Slot = { key: string; label: string; placeholder: string };
type Template = { id: string; name: string; slots: Slot[] };

/* Title falls back to the brand placeholder server-side; an empty
   description simply doesn't appear — so texts pass through as-is. */
const renderUrl = (
  img: string, template: string, ratio: RatioId,
  texts: Record<string, string>, w: number,
) => {
  const params = new URLSearchParams({ img, template, ratio, w: String(w) });
  for (const k of ["title", "subtitle"]) {
    const v = (texts[k] ?? "").trim();
    if (v) params.set(k, v);
  }
  return `/api/social/render?${params.toString()}`;
};

export default function SocialPage() {
  const [step, setStep] = useState(1);
  const [imgId, setImgId] = useState<string | null>(null);
  const [texts, setTexts] = useState<Record<string, string>>({});
  const [ratio, setRatio] = useState<RatioId>("1:1");
  const [templates, setTemplates] = useState<Template[]>([]);
  const [template, setTemplate] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragover, setDragover] = useState(false);
  const [stageLoading, setStageLoading] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const tplCache = useRef<Partial<Record<RatioId, Template[]>>>({});

  /* step 1: upload → step 2 */
  const upload = useCallback(async (f: File) => {
    setUploading(true);
    try {
      const fd = new FormData();
      fd.append("file", f);
      const res = await fetch("/api/social/upload", { method: "POST", body: fd });
      if (!res.ok) return;
      const j = await res.json();
      setImgId(j.id);
      setStep(2);
    } finally {
      setUploading(false);
    }
  }, []);

  /* step 3: templates follow the chosen platform ratio */
  useEffect(() => {
    if (step !== 3) return;
    let alive = true;
    (async () => {
      let list = tplCache.current[ratio];
      if (!list) {
        const res = await fetch(`/api/social/templates?ratio=${encodeURIComponent(ratio)}`);
        const j = await res.json();
        list = j.templates as Template[];
        tplCache.current[ratio] = list;
      }
      if (!alive || !list) return;
      setTemplates(list);
      setTemplate((cur) => (list.some((t) => t.id === cur) ? cur : list[0]?.id ?? null));
    })();
    return () => { alive = false; };
  }, [step, ratio]);

  const download = useCallback(async () => {
    if (!imgId || !template) return;
    setDownloading(true);
    try {
      const res = await fetch(renderUrl(imgId, template, ratio, texts, 1080));
      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `cj_social_${template}_${ratio.replace(":", "x")}.png`;
      a.click();
      URL.revokeObjectURL(a.href);
    } finally {
      setDownloading(false);
    }
  }, [imgId, template, ratio, texts]);

  const titles = ["Upload an image", "Add your copy", "Preview & download"];
  const tpl = templates.find((t) => t.id === template) ?? null;

  return (
    <div className="cj-soc nosnap">
      <div className={`cj-soc-head${step === 3 ? " plain" : ""}`}>
        <button
          className={`cj-soc-back${step === 1 ? " hidden" : ""}`}
          onClick={() => setStep((s) => Math.max(1, s - 1))}
          aria-label="Back"
        >
          ←
        </button>
        <span className="cj-soc-title">{titles[step - 1]}</span>
        <span className="cj-soc-step">Step {step} / 3</span>
      </div>

      {/* ═══ step 1: upload ═══ */}
      {step === 1 && (
        <div className="cj-soc-center">
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
        </div>
      )}

      {/* ═══ step 2: optional copy ═══ */}
      {step === 2 && imgId && (
        <>
          <div className="cj-soc-form">
            <p className="cj-field-lbl">Title</p>
            <input
              className="cj-underline-input"
              placeholder="Construction Junction"
              value={texts.title ?? ""}
              onChange={(e) => setTexts((t) => ({ ...t, title: e.target.value }))}
            />
            <p className="cj-field-lbl">Description · Optional</p>
            <input
              className="cj-underline-input"
              value={texts.subtitle ?? ""}
              onChange={(e) => setTexts((t) => ({ ...t, subtitle: e.target.value }))}
            />
            <p className="cj-form-note">
              Both fields are optional — leave them blank and the templates
              speak for themselves.
            </p>
          </div>
          <div className="cj-soc-actions">
            <button className="cj-next-btn" onClick={() => setStep(3)}>Next&nbsp;&nbsp;→</button>
          </div>
        </>
      )}

      {/* ═══ step 3: previews + platform + download ═══ */}
      {step === 3 && imgId && (
        <div className="cj-result">
          <div className="cj-canvas-col">
            <div className="cj-stage cj-soc-stage">
              {tpl && (
                <div className="cj-stagecard">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    key={`${tpl.id}-${ratio}`}
                    src={renderUrl(imgId, tpl.id, ratio, texts, 720)}
                    alt={tpl.name}
                    onLoad={() => setStageLoading(false)}
                    onError={() => setStageLoading(false)}
                  />
                </div>
              )}
              {stageLoading && (
                <div className="cj-stage-loading"><div className="cj-spin" /></div>
              )}
            </div>
            {/* template filmstrip — like the product-photo photo strip */}
            <div className="cj-film">
              {templates.map((t) => (
                <button
                  key={t.id}
                  title={t.name}
                  className={`cj-tpl-thumb${t.id === template ? " active" : ""}`}
                  style={{ width: Math.round(96 * (270 / RATIO_H[ratio])) }}
                  onClick={() => { if (t.id !== template) { setStageLoading(true); setTemplate(t.id); } }}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={renderUrl(imgId, t.id, ratio, texts, 176)} alt={t.name} />
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
                <p className="cj-sec">Template</p>
                <p className="cj-soc-tplname">
                  {tpl ? tpl.name : "—"}
                  <span> · {RATIO_NAME[ratio]}</span>
                </p>
              </div>
            </div>
            <div className="cj-foot">
              <button className="cj-btn-primary" disabled={downloading || !tpl} onClick={download}>
                {downloading ? "Preparing…" : "Download"}
              </button>
            </div>
          </aside>
        </div>
      )}
    </div>
  );
}
