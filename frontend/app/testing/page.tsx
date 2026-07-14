"use client";

import { useCallback, useState } from "react";

/* Testing — a Product-Photo-style editor over a FIXED sample image.
   Left panel: title + subtitle inputs, photo-ratio selector, colour selector,
   download. Right: live preview of the chosen template style + a style nav
   strip. Editing title/subtitle re-renders on Enter; ratio/style/colour
   re-render at once. Styles 1-4 reproduce the Figma "Style" sets, each in
   Deep Green / Light Green / White. */

type RatioId = "1:1" | "4:5" | "9:16";
const RATIOS: { id: RatioId; label: string }[] = [
  { id: "1:1", label: "Square" },
  { id: "4:5", label: "Portrait" },
  { id: "9:16", label: "Story" },
];
const RATIO_H: Record<RatioId, number> = { "1:1": 270, "4:5": 337.5, "9:16": 480 };

type ThemeId = "green" | "lime" | "white";
const THEMES: { id: ThemeId; label: string; fill: string }[] = [
  { id: "green", label: "Deep Green", fill: "#005618" },
  { id: "lime", label: "Light Green", fill: "#BCF00E" },
  { id: "white", label: "White", fill: "#FFFFFF" },
];

type StyleId =
  | "style1" | "style2" | "style3" | "style4" | "style5"
  | "style6" | "style7" | "style8" | "style9"
  | "style10" | "style11" | "style12" | "style13";
const STYLES: { id: StyleId; label: string }[] = [
  { id: "style1", label: "Style 1" },
  { id: "style2", label: "Style 2" },
  { id: "style3", label: "Style 3" },
  { id: "style4", label: "Style 4" },
  { id: "style5", label: "Style 5" },
  { id: "style6", label: "Style 6" },
  { id: "style7", label: "Style 7" },
  { id: "style8", label: "Style 8" },
  { id: "style9", label: "Style 9" },
  { id: "style10", label: "Style 10" },
  { id: "style11", label: "Style 11" },
  { id: "style12", label: "Style 12" },
  { id: "style13", label: "Style 13" },
];

const renderUrl = (
  style: StyleId, theme: ThemeId, ratio: RatioId,
  title: string, subtitle: string, w: number,
) => {
  const p = new URLSearchParams({ style, theme, ratio, w: String(w) });
  if (title.trim()) p.set("title", title.trim());
  if (subtitle.trim()) p.set("subtitle", subtitle.trim());
  return `/api/testing/render?${p.toString()}`;
};

export default function TestingPage() {
  const [style, setStyle] = useState<StyleId>("style1");
  const [theme, setTheme] = useState<ThemeId>("green");
  const [ratio, setRatio] = useState<RatioId>("1:1");
  // Live input values vs. the applied text that actually drives the render —
  // the preview only refreshes when the user commits with Enter.
  const [title, setTitle] = useState("");
  const [subtitle, setSubtitle] = useState("");
  const [applied, setApplied] = useState({ title: "", subtitle: "" });
  const [stageLoading, setStageLoading] = useState(false);
  const [downloading, setDownloading] = useState(false);

  const commit = () => {
    if (applied.title !== title || applied.subtitle !== subtitle) {
      setStageLoading(true);
      setApplied({ title, subtitle });
    }
  };
  const dirty = applied.title !== title || applied.subtitle !== subtitle;

  const download = useCallback(async () => {
    setDownloading(true);
    try {
      const res = await fetch(renderUrl(style, theme, ratio, applied.title, applied.subtitle, 1080));
      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `cj_testing_${style}_${theme}_${ratio.replace(":", "x")}.png`;
      a.click();
      URL.revokeObjectURL(a.href);
    } finally {
      setDownloading(false);
    }
  }, [style, theme, ratio, applied]);

  return (
    <div className="cj-result">
      {/* ── left: content + ratio + colour controls ── */}
      <aside className="cj-panel left">
        <div className="cj-panel-body">
          <div className="cj-grp">
            <p className="cj-sec">Title</p>
            <input
              className="cj-desc-input"
              placeholder="Construction Junction"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") commit(); }}
            />
          </div>
          <div className="cj-grp">
            <p className="cj-sec">Subtitle (Optional)</p>
            <input
              className="cj-desc-input"
              placeholder="Enter your short description here"
              value={subtitle}
              onChange={(e) => setSubtitle(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") commit(); }}
            />
          </div>
          {dirty && (
            <button className="cj-apply-wide" onClick={commit}>
              Apply&nbsp;&nbsp;↵
            </button>
          )}
          <div className="cj-grp">
            <p className="cj-sec">Photo Ratio</p>
            <div className="cj-ratio-row">
              {RATIOS.map((r) => (
                <button
                  key={r.id}
                  className={`cj-ratio-card${ratio === r.id ? " active" : ""}`}
                  onClick={() => { if (ratio !== r.id) { setStageLoading(true); setRatio(r.id); } }}
                >
                  {r.id}
                  <small>{r.label}</small>
                </button>
              ))}
            </div>
          </div>
          <div className="cj-grp">
            <p className="cj-sec">Colors</p>
            <div className="cj-color-row">
              {THEMES.map((t) => (
                <button
                  key={t.id}
                  title={t.label}
                  className={`cj-color-card${theme === t.id ? " active" : ""}`}
                  onClick={() => { if (theme !== t.id) { setStageLoading(true); setTheme(t.id); } }}
                >
                  <span className="cj-color-dot" style={{ background: t.fill }} />
                  <small>{t.label}</small>
                </button>
              ))}
            </div>
          </div>
        </div>
        <div className="cj-foot">
          <button className="cj-btn-primary" disabled={downloading} onClick={download}>
            {downloading ? "Preparing…" : "Download"}
          </button>
        </div>
      </aside>

      {/* ── right: preview + style strip ── */}
      <div className="cj-canvas-col">
        <div className="cj-stage">
          <div className="cj-stagecard">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              key={`${style}-${theme}-${ratio}`}
              src={renderUrl(style, theme, ratio, applied.title, applied.subtitle, 900)}
              alt={`${style} ${theme} ${ratio}`}
              onLoad={() => setStageLoading(false)}
              onError={() => setStageLoading(false)}
            />
          </div>
          {stageLoading && (
            <div className="cj-stage-loading"><div className="cj-spin" /></div>
          )}
        </div>
        <div className="cj-film">
          {STYLES.map((s) => (
            <button
              key={s.id}
              title={s.label}
              className={`cj-tpl-thumb${s.id === style ? " active" : ""}`}
              style={{ width: Math.round(96 * (270 / RATIO_H[ratio])) }}
              onClick={() => { if (s.id !== style) { setStageLoading(true); setStyle(s.id); } }}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={renderUrl(s.id, theme, ratio, applied.title, applied.subtitle, 200)} alt={s.label} />
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
