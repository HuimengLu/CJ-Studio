"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { CameraPlusIcon, PlusIcon } from "@/components/icons";
import OptionCard from "@/components/OptionCard";

/* Social — the official social-graphic editor.
   Left panel: single-image upload, title + subtitle inputs, photo-ratio
   selector, colour selector, download. Right: live preview of the chosen
   template + a filmstrip grouped by category (Cover / Text / Secondary /
   Image). Templates the current content can't use are dimmed in place; the
   selection never auto-switches — a banner explains any mismatch instead.
   Without an upload, a neutral gray placeholder serves as the base photo. */

type RatioId = "1:1" | "4:5" | "9:16";
/* Labels name the destination, not the geometry — staff pick by "where is
   this going", and the ratio digits above already say the shape. */
const RATIOS: { id: RatioId; label: string }[] = [
  { id: "1:1", label: "Facebook" },
  { id: "4:5", label: "Instagram" },
  { id: "9:16", label: "Story" },
];
const RATIO_H: Record<RatioId, number> = { "1:1": 270, "4:5": 337.5, "9:16": 480 };

/* Input caps ≈ two rendered lines; enforced at typing time so nothing is
   silently truncated at render time. */
const TITLE_MAX = 40;
const SUBTITLE_MAX = 120;

type ThemeId = "green" | "lime" | "white";
const THEMES: { id: ThemeId; label: string; fill: string }[] = [
  { id: "green", label: "Dark", fill: "#005618" },
  { id: "lime", label: "Light", fill: "#BCF00E" },
  { id: "white", label: "White", fill: "#FFFFFF" },
];

type StyleId =
  | "cover1" | "cover2" | "cover3" | "cover4" | "cover5" | "cover6" | "cover7"
  | "textonly"
  | "sec1a" | "sec1b" | "sec1c" | "sec1d" | "sec2" | "sec3" | "sec4a" | "sec4b"
  | "imageonly";

/* The Figma template library, grouped by category. Content-requirement flags
   mirror backend/social_engine.py TEMPLATES and drive the dynamic filmstrip:
   templates stay in a fixed order (no reflow while typing) and the ones the
   current content can't use are dimmed in place with a reason tooltip. */
type Template = {
  id: StyleId;
  label: string;
  requiresImage: boolean;
  supportsTitle: boolean;
  supportsSubtitle: boolean;
};
type TemplateGroup = { label: string; templates: Template[] };
const tpl = (
  id: StyleId, label: string,
  image = true, title = true, sub = true,
): Template => ({ id, label, requiresImage: image, supportsTitle: title, supportsSubtitle: sub });

const TEMPLATE_GROUPS: TemplateGroup[] = [
  {
    label: "Cover",
    templates: [
      tpl("cover1", "Cover 1"), tpl("cover2", "Cover 2"), tpl("cover3", "Cover 3"),
      tpl("cover4", "Cover 4"), tpl("cover5", "Cover 5"), tpl("cover6", "Cover 6"),
      tpl("cover7", "Cover 7"),
    ],
  },
  { label: "Text", templates: [tpl("textonly", "Text Only", false)] },
  {
    label: "Secondary",
    templates: [
      // Secondary 1/4 sub-styles are named by their visual signature so the
      // tooltip/aria-label says what actually differs between them.
      tpl("sec1a", "Secondary 1 · Brush"), tpl("sec1b", "Secondary 1 · Zigzag"),
      tpl("sec1c", "Secondary 1 · Claw"), tpl("sec1d", "Secondary 1 · Footer"),
      tpl("sec2", "Secondary 2 · Band Frame", true, true, false),
      tpl("sec3", "Secondary 3 · Sketch Frame", true, true, false),
      tpl("sec4a", "Secondary 4 · Brush & Zigzag", true, true, false),
      tpl("sec4b", "Secondary 4 · Claw & Arrow", true, true, false),
    ],
  },
  { label: "Image", templates: [tpl("imageonly", "Image Only", true, false, false)] },
];
const ALL_TEMPLATES: Template[] = TEMPLATE_GROUPS.flatMap((g) => g.templates);

/* Two distinct states drive the filmstrip:
   - CONFLICT (incompatReason ≠ null): the template would silently DROP some
     of the user's content — dimmed in place, with the reason as tooltip.
   - INCOMPLETE (needsPhoto): the template just lacks a photo. That's a step
     not a mismatch, so the thumb stays active (the gray placeholder already
     reads as "your photo goes here") and only gains a small camera badge. */
function incompatReason(
  t: Template, hasImage: boolean, hasTitle: boolean, hasSubtitle: boolean,
): string | null {
  if (!t.requiresImage && hasImage) return "Text-only template — your photo won't be used";
  if (!t.supportsTitle && (hasTitle || hasSubtitle)) return "Image-only template — your text won't appear";
  if (!t.supportsSubtitle && hasSubtitle) return "This template has no subtitle slot — it won't appear";
  return null;
}

function needsPhoto(
  t: Template, hasImage: boolean, hasTitle: boolean, hasSubtitle: boolean,
): boolean {
  return t.requiresImage && !hasImage && (hasTitle || hasSubtitle);
}

/* Templates that don't consume a field don't get it in the URL — the URL
   stays stable while the user types, so the (cacheable) responses are served
   straight from the browser cache instead of re-hitting the backend. */
const renderUrl = (
  style: StyleId, theme: ThemeId, ratio: RatioId,
  title: string, subtitle: string, w: number, imgId?: string,
) => {
  const t = ALL_TEMPLATES.find((x) => x.id === style);
  const p = new URLSearchParams({ style, theme, ratio, w: String(w) });
  if (t?.supportsTitle && title.trim()) p.set("title", title.trim());
  if (t?.supportsTitle && t?.supportsSubtitle && subtitle.trim()) p.set("subtitle", subtitle.trim());
  if (imgId && t?.requiresImage !== false) p.set("img", imgId);
  return `/api/testing/render?${p.toString()}`;
};

export default function SocialPage() {
  const [style, setStyle] = useState<StyleId>("cover1");
  const [theme, setTheme] = useState<ThemeId>("green");
  const [ratio, setRatio] = useState<RatioId>("1:1");
  // One base photo at a time; id comes from the backend, preview is local.
  const [photo, setPhoto] = useState<{ id: string; preview: string } | null>(null);
  const [uploadErr, setUploadErr] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  // Live input values vs. the applied text that actually drives the render —
  // the preview follows typing after a debounce (renders are memoized
  // server-side, so refreshes are cheap).
  const [title, setTitle] = useState("");
  const [subtitle, setSubtitle] = useState("");
  const [applied, setApplied] = useState({ title: "", subtitle: "" });
  const [stageLoading, setStageLoading] = useState(false);
  const [renderErr, setRenderErr] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [downloadDone, setDownloadDone] = useState(false);
  const [titleTruncated, setTitleTruncated] = useState(false);

  useEffect(() => {
    if (applied.title === title && applied.subtitle === subtitle) return;
    const t = setTimeout(() => {
      setStageLoading(true);
      setApplied({ title, subtitle });
    }, 500);
    return () => clearTimeout(t);
  }, [title, subtitle, applied]);

  /* The stage render was just produced (and cached) server-side, so this
     re-fetch of the same URL is a cache hit — it only reads the
     X-Title-Truncated header to warn when the title gets ellipsised. */
  useEffect(() => {
    if (!applied.title.trim()) {
      setTitleTruncated(false);
      return;
    }
    const ctrl = new AbortController();
    fetch(renderUrl(style, theme, ratio, applied.title, applied.subtitle, 900, photo?.id),
      { signal: ctrl.signal })
      .then((r) => setTitleTruncated(r.headers.get("X-Title-Truncated") === "1"))
      .catch(() => {});
    return () => ctrl.abort();
  }, [style, theme, ratio, applied, photo]);

  const uploadPhoto = useCallback(async (f: File) => {
    if (!f.type.startsWith("image/")) return;
    setUploading(true);
    setUploadErr(null);
    try {
      const fd = new FormData();
      fd.append("file", f);
      const res = await fetch("/api/social/upload", { method: "POST", body: fd });
      const j = await res.json().catch(() => ({}));
      if (!res.ok || !j.id) {
        setUploadErr("Upload failed — please try another image.");
        return;
      }
      setPhoto((prev) => {
        if (prev) URL.revokeObjectURL(prev.preview);
        return { id: j.id, preview: URL.createObjectURL(f) };
      });
      setStageLoading(true);
    } catch {
      setUploadErr("Upload failed — connection error.");
    } finally {
      setUploading(false);
    }
  }, []);

  const removePhoto = useCallback(() => {
    setPhoto((prev) => {
      if (prev) URL.revokeObjectURL(prev.preview);
      return null;
    });
    setStageLoading(true);
  }, []);

  /* Compatibility follows the live inputs (not the debounced applied text) so
     the filmstrip reacts as the user types. The selection never auto-switches:
     if the current template stops matching the content, a banner explains why
     and the user decides whether to adjust content or switch template. */
  const hasImage = !!photo;
  const hasTitle = title.trim().length > 0;
  const hasSubtitle = subtitle.trim().length > 0;
  const selected = ALL_TEMPLATES.find((t) => t.id === style);
  const selectedReason = selected
    ? incompatReason(selected, hasImage, hasTitle, hasSubtitle)
    : null;

  const download = useCallback(async () => {
    setDownloading(true);
    try {
      // The dedicated download endpoint also saves the PNG into the Library.
      const res = await fetch(
        renderUrl(style, theme, ratio, applied.title, applied.subtitle, 1080, photo?.id)
          .replace("/api/testing/render", "/api/social/download"),
      );
      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `cj_social_${style}_${theme}_${ratio.replace(":", "x")}.png`;
      a.click();
      URL.revokeObjectURL(a.href);
      setDownloadDone(true);
      setTimeout(() => setDownloadDone(false), 1600);
    } finally {
      setDownloading(false);
    }
  }, [style, theme, ratio, applied, photo]);

  return (
    <div className="cj-result">
      {/* ── left: content + ratio + colour controls ── */}
      <aside className="cj-panel left">
        <div className="cj-panel-body">
          {/* Two clusters — content above the line, format below. The field
              labels carry all the naming; section headers would repeat them. */}
          <div className="cj-grp">
            <div className="cj-field">
              <span className="cj-label">Image</span>
              {photo ? (
                <div className="cj-tw" style={{ alignSelf: "flex-start" }}>
                  <button
                    className="cj-thumb"
                    style={{ backgroundImage: `url(${photo.preview})` }}
                    onClick={() => fileRef.current?.click()}
                    title="Click to replace"
                    aria-label="Replace image"
                  />
                  <button className="cj-delbtn" onClick={removePhoto} aria-label="Remove image">
                    ✕
                  </button>
                </div>
              ) : (
                <button
                  className="cj-addtile"
                  onClick={() => fileRef.current?.click()}
                  aria-label="Upload image"
                  disabled={uploading}
                >
                  <PlusIcon size={24} />
                </button>
              )}
              {(uploading || uploadErr) && (
                <p className="cj-hint" style={uploadErr ? { color: "var(--color-text-danger)" } : undefined}>
                  {uploading ? "Uploading…" : uploadErr}
                </p>
              )}
            </div>
            <input
              ref={fileRef} type="file" hidden
              accept="image/jpeg,image/png,image/webp"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) uploadPhoto(f);
                e.target.value = "";
              }}
            />
            <div className="cj-field">
              <div className="cj-sec-row">
                <span className="cj-label">Title</span>
                <span className="cj-count">{title.length}/{TITLE_MAX}</span>
              </div>
              <input
                className="cj-desc-input"
                placeholder="Construction Junction"
                maxLength={TITLE_MAX}
                value={title}
                onChange={(e) => setTitle(e.target.value)}
              />
              {titleTruncated && (
                <p className="cj-hint cj-trunc-hint" role="status">
                  Part of the title won&apos;t fit this template — it&apos;ll be
                  shortened with &ldquo;…&rdquo; in the image.
                </p>
              )}
            </div>
            <div className="cj-field">
              <div className="cj-sec-row">
                <span className="cj-label">Subtitle (Optional)</span>
                <span className="cj-count">{subtitle.length}/{SUBTITLE_MAX}</span>
              </div>
              <input
                className="cj-desc-input"
                placeholder="Enter your short description here"
                maxLength={SUBTITLE_MAX}
                value={subtitle}
                onChange={(e) => setSubtitle(e.target.value)}
              />
            </div>
          </div>
          <div className="cj-grp">
            <div className="cj-field">
              <span className="cj-label">Photo Ratio</span>
              <div className="cj-ratio-row">
                {RATIOS.map((r) => (
                  <OptionCard
                    key={r.id}
                    active={ratio === r.id}
                    title={r.id}
                    label={r.label}
                    onClick={() => { if (ratio !== r.id) { setStageLoading(true); setRatio(r.id); } }}
                  />
                ))}
              </div>
            </div>
            <div className="cj-field">
              <span className="cj-label">Colors</span>
              <div className="cj-ratio-row">
                {THEMES.map((t) => (
                  <OptionCard
                    key={t.id}
                    active={theme === t.id}
                    dot={t.fill}
                    label={t.label}
                    onClick={() => { if (theme !== t.id) { setStageLoading(true); setTheme(t.id); } }}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>
        <div className="cj-foot cj-soc-foot">
          <button className="cj-btn-primary" disabled={downloading} onClick={download}>
            {downloading ? "Preparing…" : downloadDone ? "Saved ✓" : "Download"}
          </button>
        </div>
      </aside>

      {/* ── right: preview + style strip ── */}
      <div className="cj-canvas-col">
        <div className="cj-stage">
          <div className="cj-stagecard">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              key={`${style}-${theme}-${ratio}-${photo?.id ?? "fixed"}`}
              src={renderUrl(style, theme, ratio, applied.title, applied.subtitle, 900, photo?.id)}
              alt={`${style} ${theme} ${ratio}`}
              onLoad={() => { setStageLoading(false); setRenderErr(false); }}
              onError={() => { setStageLoading(false); setRenderErr(true); }}
            />
          </div>
          {renderErr && (
            <div className="cj-warn cj-stage-err">
              Preview failed to load — is the backend running? Change any option to retry.
            </div>
          )}
          {stageLoading && (
            <div className="cj-stage-loading"><div className="cj-spin" /></div>
          )}
        </div>
        {selectedReason && (
          <div className="cj-warn cj-film-warn">
            {selectedReason} — pick a highlighted template, or adjust your content.
          </div>
        )}
        <div className="cj-film">
          {TEMPLATE_GROUPS.map((g) => (
            <div className="cj-film-group" key={g.label}>
              <span className="cj-film-cat">{g.label}</span>
              {g.templates.map((t) => {
                const reason = incompatReason(t, hasImage, hasTitle, hasSubtitle);
                const photoBadge = !reason && needsPhoto(t, hasImage, hasTitle, hasSubtitle);
                const tip = reason ? `${t.label} — ${reason}`
                  : photoBadge ? `${t.label} — add a photo to complete it` : t.label;
                return (
                  <button
                    key={t.id}
                    title={tip}
                    aria-label={tip}
                    aria-disabled={!!reason}
                    className={`cj-tpl-thumb${t.id === style ? " active" : ""}${reason ? " dim" : ""}`}
                    onClick={() => {
                      if (reason || t.id === style) return;
                      setStageLoading(true);
                      setStyle(t.id);
                    }}
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={renderUrl(t.id, theme, ratio, applied.title, applied.subtitle, 200, photo?.id)} alt={t.label} />
                    {photoBadge && (
                      <span className="cj-needs-photo" aria-hidden>
                        <CameraPlusIcon size={12} />
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
