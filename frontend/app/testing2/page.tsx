"use client";

import { useCallback, useRef, useState } from "react";

/* Testing 2 — shadow-method comparison. Upload one image; the backend removes
   the background once and composites five shadow strategies (S1–S5) under the
   same cut-out, shown side by side so we can eyeball which grounds the item
   most naturally on the backdrop. */

type Panel = { key: string; label: string; desc: string };

export default function Testing2Page() {
  const [phase, setPhase] = useState<"home" | "processing" | "result">("home");
  const [id, setId] = useState<string | null>(null);
  const [panels, setPanels] = useState<Panel[]>([]);
  const [preview, setPreview] = useState<string | null>(null);
  const [warn, setWarn] = useState<string | null>(null);
  const [dragover, setDragover] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const run = useCallback(async (files: File[]) => {
    const f = files.find((x) => x.type.startsWith("image/"));
    if (!f) return;
    setWarn(null);
    setPreview(URL.createObjectURL(f));
    setPhase("processing");
    try {
      const fd = new FormData();
      fd.append("file", f);
      const res = await fetch("/api/testing2", { method: "POST", body: fd });
      const j = await res.json();
      if (!res.ok || !j.id) {
        setWarn(j.warn ?? "processing failed");
        setPhase("home");
        return;
      }
      setId(j.id);
      setPanels(j.panels);
      setPhase("result");
    } catch {
      setWarn("network error");
      setPhase("home");
    }
  }, []);

  const reset = () => {
    setPhase("home");
    setId(null);
    setPanels([]);
    setPreview(null);
    setWarn(null);
  };

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
          onDrop={(e) => { e.preventDefault(); setDragover(false); run(Array.from(e.dataTransfer.files)); }}
        >
          <div className="cj-drop-icon"><span className="ms">biotech</span></div>
          <div className="cj-drop-title">Compare shadow methods (S1–S5)</div>
          <div className="cj-drop-hint">Upload a product photo · JPG · PNG · WEBP</div>
          <input
            ref={fileRef} type="file" hidden
            accept="image/jpeg,image/png,image/webp"
            onChange={(e) => { run(Array.from(e.target.files ?? [])); e.target.value = ""; }}
          />
        </div>
      </div>
    );
  }

  /* ═══ processing ═══ */
  if (phase === "processing") {
    return (
      <div className="cj-home">
        <div className="cj-home-top" />
        <div className="cj-proc" style={{ alignItems: "center", gap: 20 }}>
          <div className="cj-spin" />
          <div style={{ color: "var(--muted, #888)" }}>Removing background, compositing five shadow styles…</div>
        </div>
      </div>
    );
  }

  /* ═══ result ═══ */
  return (
    <div style={{ padding: "24px 28px", overflowY: "auto", height: "100%" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 18 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          {preview && (
            /* eslint-disable-next-line @next/next/no-img-element */
            <img src={preview} alt="source" style={{ width: 52, height: 52, objectFit: "cover", borderRadius: 8, border: "1px solid var(--line-soft, #e5e5e5)" }} />
          )}
          <div>
            <h2 style={{ margin: 0, fontSize: 22 }}>Shadow-method comparison</h2>
            <div style={{ fontSize: 13, color: "var(--muted, #888)" }}>Same cut-out, five shadow strategies</div>
          </div>
        </div>
        <button className="cj-btn-outline" onClick={reset}>Try another photo</button>
      </div>

      {warn && <div className="cj-warn">&#9888; {warn}</div>}

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 18 }}>
        {panels.map((p) => (
          <div key={p.key} style={{ border: "1px solid var(--line-soft, #e5e5e5)", borderRadius: 12, overflow: "hidden", background: "var(--card, #fff)" }}>
            <div style={{ aspectRatio: "1 / 1", background: "#f4f4ef" }}>
              {id && (
                /* eslint-disable-next-line @next/next/no-img-element */
                <img
                  src={`/api/testing2/${id}/${p.key}?w=800`}
                  alt={p.label}
                  style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
                />
              )}
            </div>
            <div style={{ padding: "12px 14px 14px" }}>
              <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
                <span style={{ fontSize: 12, fontWeight: 700, letterSpacing: 0.5, color: "var(--accent, #005618)" }}>{p.key}</span>
                <span style={{ fontSize: 15, fontWeight: 600 }}>{p.label}</span>
              </div>
              <p style={{ margin: "6px 0 0", fontSize: 12.5, lineHeight: 1.5, color: "var(--muted, #777)" }}>{p.desc}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
