/* Save a generated file on any device.

   Mobile browsers cannot write to the photo library from a download link —
   an <a download> on iOS Safari either opens the image or drops it in the
   Files app. The closest thing to "save to camera roll" the web allows is
   the system share sheet (navigator.share with a File), where the user taps
   "Save Image" and it lands in the photo album. So: on touch devices,
   shareable PNGs go through the share sheet; everywhere else (and for ZIPs,
   which the photo album can't hold) a plain anchor download. */
export async function saveBlob(blob: Blob, filename: string): Promise<void> {
  const coarse =
    typeof window !== "undefined" && window.matchMedia?.("(pointer: coarse)").matches;
  if (coarse && filename.endsWith(".png") && typeof navigator !== "undefined" && navigator.canShare) {
    const file = new File([blob], filename, { type: "image/png" });
    if (navigator.canShare({ files: [file] })) {
      try {
        await navigator.share({ files: [file] });
        return;
      } catch (e) {
        // AbortError = the user closed the sheet on purpose — not a failure.
        if ((e as DOMException)?.name === "AbortError") return;
        // Anything else (e.g. the user-gesture window expired while the file
        // was being fetched) falls through to a regular download.
      }
    }
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}
