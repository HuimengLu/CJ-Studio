import type { ReactNode } from "react";

/* Shared option card for every ratio / color picker — one component, one look.
   Shows either a text title (ratio digits) or a color swatch, with a small
   uppercase label under it. With `stack`, multi-word labels break one word
   per line so sibling cards keep equal label heights ("FACEBOOK POST" and
   "INSTAGRAM STORY" both take two lines); without it the label stays on a
   single line. */
export default function OptionCard({
  active,
  onClick,
  title,
  dot,
  label,
  stack = false,
}: {
  active: boolean;
  onClick: () => void;
  /** Card headline (e.g. the ratio digits). Ignored when `dot` is given. */
  title?: ReactNode;
  /** CSS color — renders a round swatch instead of a title. */
  dot?: string;
  label: string;
  /** Break the label one word per line (for label sets of equal word count). */
  stack?: boolean;
}) {
  return (
    <button
      className={`cj-ratio-card${active ? " active" : ""}`}
      onClick={onClick}
      title={label}
    >
      {dot ? <span className="cj-color-dot" style={{ background: dot }} /> : title}
      <small className={stack ? "stack" : undefined}>
        {stack ? label.split(" ").join("\n") : label}
      </small>
    </button>
  );
}
