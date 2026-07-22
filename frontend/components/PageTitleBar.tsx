"use client";

import { usePathname } from "next/navigation";

/* Sticky page-title bar (platform-console spec): 12px 24px padding, white
   surface, hairline bottom border, 20px/32px semibold title — 57px total. */
const TITLES: Record<string, string> = {
  "/": "New Listing",
  "/legacy": "Legacy Listing",
  "/social": "Social",
  "/library": "Library",
};

export default function PageTitleBar() {
  const pathname = usePathname();
  const title = TITLES[pathname];
  if (!title) return null;
  return (
    <div className="cj-pagebar">
      <h1>{title}</h1>
    </div>
  );
}
