"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const items = [
  { href: "/", icon: "sell", label: "New Listing" },
  { href: "/social", icon: "grid_view", label: "Social" },
  { href: "/testing", icon: "science", label: "Testing" },
  { href: "#", icon: "photo_library", label: "Library" },
  { href: "/testing2", icon: "biotech", label: "Testing 2" },
];

export default function Sidebar() {
  const pathname = usePathname();
  return (
    <div id="cj-side">
      <div className="cj-brand">
        CJ&nbsp;Studio<small>Internal Tools</small>
      </div>
      <div className="cj-navcol">
        {items.map((it) => (
          <Link
            key={it.label}
            href={it.href}
            className={`cj-item${pathname === it.href ? " active" : ""}`}
            // The Listing item returns to its upload screen. When already on "/",
            // Link can't remount the page, so signal it to reset its own state.
            onClick={() => {
              if (it.href === "/") {
                window.dispatchEvent(new Event("cj:reset-listing"));
              }
            }}
          >
            <span className="ms">{it.icon}</span>
            <span>{it.label}</span>
          </Link>
        ))}
      </div>
      <div>
        <Link href="#" className="cj-item">
          <span className="ms">settings</span>
          <span>Settings</span>
        </Link>
      </div>
    </div>
  );
}
