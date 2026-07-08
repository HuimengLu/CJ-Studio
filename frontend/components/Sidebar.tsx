"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const items = [
  { href: "/", icon: "upload_file", label: "Upload" },
  { href: "/social", icon: "grid_view", label: "Social" },
  { href: "#", icon: "photo_library", label: "Library" },
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
