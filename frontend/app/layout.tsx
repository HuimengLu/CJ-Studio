import type { Metadata, Viewport } from "next";
import { Hanken_Grotesk, Libre_Caslon_Text } from "next/font/google";
import { Analytics } from "@vercel/analytics/next";
import { SpeedInsights } from "@vercel/speed-insights/next";
import "./globals.css";
import Sidebar from "@/components/Sidebar";
import PageTitleBar from "@/components/PageTitleBar";

const caslon = Libre_Caslon_Text({
  weight: ["400", "700"],
  style: ["normal", "italic"],
  subsets: ["latin"],
  variable: "--font-caslon",
});

const hanken = Hanken_Grotesk({
  subsets: ["latin"],
  variable: "--font-hanken",
});

export const metadata: Metadata = {
  title: "CJ Studio",
  description: "Construction Junction internal tools",
};

/* viewport-fit=cover lets the mobile top bar / pinned Download extend into
   the notch and home-indicator areas via env(safe-area-inset-*). */
export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${caslon.variable} ${hanken.variable}`}>
      <body>
        <Sidebar />
        <main className="cj-main">
          <PageTitleBar />
          {children}
        </main>
        <Analytics />
        <SpeedInsights />
      </body>
    </html>
  );
}
