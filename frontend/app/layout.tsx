import type { Metadata } from "next";
import { Hanken_Grotesk, Libre_Caslon_Text } from "next/font/google";
import { Analytics } from "@vercel/analytics/next";
import { SpeedInsights } from "@vercel/speed-insights/next";
import "./globals.css";
import Sidebar from "@/components/Sidebar";

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

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${caslon.variable} ${hanken.variable}`}>
      <head>
        {/* Material Symbols for sidebar / UI icons */}
        <link
          href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0..1,0&display=block"
          rel="stylesheet"
        />
      </head>
      <body>
        <Sidebar />
        <main className="cj-main">{children}</main>
        <Analytics />
        <SpeedInsights />
      </body>
    </html>
  );
}
