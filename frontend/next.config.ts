import type { NextConfig } from "next";

// All /api/* calls proxy to the FastAPI backend, so the browser only ever
// talks same-origin (no CORS in dev or prod). Override for deployment.
const backend = process.env.BACKEND_URL ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${backend}/api/:path*` }];
  },
};

export default nextConfig;
