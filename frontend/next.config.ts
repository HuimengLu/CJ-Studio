import type { NextConfig } from "next";

// All /api/* calls proxy to the FastAPI backend, so the browser only ever
// talks same-origin (no CORS in dev or prod). Override for deployment.
const backend = process.env.BACKEND_URL ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  // The dev-tools floating indicator intercepts pointer events over the
  // canvas (breaks hover/drag on dimension annotations in dev).
  devIndicators: false,
  // The /api/* rewrite proxies through Next, which buffers the request body
  // with a 10MB default cap — anything larger is silently truncated and the
  // proxied upload fails. Raise it to match the 200MB the upload UI advertises.
  // proxyTimeout: gpt-image generations can run past the 30s default, which
  // kills the socket mid-request ("network error" + a costly duplicate retry).
  experimental: {
    proxyClientMaxBodySize: "200mb",
    proxyTimeout: 300_000,
  },
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${backend}/api/:path*` }];
  },
};

export default nextConfig;
