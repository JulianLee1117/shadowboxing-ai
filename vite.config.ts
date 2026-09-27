import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

const privacyHeaders = {
  "Content-Security-Policy":
    "default-src 'self'; script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; media-src 'self' blob:; worker-src 'self' blob:; connect-src 'self' ws://127.0.0.1:5173 ws://localhost:5173; object-src 'none'; base-uri 'self'; frame-ancestors 'self'",
  "Referrer-Policy": "no-referrer",
  "Permissions-Policy": "camera=(self), microphone=()",
};

const localPoseProxy = {
  "/local-pose": {
    target: "http://127.0.0.1:8765",
    changeOrigin: true,
    rewrite: (path: string) => path.replace(/^\/local-pose/, ""),
  },
};

export default defineConfig({
  // Unit tests and the running studio must not rewrite each other's optimized
  // dependency metadata; doing so leaves open tabs with stale module URLs.
  cacheDir: process.env.VITEST
    ? "node_modules/.vite-vitest"
    : "node_modules/.vite",
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    headers: privacyHeaders,
    proxy: localPoseProxy,
  },
  preview: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    headers: privacyHeaders,
    proxy: localPoseProxy,
  },
  test: { include: ["src/**/*.test.ts"], environment: "node" },
  worker: { format: "es" },
});
