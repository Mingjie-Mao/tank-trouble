import app from "./vinext-index.js";

const STATIC_PREFIXES = ["/_next/static/"];
const STATIC_FILES = new Set([
  "/favicon.svg",
  "/file.svg",
  "/globe.svg",
  "/window.svg",
  "/og.png",
  "/league-tactical-latest.json",
  "/league-killfield-latest.json",
]);

function isStaticAsset(pathname) {
  return STATIC_FILES.has(pathname)
    || STATIC_PREFIXES.some((prefix) => pathname.startsWith(prefix));
}

export default {
  async fetch(request, env, context) {
    const pathname = new URL(request.url).pathname;
    if (isStaticAsset(pathname)) {
      const response = await env.ASSETS.fetch(request);
      if (response.status !== 404) return response;
    }
    return app.fetch(request, env, context);
  },
};
