import { cp, mkdir, rename, rm } from "node:fs/promises";
import path from "node:path";

const root = process.cwd();
const output = path.join(root, "dist", "pages");
const worker = path.join(output, "_worker.js");

await rm(output, { recursive: true, force: true });
await cp(path.join(root, "dist", "client"), output, { recursive: true });
await mkdir(worker, { recursive: true });
await cp(path.join(root, "dist", "server"), worker, { recursive: true });
await rename(path.join(worker, "index.js"), path.join(worker, "vinext-index.js"));
await cp(
  path.join(root, "cloudflare-pages", "worker-wrapper.js"),
  path.join(worker, "index.js"),
);

console.log(`prepared ${path.relative(root, output)}`);
