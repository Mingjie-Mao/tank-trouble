import { BrowserArenaWorkerRuntime } from "../lib/browser-arena-worker-runtime.js";

const runtime = new BrowserArenaWorkerRuntime({
  emit: (message) => globalThis.postMessage(message),
});

// Messages are handled in order: commands wait for the WebAssembly to load.
let queue = Promise.resolve();
globalThis.onmessage = (event) => {
  queue = queue.then(() => runtime.handle(event.data)).catch((error) => {
    globalThis.postMessage({
      type: "error",
      id: event.data?.id,
      message: error instanceof Error ? error.message : String(error),
    });
  });
};
