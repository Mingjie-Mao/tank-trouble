import { BrowserArenaWorkerRuntime } from "../lib/browser-arena-worker-runtime.js";

const runtime = new BrowserArenaWorkerRuntime({
  emit: (message) => globalThis.postMessage(message),
});

globalThis.onmessage = (event) => {
  try {
    runtime.handle(event.data);
  } catch (error) {
    globalThis.postMessage({
      type: "error",
      id: event.data?.id,
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
