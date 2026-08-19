import { BrowserArena } from "./browser-arena.js";

const FRAME_MS = 1000 / 25;

/**
 * Message-driven owner of BrowserArena. Production wires this to a dedicated
 * Web Worker; tests inject a fake clock and call advance() directly so the
 * threaded and direct execution paths can be compared frame for frame.
 */
export class BrowserArenaWorkerRuntime {
  constructor({ emit, schedule, cancel, now } = {}) {
    this.emit = emit ?? (() => {});
    this.schedule = schedule ?? ((callback, delay) => setTimeout(callback, delay));
    this.cancel = cancel ?? ((handle) => clearTimeout(handle));
    this.now = now ?? (() => performance.now());
    this.arena = null;
    this.running = false;
    this.timer = null;
    this.nextFrameAt = 0;
  }

  handle(message) {
    if (message?.type === "init") {
      this.stop();
      this.arena = new BrowserArena({
        seed: Number(message.seed ?? 970000),
        // K4 wall-contact physics is a property of the world, not of a
        // policy, so the deployed arena runs it for both tanks. The library
        // default stays false so every frozen baseline and the Python port
        // fidelity checks keep describing the original collision model.
        wallSliding: true,
      });
      const state = this.arena.state();
      this.emit({ type: "state", state });
      if (message.autostart !== false) this.start();
      return state;
    }
    if (message?.type === "command") {
      if (this.arena === null) throw new Error("browser arena worker is not initialised");
      try {
        const state = this.arena.command(message.payload ?? {});
        this.emit({ type: "response", id: message.id, state });
        return state;
      } catch (error) {
        this.emit({
          type: "error",
          id: message.id,
          message: error instanceof Error ? error.message : String(error),
        });
        return null;
      }
    }
    if (message?.type === "dispose") {
      this.stop();
      return null;
    }
    throw new Error(`unknown worker message: ${message?.type}`);
  }

  advance(now = this.now()) {
    if (this.arena === null) throw new Error("browser arena worker is not initialised");
    this.arena.step(now);
    const state = this.arena.state();
    this.emit({ type: "state", state });
    return state;
  }

  start() {
    if (this.running || this.arena === null) return;
    this.running = true;
    this.nextFrameAt = this.now() + FRAME_MS;
    this.queueNext();
  }

  queueNext() {
    if (!this.running) return;
    const delay = Math.max(0, this.nextFrameAt - this.now());
    this.timer = this.schedule(() => {
      if (!this.running) return;
      const current = this.now();
      this.advance(current);
      this.nextFrameAt += FRAME_MS;
      if (current - this.nextFrameAt > FRAME_MS * 4) {
        this.nextFrameAt = current + FRAME_MS;
      }
      this.queueNext();
    }, delay);
  }

  stop() {
    this.running = false;
    if (this.timer !== null) this.cancel(this.timer);
    this.timer = null;
  }
}
