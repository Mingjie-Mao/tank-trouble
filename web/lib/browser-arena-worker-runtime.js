import { BrowserArena, loadArena } from "./browser-arena.js";
import { SearchPool, searchFor } from "./search-pool.js";

const FRAME_MS = 1000 / 25;
const BUDGET_MS = 30;

/**
 * Message-driven owner of BrowserArena. Production wires this to a dedicated
 * Web Worker; tests inject a fake clock and call advance() directly so the
 * threaded and direct execution paths can be compared frame for frame.
 * Given helper ports (`init.helpers`), the search's rollouts are shared with
 * helper workers holding copies of the arena, and the search widens.
 */
export class BrowserArenaWorkerRuntime {
  constructor({ emit, schedule, cancel, now } = {}) {
    this.emit = emit ?? (() => {});
    this.schedule = schedule ?? ((callback, delay) => setTimeout(callback, delay));
    this.cancel = cancel ?? ((handle) => clearTimeout(handle));
    this.now = now ?? (() => performance.now());
    this.arena = null;
    this.pool = null;
    this.running = false;
    this.timer = null;
    this.nextFrameAt = 0;
    this.inFlight = Promise.resolve();
    this.budgetMs = BUDGET_MS;
  }

  async handle(message) {
    if (message?.type === "init") {
      this.stop();
      await this.inFlight;
      await loadArena(message.wasm);
      const seed = Number(message.seed ?? 970000);
      this.budgetMs = message.budget_ms ?? BUDGET_MS;
      this.arena = new BrowserArena({ seed });
      this.pool?.break();
      this.pool = null;
      if (message.helpers?.length) {
        this.pool = new SearchPool(message.helpers, { onBroken: () => this.narrow() });
        this.pool.broadcast({ type: "init", wasm: message.wasm, seed });
        this.apply({ action: "search", ...searchFor(this.pool.workers), budget_ms: this.budgetMs });
      }
      const state = this.state();
      this.emit({ type: "state", state });
      if (message.autostart !== false) this.start();
      return state;
    }
    if (message?.type === "command") {
      if (this.arena === null) throw new Error("browser arena worker is not initialised");
      await this.inFlight;
      try {
        const state = this.apply(message.payload ?? {});
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
      this.pool?.break();
      return null;
    }
    throw new Error(`unknown worker message: ${message?.type}`);
  }

  /** Run a command here and on every copy (a random maze gets its seed here). */
  apply(payload) {
    const resolved = payload.action === "new_maze" && (payload.seed === undefined || payload.seed === "" || payload.seed === "random")
      ? { ...payload, seed: Math.floor(Math.random() * 2147483647) }
      : payload;
    this.arena.command(resolved);
    if (this.pool && !this.pool.broken) this.pool.broadcast({ type: "command", payload: resolved });
    return this.state();
  }

  /** A copy drifted or failed: search alone, at the size one worker affords. */
  narrow() {
    this.arena?.command({ action: "search", ...searchFor(1), budget_ms: this.budgetMs });
  }

  state() {
    const state = this.arena.state();
    state.runtime.search_workers = this.pool && !this.pool.broken ? this.pool.workers : 1;
    return state;
  }

  async advance(now = this.now()) {
    if (this.arena === null) throw new Error("browser arena worker is not initialised");
    const pool = this.pool && !this.pool.broken ? this.pool : null;
    if (pool) {
      const state = await this.arena.stepWith((plan) => pool.search(plan, (request) => this.arena.rollouts(request), this.budgetMs), now);
      if (!state.paused && !pool.broken) pool.broadcast({ type: "follow", frame: this.arena.frame() });
    } else {
      this.arena.step(now);
    }
    const state = this.state();
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
    this.timer = this.schedule(async () => {
      if (!this.running) return;
      const current = this.now();
      this.inFlight = this.advance(current);
      await this.inFlight;
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
