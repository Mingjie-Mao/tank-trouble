import init, { initSync, Arena } from "./wasm/tank.js";

let ready = null;

/**
 * Load the arena's WebAssembly once: from a URL (the page resolves it, since
 * a bundled worker cannot) or, in Node, from the module bytes.
 */
export function loadArena(source) {
  ready ??= typeof source === "string" || source instanceof URL
    ? init({ module_or_path: source })
    : Promise.resolve(initSync({ module: source }));
  return ready;
}

/**
 * The page's fixed-step duel, run by the Rust engine compiled to WebAssembly
 * (engine/wasm): native Laika, the deployed controller and a random driver.
 * A dedicated Worker owns it; call loadArena() first.
 */
export class BrowserArena {
  constructor({ seed = 970000 } = {}) {
    this.arena = new Arena(Number(seed) >>> 0);
    this.latest = this.arena.state();
    this.lastStepMs = 0;
    this.stepTimestamps = [];
  }

  step(now = performance.now()) {
    const started = performance.now();
    this.latest = this.arena.step();
    this.lastStepMs = performance.now() - started;
    this.stepTimestamps.push(now);
    if (this.stepTimestamps.length > 120) this.stepTimestamps.shift();
    return this.state();
  }

  /**
   * step() with the searches played by `play(plan)` (a SearchPool): returns
   * a promise of the state.
   */
  async stepWith(play, now = performance.now()) {
    const started = performance.now();
    const plans = JSON.parse(this.arena.plan());
    const values = [];
    for (const plan of plans) values.push(await play(plan));
    this.latest = this.arena.finish(JSON.stringify(values), plans.length ? performance.now() - started : 0);
    this.lastStepMs = performance.now() - started;
    this.stepTimestamps.push(now);
    if (this.stepTimestamps.length > 120) this.stepTimestamps.shift();
    return this.state();
  }

  /** This worker's share of a planned search. */
  rollouts(request) {
    return JSON.parse(this.arena.rollouts(JSON.stringify(request)));
  }

  /** The last frame's moves, for copies kept in step. */
  frame() {
    return this.arena.frame();
  }

  command(payload) {
    this.latest = this.arena.command(JSON.stringify(payload ?? {}));
    return this.state();
  }

  state() {
    const state = JSON.parse(this.latest);
    const t = this.stepTimestamps;
    state.effective_fps = t.length > 1 ? ((t.length - 1) * 1000) / (t.at(-1) - t[0]) : 0;
    state.runtime.step_ms = this.lastStepMs;
    return state;
  }
}
