import { Arena } from "./wasm/tank.js";
import { loadArena } from "./browser-arena.js";

/**
 * Our AI's search by how many workers share its rollouts (this one
 * included): 4×4 deals each at most three of its up to 20 rollouts.
 */
export function searchFor(workers) {
  return workers >= 7 ? { candidates: 4, rollouts: 4 } : { candidates: 2, rollouts: 2 };
}

/** How long past the budget a helper's answer is still waited for. */
const GRACE_MS = 5;

/**
 * The arena worker's side of a parallel search. Each helper keeps a copy of
 * the arena in step (every command, then every frame's moves); a danger
 * frame's rollouts are dealt round-robin to this worker and the helpers, and
 * a helper that answers late counts as unplayed.
 */
export class SearchPool {
  constructor(ports, { onBroken } = {}) {
    this.ports = ports;
    this.onBroken = onBroken ?? (() => {});
    this.broken = false;
    this.nextId = 0;
    this.waiting = new Map();
    ports.forEach((port) => {
      port.onmessage = (event) => this.receive(event.data);
      port.start?.();
    });
  }

  get workers() {
    return this.ports.length + 1;
  }

  broadcast(message) {
    for (const port of this.ports) port.postMessage(message);
  }

  receive(message) {
    if (message?.type === "values") {
      this.waiting.get(message.id)?.(message.values);
      this.waiting.delete(message.id);
    } else if (message?.type === "desync" || message?.type === "error") {
      this.break();
    }
  }

  break() {
    if (this.broken) return;
    this.broken = true;
    for (const resolve of this.waiting.values()) resolve(null);
    this.waiting.clear();
    this.onBroken();
  }

  ask(port, request, patienceMs) {
    const id = (this.nextId += 1);
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        this.waiting.delete(id);
        resolve(null);
      }, patienceMs);
      this.waiting.set(id, (values) => {
        clearTimeout(timer);
        resolve(values);
      });
      port.postMessage({ type: "rollouts", id, request });
    });
  }

  /**
   * Play one planned search (from Arena.plan) within `budgetMs`; `local`
   * plays this worker's share. Values are candidate-major, null where unplayed.
   */
  async search(plan, local, budgetMs) {
    const jobs = plan.candidates.flatMap((_, i) => plan.seeds.map((_, j) => [i, j]));
    const shares = Array.from({ length: this.workers }, () => []);
    jobs.forEach((_, k) => shares[k % this.workers].push(k));
    const request = (share) => ({ ...plan, jobs: share.map((k) => jobs[k]), budget_ms: budgetMs });
    const replies = this.ports.map((port, w) =>
      shares[w + 1].length ? this.ask(port, request(shares[w + 1]), budgetMs + GRACE_MS) : Promise.resolve([]));
    const values = jobs.map(() => null);
    const fill = (share, got) => share.forEach((k, x) => { values[k] = got?.[x] ?? null; });
    fill(shares[0], local(request(shares[0])));
    (await Promise.all(replies)).forEach((got, w) => fill(shares[w + 1], got));
    return values;
  }
}

/**
 * A helper worker's copy of the arena, driven over `port` by a SearchPool.
 */
export class SearchHelper {
  constructor(port) {
    this.port = port;
    this.arena = null;
    this.queue = Promise.resolve();
    port.onmessage = (event) => {
      this.queue = this.queue.then(() => this.handle(event.data)).catch((error) => {
        port.postMessage({ type: "error", message: error instanceof Error ? error.message : String(error) });
      });
    };
    port.start?.();
  }

  async handle(message) {
    if (message?.type === "init") {
      await loadArena(message.wasm);
      this.arena?.free();
      this.arena = new Arena(Number(message.seed) >>> 0);
    } else if (message?.type === "command") {
      this.arena.command(JSON.stringify(message.payload ?? {}));
    } else if (message?.type === "follow") {
      if (!this.arena.follow(message.frame)) this.port.postMessage({ type: "desync" });
    } else if (message?.type === "rollouts") {
      const values = JSON.parse(this.arena.rollouts(JSON.stringify(message.request)));
      this.port.postMessage({ type: "values", id: message.id, values });
    }
  }
}
