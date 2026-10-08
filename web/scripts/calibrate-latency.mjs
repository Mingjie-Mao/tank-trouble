/**
 * Node side of the Node<->browser latency calibration.
 *
 * Records exactly what the deployed page displays: `telemetry.p95_ms`, which
 * is percentile(decisionSamples, 0.95) over a rolling 600-frame window of
 * `agent.drive(game)` for side 0. The page is watched by a human over time, so
 * the comparable quantity is the DISTRIBUTION of that rolling reading, not a
 * single pooled percentile — pooling is what every previous report in this
 * repository did, and it is why the published figure is ~20 ms.
 *
 * Sampled every 12 frames to match a browser sampler polling at 500 ms / 25 FPS.
 */
import { BrowserArena } from "../lib/browser-arena.js";

const policy = process.argv[2] ?? "p27-js-tactical-v3";
const frames = Number(process.argv[3] ?? 25000);
const seed = Number(process.argv[4] ?? 970000);

const arena = new BrowserArena({ seed });
arena.command({
  action: "mode", mode: "watch", left_policy: policy, right_policy: "laika-js",
});

const rolling = [];
for (let frame = 0; frame < frames; frame += 1) {
  arena.step(frame * 40);
  if (frame >= 600 && frame % 12 === 0) rolling.push(arena.state().telemetry.p95_ms);
}

const state = arena.state();
const sorted = rolling.slice().sort((a, b) => a - b);
const q = (p) => Number(sorted[Math.floor(p * (sorted.length - 1))].toFixed(2));

console.log(JSON.stringify({
  policy,
  frames,
  seed,
  rounds: state.round ?? null,
  wallSliding: arena.wallSliding,
  decision_p50_ms: Number(state.telemetry.p50_ms.toFixed(3)),
  rolling_p95_samples: sorted.length,
  rolling_p95_min: q(0),
  rolling_p95_median: q(0.5),
  rolling_p95_p90: q(0.9),
  rolling_p95_max: q(1),
  share_over_40ms: Number(
    (100 * rolling.filter((v) => v > 40).length / rolling.length).toFixed(1),
  ),
}, null, 2));
