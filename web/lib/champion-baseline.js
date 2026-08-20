export const CHAMPION_BASELINE = Object.freeze({
  policy: "p27-js-tactical-v3",
  name: "Tactical Smooth champion",
  status: "frozen",
  engine: "browser-js-search",
  architecture: [
    "KillField H36 attack search",
    "visible-bullet safety verification",
    "sparse two-stage correction",
    "post-kill settlement survival",
    "topology anti-stall pursuit",
    "moving-target ricochet interception",
    "pre-fire settlement safety audit",
    "K4 wall-contact physics",
    "K1 fire continuation",
  ],
  // Pre-registered non-inferiority holdout on a seed base that never informed
  // a tuning decision. Reported instead of the tuning bases so the headline
  // number is not the one the candidate was selected on.
  laikaBenchmark: Object.freeze({
    rounds: 2000,
    seed: 3500000,
    wins: 1912,
    losses: 66,
    doubleDeaths: 22,
    draws: 0,
    winRate: 0.956,
    predecessorWins: 1914,
    predecessorWinRate: 0.957,
    pairedDifferencePp: -0.1,
    pairedCi95Pp: Object.freeze([-1.34, 1.14]),
    nonInferiorityMarginPp: -1.5,
    killfieldWins: 262,
    killfieldWinRate: 0.8733333333333333,
  }),
  opponentPoolBenchmark: Object.freeze({
    rounds: 320,
    seeds: Object.freeze([4100000]),
    wins: 302,
    losses: 10,
    doubleDeaths: 4,
    draws: 4,
    winRate: 0.94375,
    colorGap: 0.0375,
    predecessorWins: 298,
  }),
  // Isolated single-process runs. Figures measured while other evaluations run
  // in parallel inflate p95 by up to 1.9x and are not admissible for the gate.
  runtimeBenchmark: Object.freeze({
    decisionP50Ms: 0.66,
    decisionP95Ms: 20.05,
    frameBudgetMs: 40,
  }),
  movementQuality: Object.freeze({
    wallContactFrames: 0.03,
    zeroMotionUnderCommand: 0.0012,
    directionChangesPerSecond: 3.6,
    predecessorWallContactFrames: 0.2135,
    predecessorZeroMotionUnderCommand: 0.1252,
    predecessorDirectionChangesPerSecond: 7.23,
  }),
  promotionGate: Object.freeze({
    // Pooled across both required seed bases, never the better one. The old
    // value of 0.97 was a single base's result and would reject candidates
    // that actually meet the published standard.
    minimumLaikaWinRate: 0.96,
    maximumLaikaRegression: 0,
    maximumOpponentPoolRegression: 0.02,
    maximumP95DecisionMs: 40,
    maximumColorGap: 0.154,
  }),
});
