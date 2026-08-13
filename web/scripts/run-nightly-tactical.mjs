import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { performance } from "node:perf_hooks";

import { BrowserArena } from "../lib/browser-arena.js";
import {
  DEFAULT_LEAGUE_OPPONENTS, playLeagueGame, summarizeLeague,
} from "../lib/browser-league.js";
import { CHAMPION_BASELINE } from "../lib/champion-baseline.js";
import { summarizeSelfplayFlywheel } from "../lib/selfplay-flywheel.js";

function argument(name, fallback) {
  const index = process.argv.indexOf(`--${name}`);
  return index === -1 ? fallback : process.argv[index + 1];
}

function percentile(values, q) {
  if (!values.length) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1)))];
}

const dateId = new Date().toISOString().slice(0, 10);
const runId = argument("run-id", dateId);
const outputRoot = argument("output-root", "data/nightly");
const runDir = path.join(outputRoot, runId);
const candidates = argument(
  "candidates",
  "p27-js-tactical-opponent-model,p27-js-tactical-combined",
).split(",").map((value) => value.trim()).filter(Boolean);
const roundsPerSide = Number(argument("rounds-per-side", "13"));
const maxFrames = Number(argument("max-frames", "3000"));
const blindRounds = Number(argument("blind-rounds", "300"));
const regressionPath = argument("regression", "data/regression/tactical-champion.json");
const dayIndex = Math.floor(new Date(`${runId}T00:00:00Z`).getTime() / 86400000);
if (!Number.isFinite(dayIndex)) throw new Error(`run-id must begin with an ISO date: ${runId}`);
const poolSeed = Number(argument("pool-seed", String(3000000 + dayIndex * 1000)));
const blindSeed = Number(argument("blind-seed", String(poolSeed + 500000)));
const checkpointPath = path.join(runDir, "checkpoint.json");
const summaryPath = path.join(runDir, "summary.json");
const latestPath = path.join(outputRoot, "latest.json");

await mkdir(runDir, { recursive: true });
const checkpoint = existsSync(checkpointPath)
  ? JSON.parse(await readFile(checkpointPath, "utf8"))
  : { schemaVersion: 1, runId, poolSeed, blindSeed, pool: {}, regression: {}, blind: {} };

async function saveCheckpoint() {
  await writeFile(checkpointPath, `${JSON.stringify(checkpoint, null, 2)}\n`, "utf8");
}

function poolCases() {
  const cases = [];
  for (let opponentIndex = 0; opponentIndex < DEFAULT_LEAGUE_OPPONENTS.length; opponentIndex += 1) {
    const opponent = DEFAULT_LEAGUE_OPPONENTS[opponentIndex];
    for (let index = 0; index < roundsPerSide; index += 1) {
      for (const candidateSide of [0, 1]) {
        cases.push({
          opponent,
          candidateSide,
          seed: poolSeed + opponentIndex * 100000 + index,
        });
      }
    }
  }
  return cases;
}

async function ensurePool(policy) {
  checkpoint.pool[policy] ??= {};
  const cases = poolCases();
  for (const item of cases) {
    const key = `${item.opponent}:${item.candidateSide}:${item.seed}`;
    if (checkpoint.pool[policy][key]) continue;
    checkpoint.pool[policy][key] = playLeagueGame({
      candidate: policy,
      ...item,
      maxFrames,
    });
    await saveCheckpoint();
    process.stdout.write(`\rpool ${policy} ${Object.keys(checkpoint.pool[policy]).length}/${cases.length}   `);
  }
  process.stdout.write("\n");
  return summarizeLeague(Object.values(checkpoint.pool[policy]), {
    candidate: policy, roundsPerSide, seed: poolSeed, maxFrames,
  });
}

async function ensureRegression(policy, regression) {
  checkpoint.regression[policy] ??= {};
  for (const item of regression.cases) {
    if (checkpoint.regression[policy][item.key]) continue;
    checkpoint.regression[policy][item.key] = {
      ...playLeagueGame({
        candidate: policy,
        opponent: item.opponent,
        candidateSide: item.candidateSide,
        seed: item.seed,
        maxFrames,
      }),
      baselineOutcome: item.baselineOutcome,
      tags: item.tags,
    };
    await saveCheckpoint();
    process.stdout.write(`\rregression ${policy} ${Object.keys(checkpoint.regression[policy]).length}/${regression.cases.length}   `);
  }
  process.stdout.write("\n");
  const games = Object.values(checkpoint.regression[policy]);
  const regressions = games.filter((row) => (
    row.baselineOutcome === "win" && row.outcome !== "win"
  ));
  const recoveries = games.filter((row) => (
    row.baselineOutcome !== "win" && row.outcome === "win"
  ));
  return { games, regressions, recoveries, passed: regressions.length === 0 };
}

function playWatchGame(policy, seed) {
  const arena = new BrowserArena({ seed });
  arena.command({ action: "mode", mode: "watch", left_policy: policy, right_policy: "laika-js" });
  const decisionMs = [];
  const started = performance.now();
  for (let frame = 0; frame < maxFrames; frame += 1) {
    arena.step(frame * 40);
    if (arena.lastDecisionMs[0] > 0) decisionMs.push(arena.lastDecisionMs[0]);
    const roundEnd = arena.lastEvents.find((event) => event[0] === "round_end");
    if (roundEnd) {
      const outcome = roundEnd[1] === 0 ? "win"
        : roundEnd[1] === 1 ? "loss" : "double_death";
      return {
        policy, seed, outcome, frames: frame + 1,
        elapsedMs: performance.now() - started,
        decisionP95Ms: percentile(decisionMs, 0.95),
      };
    }
  }
  return {
    policy, seed, outcome: "draw", frames: maxFrames,
    elapsedMs: performance.now() - started,
    decisionP95Ms: percentile(decisionMs, 0.95),
  };
}

async function ensureBlind(policy) {
  checkpoint.blind[policy] ??= {};
  for (let index = 0; index < blindRounds; index += 1) {
    const seed = blindSeed + index;
    if (checkpoint.blind[policy][seed]) continue;
    checkpoint.blind[policy][seed] = playWatchGame(policy, seed);
    await saveCheckpoint();
    if ((index + 1) % 10 === 0 || index + 1 === blindRounds) {
      process.stdout.write(`\rblind ${policy} ${index + 1}/${blindRounds}   `);
    }
  }
  process.stdout.write("\n");
  const games = Object.values(checkpoint.blind[policy]);
  const count = (outcome) => games.filter((row) => row.outcome === outcome).length;
  return {
    policy,
    seed: blindSeed,
    rounds: games.length,
    wins: count("win"),
    losses: count("loss"),
    doubleDeaths: count("double_death"),
    draws: count("draw"),
    winRate: count("win") / Math.max(1, games.length),
    p95DecisionMs: percentile(games.map((row) => row.decisionP95Ms), 0.95),
    games,
  };
}

if (!existsSync(regressionPath)) {
  throw new Error(`missing permanent regression artifact: ${regressionPath}`);
}
const regression = JSON.parse(await readFile(regressionPath, "utf8"));
const championPool = await ensurePool(CHAMPION_BASELINE.policy);
const evaluations = [];
for (const candidate of candidates) {
  const regressionResult = await ensureRegression(candidate, regression);
  if (!regressionResult.passed) {
    evaluations.push({
      candidate,
      regression: {
        passed: false,
        recoveries: regressionResult.recoveries.length,
        regressions: regressionResult.regressions.length,
      },
      paired: null,
      significantPoolGain: false,
      eligibleForBlind: false,
      blind: null,
      promotionRecommended: false,
      rejectedAt: "permanent_regression",
    });
    continue;
  }
  const candidatePool = await ensurePool(candidate);
  const paired = summarizeSelfplayFlywheel({ championReport: championPool, candidateReport: candidatePool });
  const significantPoolGain = paired.recoveries.length >= paired.regressions.length + 2
    && paired.candidateOverall.win >= paired.championOverall.win + 2;
  const eligibleForBlind = regressionResult.passed
    && paired.readyForLaikaBlind
    && significantPoolGain;
  evaluations.push({
    candidate,
    regression: {
      passed: regressionResult.passed,
      recoveries: regressionResult.recoveries.length,
      regressions: regressionResult.regressions.length,
    },
    paired,
    significantPoolGain,
    eligibleForBlind,
    blind: null,
    promotionRecommended: false,
  });
}

for (const evaluation of evaluations.filter((item) => item.eligibleForBlind)) {
  const championBlind = await ensureBlind(CHAMPION_BASELINE.policy);
  const candidateBlind = await ensureBlind(evaluation.candidate);
  const gates = {
    wins: candidateBlind.wins >= championBlind.wins + 3
      && candidateBlind.wins > CHAMPION_BASELINE.laikaBenchmark.wins,
    doubleDeath: candidateBlind.doubleDeaths <= championBlind.doubleDeaths,
    draw: candidateBlind.draws <= championBlind.draws,
    latency: candidateBlind.p95DecisionMs <= championBlind.p95DecisionMs * 1.02
      && candidateBlind.p95DecisionMs <= CHAMPION_BASELINE.promotionGate.maximumP95DecisionMs,
  };
  evaluation.blind = { champion: championBlind, candidate: candidateBlind, gates };
  evaluation.promotionRecommended = Object.values(gates).every(Boolean);
}

const summary = {
  schemaVersion: 1,
  generatedAt: new Date().toISOString(),
  runId,
  frozenChampion: CHAMPION_BASELINE.policy,
  poolSeed,
  blindSeed,
  candidates,
  championPool,
  evaluations,
  promotionRecommended: evaluations.some((item) => item.promotionRecommended),
  deploymentPolicy: "never commit, publish, or deploy unless promotionRecommended is true and full tests pass",
  checkpoint: checkpointPath,
};
await writeFile(summaryPath, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
await writeFile(latestPath, `${JSON.stringify({
  schemaVersion: 1,
  generatedAt: summary.generatedAt,
  runId,
  summary: summaryPath,
  promotionRecommended: summary.promotionRecommended,
}, null, 2)}\n`, "utf8");
console.log(summary.promotionRecommended ? "PROMOTION RECOMMENDED" : "NO PROMOTION; TACTICAL REMAINS FROZEN");
console.log(`wrote ${summaryPath}`);
