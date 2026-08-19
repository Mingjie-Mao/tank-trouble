import { writeFile } from "node:fs/promises";

import { playWatchGame } from "../lib/browser-league.js";

function argument(name, fallback) {
  const index = process.argv.indexOf(`--${name}`);
  return index === -1 ? fallback : process.argv[index + 1];
}

function percentile(values, q) {
  const sorted = values.slice().sort((left, right) => left - right);
  return sorted[Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1)))] ?? 0;
}

const seed = Number(argument("seed", "3200000"));
const rounds = Number(argument("rounds", "300"));
const maxFrames = Number(argument("max-frames", "3000"));
const output = argument("output", "/tmp/tactical-paired-blind.json");
const policies = argument("policies", "p27-js-tactical-v2,p27-js-tactical-anti-stall")
  .split(",").map((value) => value.trim()).filter(Boolean);
const report = {
  schemaVersion: 1,
  generatedAt: new Date().toISOString(),
  protocol: "paired fresh-seed Laika blind evaluation",
  seed,
  rounds,
  maxFrames,
  policies: {},
};

for (const policy of policies) {
  const games = [];
  for (let index = 0; index < rounds; index += 1) {
    games.push(playWatchGame({ candidate: policy, seed: seed + index, maxFrames }));
    if ((index + 1) % 10 === 0 || index + 1 === rounds) {
      process.stdout.write(`\r${policy} ${index + 1}/${rounds}   `);
    }
  }
  process.stdout.write("\n");
  const count = (outcome) => games.filter((game) => game.outcome === outcome).length;
  report.policies[policy] = {
    policy,
    wins: count("win"),
    losses: count("loss"),
    doubleDeaths: count("double_death"),
    draws: count("draw"),
    winRate: count("win") / Math.max(1, games.length),
    decisionP50Ms: percentile(games.map((game) => game.candidateDecisionP50Ms), 0.5),
    decisionP95Ms: percentile(games.map((game) => game.candidateDecisionP95Ms), 0.95),
    decisionP99Ms: percentile(games.map((game) => game.candidateDecisionP95Ms), 0.99),
    games,
  };
  await writeFile(output, `${JSON.stringify(report, null, 2)}\n`, "utf8");
}

console.log(JSON.stringify(Object.fromEntries(Object.entries(report.policies).map(
  ([policy, result]) => [policy, {
    wins: result.wins,
    losses: result.losses,
    doubleDeaths: result.doubleDeaths,
    draws: result.draws,
    winRate: result.winRate,
    p95DecisionMs: result.decisionP95Ms,
    p99DecisionMs: result.decisionP99Ms,
  }],
)), null, 2));
