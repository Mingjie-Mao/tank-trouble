import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

import { playLeagueGame, playWatchGame } from "../lib/browser-league.js";

function argument(name, fallback) {
  const index = process.argv.indexOf(`--${name}`);
  return index === -1 ? fallback : process.argv[index + 1];
}

const candidate = argument("candidate", "p27-js-tactical-v2");
const seedsPath = argument("seeds", "data/regression/tactical-champion.json");
const output = argument("output", "data/regression/latest.json");
const maxFrames = Number(argument("max-frames", "3000"));
const artifact = JSON.parse(await readFile(seedsPath, "utf8"));

const games = [];
for (const item of artifact.cases) {
  const row = item.mode === "watch"
    ? playWatchGame({ candidate, seed: item.seed, maxFrames })
    : playLeagueGame({
      candidate,
      opponent: item.opponent,
      candidateSide: item.candidateSide,
      seed: item.seed,
      maxFrames,
    });
  games.push({
    ...row,
    baselineOutcome: item.baselineOutcome,
    baselineFrames: item.baselineFrames,
    tags: item.tags,
  });
  process.stdout.write(`\r${games.length}/${artifact.cases.length} ${item.key} ${row.outcome}   `);
}
process.stdout.write("\n");

const regressions = games.filter((row) => (
  row.baselineOutcome === "win" && row.outcome !== "win"
));
const recoveries = games.filter((row) => (
  row.baselineOutcome !== "win" && row.outcome === "win"
));
const doubleDeaths = games.filter((row) => row.outcome === "double_death");
const report = {
  schemaVersion: 1,
  generatedAt: new Date().toISOString(),
  candidate,
  frozenChampion: artifact.champion,
  seeds: seedsPath,
  games,
  counts: {
    games: games.length,
    wins: games.filter((row) => row.outcome === "win").length,
    losses: games.filter((row) => row.outcome === "loss").length,
    doubleDeaths: doubleDeaths.length,
    draws: games.filter((row) => row.outcome === "draw").length,
    recoveries: recoveries.length,
    regressions: regressions.length,
  },
  recoveries,
  regressions,
  doubleDeaths,
  passed: regressions.length === 0,
};
await mkdir(path.dirname(output), { recursive: true });
await writeFile(output, `${JSON.stringify(report, null, 2)}\n`, "utf8");
console.log(`recoveries ${recoveries.length}, regressions ${regressions.length}, double deaths ${doubleDeaths.length}`);
console.log(`${report.passed ? "PASS" : "FAIL"}: ${output}`);
