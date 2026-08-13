import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { DEFAULT_LEAGUE_OPPONENTS, runBrowserLeague } from "../lib/browser-league.js";
import { CHAMPION_BASELINE } from "../lib/champion-baseline.js";
import { summarizeSelfplayFlywheel } from "../lib/selfplay-flywheel.js";

function argument(name, fallback) {
  const index = process.argv.indexOf(`--${name}`);
  return index === -1 ? fallback : process.argv[index + 1];
}

const candidate = argument("candidate", "p27-js-two-stage");
const champion = argument("champion", CHAMPION_BASELINE.policy);
const opponents = argument("opponents", DEFAULT_LEAGUE_OPPONENTS.join(","))
  .split(",").map((value) => value.trim()).filter(Boolean);
const roundsPerSide = Number(argument("rounds-per-side", "10"));
const seed = Number(argument("seed", "2400000"));
const maxFrames = Number(argument("max-frames", "3000"));
const output = argument("output", "data/selfplay-flywheel-latest.json");
const hardCasesOutput = argument("hard-cases", "data/selfplay-hard-cases.json");

function progress(label) {
  return (row, completed, total) => {
    process.stdout.write(`\r${label} ${completed}/${total} ${row.opponent} ${row.outcome}   `);
  };
}

const championReport = runBrowserLeague({
  candidate: champion, opponents, roundsPerSide, seed, maxFrames,
  onGame: progress("champion"),
});
process.stdout.write("\n");
const candidateReport = runBrowserLeague({
  candidate, opponents, roundsPerSide, seed, maxFrames,
  onGame: progress("candidate"),
});
process.stdout.write("\n");
const report = summarizeSelfplayFlywheel({ championReport, candidateReport });

await mkdir(path.dirname(output), { recursive: true });
await mkdir(path.dirname(hardCasesOutput), { recursive: true });
await writeFile(output, `${JSON.stringify({
  ...report, championReport, candidateReport,
}, null, 2)}\n`, "utf8");
await writeFile(hardCasesOutput, `${JSON.stringify({
  schemaVersion: 1,
  generatedAt: report.generatedAt,
  source: output,
  cases: report.hardCases,
}, null, 2)}\n`, "utf8");

console.log(`champion ${(100 * report.championOverall.winRate).toFixed(1)}%`);
console.log(`candidate ${(100 * report.candidateOverall.winRate).toFixed(1)}% (${report.winRateDelta >= 0 ? "+" : ""}${(100 * report.winRateDelta).toFixed(1)} pp)`);
console.log(`recoveries ${report.recoveries.length}, regressions ${report.regressions.length}`);
console.log(`gates ${JSON.stringify(report.gates)}`);
console.log(report.readyForLaikaBlind ? "READY FOR LAIKA BLIND" : "REJECTED BEFORE BLIND");
console.log(`wrote ${output} and ${hardCasesOutput}`);

