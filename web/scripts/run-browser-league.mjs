import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { DEFAULT_LEAGUE_OPPONENTS, runBrowserLeague } from "../lib/browser-league.js";

function argument(name, fallback) {
  const index = process.argv.indexOf(`--${name}`);
  return index === -1 ? fallback : process.argv[index + 1];
}

const candidate = argument("candidate", "p27-js-tactical-v2");
const opponents = argument("opponents", DEFAULT_LEAGUE_OPPONENTS.join(","))
  .split(",")
  .map((value) => value.trim())
  .filter(Boolean);
const roundsPerSide = Number(argument("rounds-per-side", "10"));
const seed = Number(argument("seed", "1100000"));
const maxFrames = Number(argument("max-frames", "3000"));
const output = argument("output", "public/league-latest.json");

const report = runBrowserLeague({
  candidate,
  opponents,
  roundsPerSide,
  seed,
  maxFrames,
  onGame(row, completed, total) {
    process.stdout.write(
      `\r${completed}/${total} ${row.opponent} ${row.candidateSide === 0 ? "red" : "black"} ${row.outcome}   `,
    );
  },
});

await mkdir(path.dirname(output), { recursive: true });
await writeFile(output, `${JSON.stringify(report, null, 2)}\n`, "utf8");
process.stdout.write("\n");
console.log(`league: ${report.overall.win}/${report.overall.games} wins (${(report.overall.winRate * 100).toFixed(1)}%)`);
for (const [opponent, summary] of Object.entries(report.opponents)) {
  console.log(
    `${opponent}: ${summary.win}/${summary.games} wins (${(summary.winRate * 100).toFixed(1)}%), color gap ${(summary.colorGap * 100).toFixed(1)}%`,
  );
}
console.log(`wrote ${output}`);
