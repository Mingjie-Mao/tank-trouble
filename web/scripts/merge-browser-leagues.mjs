import { readFile, writeFile } from "node:fs/promises";

import { summarizeLeague } from "../lib/browser-league.js";

const [output, ...inputs] = process.argv.slice(2);
if (!output || inputs.length < 2) {
  throw new Error("usage: node merge-browser-leagues.mjs OUTPUT INPUT...");
}

const reports = await Promise.all(inputs.map(async (input) => (
  JSON.parse(await readFile(input, "utf8"))
)));
const candidate = reports[0].candidate;
if (reports.some((report) => report.candidate !== candidate)) {
  throw new Error("all reports must evaluate the same candidate");
}
const games = reports.flatMap((report) => report.games);
const merged = summarizeLeague(games, {
  candidate,
  roundsPerSide: reports.reduce((sum, report) => sum + report.roundsPerSide, 0),
  seed: reports.map((report) => report.seed),
  maxFrames: reports[0].maxFrames,
});
merged.sourceReports = inputs;
await writeFile(output, `${JSON.stringify(merged, null, 2)}\n`, "utf8");
console.log(`merged ${games.length} games into ${output}`);
