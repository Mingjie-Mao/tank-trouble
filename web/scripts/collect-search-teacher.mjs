import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { BrowserArena } from "../lib/browser-arena.js";

function argument(name, fallback) {
  const index = process.argv.indexOf(`--${name}`);
  return index === -1 ? fallback : process.argv[index + 1];
}

const opponents = argument("opponents", "laika-js,hunter-js,dodger-js,random-js")
  .split(",").map((value) => value.trim()).filter(Boolean);
const roundsPerSide = Number(argument("rounds-per-side", "5"));
const seed = Number(argument("seed", "2100000"));
const maxFrames = Number(argument("max-frames", "3000"));
const output = argument("output", "data/search-teacher.json");
const samples = [];
const games = [];

for (let opponentIndex = 0; opponentIndex < opponents.length; opponentIndex += 1) {
  const opponent = opponents[opponentIndex];
  for (let round = 0; round < roundsPerSide; round += 1) {
    const roundSeed = seed + opponentIndex * 100000 + round;
    for (const candidateSide of [0, 1]) {
      const arena = new BrowserArena({ seed: roundSeed });
      arena.command({
        action: "mode",
        mode: "selfplay",
        left_policy: candidateSide === 0 ? "p27-js-tactical-v2-recorder" : opponent,
        right_policy: candidateSide === 1 ? "p27-js-tactical-v2-recorder" : opponent,
      });
      let outcome = "draw";
      let frames = 0;
      for (; frames < maxFrames; frames += 1) {
        arena.step(frames * 40);
        const end = arena.lastEvents.find((event) => event[0] === "round_end");
        if (!end) continue;
        outcome = end[1] === candidateSide ? "win"
          : end[1] === null || end[1] === undefined ? "double_death" : "loss";
        frames += 1;
        break;
      }
      const agent = candidateSide === 0 ? arena.leftAgent : arena.rightAgent;
      for (const sample of agent.searchSamples) {
        samples.push({ ...sample, opponent, candidateSide, seed: roundSeed });
      }
      games.push({ opponent, candidateSide, seed: roundSeed, outcome, frames });
      process.stdout.write(`\r${games.length}/${opponents.length * roundsPerSide * 2} ${opponent} ${outcome} ${samples.length} samples   `);
    }
  }
}

const report = {
  schemaVersion: 1,
  generatedAt: new Date().toISOString(),
  seed,
  roundsPerSide,
  opponents,
  featureCount: samples[0]?.features.length ?? 0,
  samples,
  games,
};
await mkdir(path.dirname(output), { recursive: true });
await writeFile(output, `${JSON.stringify(report)}\n`, "utf8");
process.stdout.write("\n");
console.log(`wrote ${samples.length} samples from ${games.length} games to ${output}`);
