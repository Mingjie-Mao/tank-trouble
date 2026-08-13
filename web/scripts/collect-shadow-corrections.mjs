import { performance } from "node:perf_hooks";
import { writeFileSync } from "node:fs";

import { BrowserArena } from "../lib/browser-arena.js";

const rounds = Number(process.argv[2] ?? 20);
const seed = Number(process.argv[3] ?? 980000);
const maxFrames = Number(process.argv[4] ?? 3000);
const outputPath = process.argv[5] ?? "";

const labels = { beneficial: 0, harmful: 0, neutral: 0, unknown: 0 };
const results = { win: 0, loss: 0, double_death: 0, draw: 0 };
const records = [];
let totalPolicyFrames = 0;
let totalVerifyMs = 0;
const started = performance.now();

for (let index = 0; index < rounds; index += 1) {
  const arena = new BrowserArena({ seed: seed + index });
  arena.command({
    action: "mode",
    mode: "watch",
    left_policy: "p27-js-shadow",
    right_policy: "laika-js",
  });
  let result = "draw";
  let frames = 0;
  for (; frames < maxFrames; frames += 1) {
    arena.step(frames * 40);
    const newRecords = arena.leftAgent?.drainShadowRecords?.() ?? [];
    for (const record of newRecords) {
      records.push(record);
      labels[record.verification.label] += 1;
      totalVerifyMs += record.verification.verifyMs;
    }
    const roundEnd = arena.lastEvents.find((event) => event[0] === "round_end");
    if (roundEnd) {
      result = roundEnd[1] === 0 ? "win" : roundEnd[1] === 1 ? "loss" : "double_death";
      frames += 1;
      break;
    }
  }
  results[result] += 1;
  totalPolicyFrames += frames;
  if ((index + 1) % 5 === 0 || index + 1 === rounds) {
    console.log(`  ${index + 1}/${rounds}  H36 wins ${results.win}  shadow labels ${records.length}`);
  }
}

const elapsed = (performance.now() - started) / 1000;
const known = labels.beneficial + labels.harmful + labels.neutral;
const proposedChanges = records.length;
const rate = (value, base) => base ? `${(100 * value / base).toFixed(1)}%` : "0.0%";
console.log(`===== H36 shadow corrections: ${rounds} games @${seed} (${elapsed.toFixed(1)}s) =====`);
console.log(`  live result win ${results.win}/${rounds} (${rate(results.win, rounds)})  loss ${results.loss}  DD ${results.double_death}  draw ${results.draw}`);
console.log(`  proposed ${proposedChanges} / ${totalPolicyFrames} frames (${rate(proposedChanges, totalPolicyFrames)})`);
console.log(`  beneficial ${labels.beneficial}  harmful ${labels.harmful}  neutral ${labels.neutral}  unknown ${labels.unknown}`);
console.log(`  known precision ${rate(labels.beneficial, known)} beneficial, ${rate(labels.harmful, known)} harmful`);
console.log(`  exact verifier mean ${(totalVerifyMs / Math.max(1, records.length)).toFixed(2)} ms/audit`);
if (outputPath) {
  writeFileSync(outputPath, records.map((record) => JSON.stringify(record)).join("\n") + "\n");
  console.log(`  wrote ${records.length} JSONL records to ${outputPath}`);
}
