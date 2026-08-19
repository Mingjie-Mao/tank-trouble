import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { BrowserArena } from "../lib/browser-arena.js";

function argument(name, fallback) {
  const index = process.argv.indexOf(`--${name}`);
  return index === -1 ? fallback : process.argv[index + 1];
}

const policy = argument("policy", "p27-js-tactical-v2");
const opponent = argument("opponent", "laika-js");
const mode = argument("mode", "watch");
const candidateSide = Number(argument("side", "0"));
const maxFrames = Number(argument("max-frames", "3000"));
const output = argument("output", "data/failures/latest.json");
const seeds = argument("seeds", "2900019,2900061,2900093,2900134,2900141,2900197,2900207,2900276,2900277")
  .split(",").map(Number).filter(Number.isFinite);

function outcomeOf(event, side) {
  if (event[1] === null || event[1] === undefined) return "double_death";
  return event[1] === side ? "win" : "loss";
}

function classify(candidateDeaths, enemyDeaths) {
  const ownBulletDeath = candidateDeaths.find((row) => row.owner === row.victim);
  if (ownBulletDeath && enemyDeaths.length > 0) return "post_kill_own_ricochet";
  if (ownBulletDeath) return "own_ricochet";
  const sameFrame = candidateDeaths.find((row) => enemyDeaths.some((enemy) => enemy.frame === row.frame));
  if (sameFrame) return "same_frame_crossfire";
  if (candidateDeaths.length > 0) return "opponent_ricochet";
  return "timeout_or_unresolved";
}

const games = [];
for (const seed of seeds) {
  const arena = new BrowserArena({ seed });
  const leftPolicy = candidateSide === 0 ? policy : opponent;
  const rightPolicy = candidateSide === 1 ? policy : opponent;
  arena.command({
    action: "mode",
    mode,
    left_policy: leftPolicy,
    right_policy: rightPolicy,
  });
  const candidateDeaths = [];
  const enemyDeaths = [];
  const fires = [];
  let outcome = "draw";
  let roundFrames = maxFrames;
  for (let frame = 0; frame < maxFrames; frame += 1) {
    const view = candidateSide === 0 ? arena.game : arena.rightView;
    const agent = candidateSide === 0 ? arena.leftAgent : arena.rightAgent;
    const bulletsBefore = view.bullets.map((bullet) => ({
      owner: bullet.owner.number,
      bounced: bullet.hasBounced,
      lifetime: bullet.lifetime,
    }));
    arena.step(frame * 40);
    for (const event of arena.lastEvents) {
      if (event[0] === "fire" && event[1] === candidateSide) {
        fires.push({ frame: arena.game.frame, decision: agent?.lastDecisionKind ?? "unknown" });
      }
      if (event[0] === "hit") {
        const row = {
          frame: arena.game.frame,
          owner: event[1] === candidateSide ? 0 : 1,
          victim: event[2] === candidateSide ? 0 : 1,
          decision: agent?.lastDecisionKind ?? "unknown",
          bulletsBefore,
        };
        if (event[2] === candidateSide) candidateDeaths.push(row);
        else enemyDeaths.push(row);
      }
      if (event[0] === "round_end") {
        outcome = outcomeOf(event, candidateSide);
        roundFrames = frame + 1;
      }
    }
    if (outcome !== "draw") break;
  }
  const agent = candidateSide === 0 ? arena.leftAgent : arena.rightAgent;
  games.push({
    seed,
    policy,
    opponent,
    mode,
    candidateSide,
    outcome,
    frames: roundFrames,
    category: classify(candidateDeaths, enemyDeaths),
    candidateDeaths,
    enemyDeaths,
    ownFireCount: fires.length,
    lastOwnFire: fires.at(-1) ?? null,
    telemetry: agent?.telemetry?.() ?? null,
  });
}

const categoryCounts = {};
for (const game of games) categoryCounts[game.category] = (categoryCounts[game.category] ?? 0) + 1;
const report = {
  schemaVersion: 1,
  generatedAt: new Date().toISOString(),
  visibleStateOnly: true,
  policy,
  opponent,
  mode,
  candidateSide,
  categoryCounts,
  games,
};
await mkdir(path.dirname(output), { recursive: true });
await writeFile(output, `${JSON.stringify(report, null, 2)}\n`, "utf8");
console.log(JSON.stringify({ output, categoryCounts }, null, 2));
