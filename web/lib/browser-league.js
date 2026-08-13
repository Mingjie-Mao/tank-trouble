import { performance } from "node:perf_hooks";

import { BrowserArena } from "./browser-arena.js";

export const DEFAULT_LEAGUE_OPPONENTS = Object.freeze([
  "laika-js",
  "hunter-js",
  "dodger-js",
  "random-js",
]);

function percentile(values, q) {
  if (!values.length) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1)))];
}
function classifyRoundEnd(event, candidateSide) {
  if (event[1] === null || event[1] === undefined) return "double_death";
  return event[1] === candidateSide ? "win" : "loss";
}

export function playLeagueGame({
  candidate,
  opponent,
  candidateSide,
  seed,
  maxFrames = 3000,
}) {
  const leftPolicy = candidateSide === 0 ? candidate : opponent;
  const rightPolicy = candidateSide === 1 ? candidate : opponent;
  const arena = new BrowserArena({ seed });
  arena.command({
    action: "mode",
    mode: "selfplay",
    left_policy: leftPolicy,
    right_policy: rightPolicy,
  });
  const decisionMs = [];
  const started = performance.now();
  for (let frame = 0; frame < maxFrames; frame += 1) {
    arena.step(frame * 40);
    decisionMs.push(arena.lastDecisionMs[candidateSide]);
    const roundEnd = arena.lastEvents.find((event) => event[0] === "round_end");
    if (roundEnd) {
      return {
        candidate,
        opponent,
        candidateSide,
        seed,
        outcome: classifyRoundEnd(roundEnd, candidateSide),
        winner: roundEnd[1],
        frames: frame + 1,
        elapsedMs: performance.now() - started,
        candidateDecisionP50Ms: percentile(decisionMs, 0.5),
        candidateDecisionP95Ms: percentile(decisionMs, 0.95),
      };
    }
  }
  return {
    candidate,
    opponent,
    candidateSide,
    seed,
    outcome: "draw",
    winner: null,
    frames: maxFrames,
    elapsedMs: performance.now() - started,
    candidateDecisionP50Ms: percentile(decisionMs, 0.5),
    candidateDecisionP95Ms: percentile(decisionMs, 0.95),
  };
}

function summarizeRows(rows) {
  const counts = { win: 0, loss: 0, double_death: 0, draw: 0 };
  const bySide = {
    left: { games: 0, wins: 0 },
    right: { games: 0, wins: 0 },
  };
  const latencies = [];
  let frames = 0;
  let elapsedMs = 0;
  for (const row of rows) {
    counts[row.outcome] += 1;
    const side = row.candidateSide === 0 ? bySide.left : bySide.right;
    side.games += 1;
    if (row.outcome === "win") side.wins += 1;
    latencies.push(row.candidateDecisionP95Ms);
    frames += row.frames;
    elapsedMs += row.elapsedMs;
  }
  const games = rows.length;
  const leftRate = bySide.left.wins / Math.max(1, bySide.left.games);
  const rightRate = bySide.right.wins / Math.max(1, bySide.right.games);
  return {
    games,
    ...counts,
    winRate: counts.win / Math.max(1, games),
    nonLossRate: (counts.win + counts.double_death + counts.draw) / Math.max(1, games),
    colorGap: Math.abs(leftRate - rightRate),
    bySide: {
      left: { ...bySide.left, winRate: leftRate },
      right: { ...bySide.right, winRate: rightRate },
    },
    averageRoundFrames: frames / Math.max(1, games),
    p95DecisionMs: percentile(latencies, 0.95),
    elapsedMs,
  };
}

export function summarizeLeague(rows, { candidate, roundsPerSide, seed, maxFrames }) {
  const opponentNames = [...new Set(rows.map((row) => row.opponent))];
  const opponents = Object.fromEntries(opponentNames.map((opponent) => [
    opponent,
    summarizeRows(rows.filter((row) => row.opponent === opponent)),
  ]));
  return {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    candidate,
    roundsPerSide,
    seed,
    maxFrames,
    overall: summarizeRows(rows),
    opponents,
    games: rows,
  };
}

export function runBrowserLeague({
  candidate = "p27-js-tactical-v2",
  opponents = DEFAULT_LEAGUE_OPPONENTS,
  roundsPerSide = 10,
  seed = 1100000,
  maxFrames = 3000,
  onGame = null,
} = {}) {
  const rows = [];
  for (let opponentIndex = 0; opponentIndex < opponents.length; opponentIndex += 1) {
    const opponent = opponents[opponentIndex];
    const seedOffset = opponentIndex * 100000;
    for (let index = 0; index < roundsPerSide; index += 1) {
      const roundSeed = seed + seedOffset + index;
      for (const candidateSide of [0, 1]) {
        const row = playLeagueGame({
          candidate,
          opponent,
          candidateSide,
          seed: roundSeed,
          maxFrames,
        });
        rows.push(row);
        onGame?.(row, rows.length, opponents.length * roundsPerSide * 2);
      }
    }
  }
  return summarizeLeague(rows, { candidate, roundsPerSide, seed, maxFrames });
}
