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

function createBehaviorTracker(candidateSide) {
  return {
    candidateSide,
    opponentSide: candidateSide === 0 ? 1 : 0,
    candidateFires: 0,
    candidateDeaths: [],
    opponentDeaths: [],
    combatFrames: 0,
    movementSwitches: 0,
    throttleReversals: 0,
    turnReversals: 0,
    previousMovement: null,
  };
}

function observeBehavior(tracker, arena, combatBefore) {
  const side = tracker.candidateSide;
  const opponent = tracker.opponentSide;
  for (const event of arena.lastEvents) {
    if (event[0] === "fire" && event[1] === side) tracker.candidateFires += 1;
    if (event[0] !== "hit") continue;
    const row = { frame: arena.game.frame, owner: event[1], victim: event[2] };
    if (event[2] === side) tracker.candidateDeaths.push(row);
    if (event[2] === opponent) tracker.opponentDeaths.push(row);
  }
  if (!combatBefore) return;
  const agent = side === 0 ? arena.leftAgent : arena.rightAgent;
  const action = agent?.lastAction ?? agent?.action ?? null;
  if (!action) return;
  tracker.combatFrames += 1;
  const movement = [action[0], action[1]];
  const previous = tracker.previousMovement;
  if (previous && (movement[0] !== previous[0] || movement[1] !== previous[1])) {
    tracker.movementSwitches += 1;
  }
  if (previous && ((movement[0] === 0 && previous[0] === 2)
      || (movement[0] === 2 && previous[0] === 0))) tracker.throttleReversals += 1;
  if (previous && ((movement[1] === 0 && previous[1] === 2)
      || (movement[1] === 2 && previous[1] === 0))) tracker.turnReversals += 1;
  tracker.previousMovement = movement;
}

function deathCategory(tracker) {
  const ownDeath = tracker.candidateDeaths.find((row) => (
    row.owner === tracker.candidateSide
  ));
  const activeKill = tracker.opponentDeaths.find((row) => (
    row.owner === tracker.candidateSide
  ));
  if (ownDeath && activeKill && activeKill.frame <= ownDeath.frame) {
    return "post_kill_own_ricochet";
  }
  if (ownDeath) return "own_ricochet";
  const crossfire = tracker.candidateDeaths.find((row) => (
    row.owner === tracker.opponentSide
      && tracker.opponentDeaths.some((other) => (
        other.owner === tracker.candidateSide && other.frame === row.frame
      ))
  ));
  if (crossfire) return "same_frame_crossfire";
  if (tracker.candidateDeaths.length > 0) return "opponent_ricochet";
  return null;
}

function behaviorFields(tracker, arena, outcome) {
  const activeKill = tracker.opponentDeaths.some((row) => (
    row.owner === tracker.candidateSide
  ));
  const passiveKill = tracker.opponentDeaths.some((row) => (
    row.owner === tracker.opponentSide
  ));
  const winType = outcome !== "win" ? null
    : activeKill ? "active_win"
      : passiveKill ? "passive_win" : "unattributed_win";
  const agent = tracker.candidateSide === 0 ? arena.leftAgent : arena.rightAgent;
  const telemetry = agent?.telemetry?.() ?? {};
  return {
    winType,
    activeKill,
    passiveKill,
    deathCategory: deathCategory(tracker),
    candidateFires: tracker.candidateFires,
    zeroFireWin: outcome === "win" && tracker.candidateFires === 0,
    combatFrames: tracker.combatFrames,
    movementSwitches: tracker.movementSwitches,
    movementSwitchesPer1000: 1000 * tracker.movementSwitches
      / Math.max(1, tracker.combatFrames),
    throttleReversals: tracker.throttleReversals,
    turnReversals: tracker.turnReversals,
    reversalsPer1000: 1000 * (tracker.throttleReversals + tracker.turnReversals)
      / Math.max(1, tracker.combatFrames),
    fireOpportunityWindows: telemetry.fireOpportunityWindows ?? 0,
    fireOpportunityCaptures: telemetry.fireOpportunityCaptures ?? 0,
    fireOpportunityCaptureRate: telemetry.fireOpportunityCaptureRate ?? 0,
    plannedOpportunityStarts: telemetry.opportunityPlanStarts ?? 0,
    plannedOpportunityFires: telemetry.opportunityFires ?? 0,
    plannedOpportunityFireRate: (telemetry.opportunityFires ?? 0)
      / Math.max(1, telemetry.opportunityPlanStarts ?? 0),
  };
}

export function playLeagueGame({
  candidate,
  opponent,
  candidateSide,
  seed,
  maxFrames = 3000,
  wallSliding = false,
}) {
  const leftPolicy = candidateSide === 0 ? candidate : opponent;
  const rightPolicy = candidateSide === 1 ? candidate : opponent;
  const arena = new BrowserArena({ seed, wallSliding });
  arena.command({
    action: "mode",
    mode: "selfplay",
    left_policy: leftPolicy,
    right_policy: rightPolicy,
  });
  const decisionMs = [];
  const behavior = createBehaviorTracker(candidateSide);
  const started = performance.now();
  for (let frame = 0; frame < maxFrames; frame += 1) {
    const combatBefore = arena.game.tanks[candidateSide].alive
      && arena.game.tanks[behavior.opponentSide].alive && !arena.game.frozen;
    arena.step(frame * 40);
    observeBehavior(behavior, arena, combatBefore);
    decisionMs.push(arena.lastDecisionMs[candidateSide]);
    const roundEnd = arena.lastEvents.find((event) => event[0] === "round_end");
    if (roundEnd) {
      const outcome = classifyRoundEnd(roundEnd, candidateSide);
      return {
        candidate,
        opponent,
        candidateSide,
        seed,
        outcome,
        winner: roundEnd[1],
        frames: frame + 1,
        elapsedMs: performance.now() - started,
        candidateDecisionP50Ms: percentile(decisionMs, 0.5),
        candidateDecisionP95Ms: percentile(decisionMs, 0.95),
        ...behaviorFields(behavior, arena, outcome),
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
    ...behaviorFields(behavior, arena, "draw"),
  };
}

/** Official Laika protocol: the engine owns Laika and the candidate sees L2. */
export function playWatchGame({ candidate, seed, maxFrames = 3000 }) {
  const arena = new BrowserArena({ seed });
  arena.command({
    action: "mode",
    mode: "watch",
    left_policy: candidate,
    right_policy: "laika-js",
  });
  const decisionMs = [];
  const behavior = createBehaviorTracker(0);
  const started = performance.now();
  for (let frame = 0; frame < maxFrames; frame += 1) {
    const combatBefore = arena.game.tanks[0].alive
      && arena.game.tanks[1].alive && !arena.game.frozen;
    arena.step(frame * 40);
    observeBehavior(behavior, arena, combatBefore);
    decisionMs.push(arena.lastDecisionMs[0]);
    const roundEnd = arena.lastEvents.find((event) => event[0] === "round_end");
    if (!roundEnd) continue;
    const outcome = classifyRoundEnd(roundEnd, 0);
    return {
      candidate,
      opponent: "laika-js",
      candidateSide: 0,
      mode: "watch",
      seed,
      outcome,
      winner: roundEnd[1],
      frames: frame + 1,
      elapsedMs: performance.now() - started,
      candidateDecisionP50Ms: percentile(decisionMs, 0.5),
      candidateDecisionP95Ms: percentile(decisionMs, 0.95),
      ...behaviorFields(behavior, arena, outcome),
    };
  }
  return {
    candidate,
    opponent: "laika-js",
    candidateSide: 0,
    mode: "watch",
    seed,
    outcome: "draw",
    winner: null,
    frames: maxFrames,
    elapsedMs: performance.now() - started,
    candidateDecisionP50Ms: percentile(decisionMs, 0.5),
    candidateDecisionP95Ms: percentile(decisionMs, 0.95),
    ...behaviorFields(behavior, arena, "draw"),
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
  let activeWins = 0;
  let passiveWins = 0;
  let unattributedWins = 0;
  let zeroFireWins = 0;
  let candidateFires = 0;
  let combatFrames = 0;
  let movementSwitches = 0;
  let throttleReversals = 0;
  let turnReversals = 0;
  let fireOpportunityWindows = 0;
  let fireOpportunityCaptures = 0;
  let plannedOpportunityStarts = 0;
  let plannedOpportunityFires = 0;
  for (const row of rows) {
    counts[row.outcome] += 1;
    const side = row.candidateSide === 0 ? bySide.left : bySide.right;
    side.games += 1;
    if (row.outcome === "win") side.wins += 1;
    latencies.push(row.candidateDecisionP95Ms);
    frames += row.frames;
    elapsedMs += row.elapsedMs;
    if (row.winType === "active_win") activeWins += 1;
    else if (row.winType === "passive_win") passiveWins += 1;
    else if (row.winType === "unattributed_win") unattributedWins += 1;
    if (row.zeroFireWin) zeroFireWins += 1;
    candidateFires += row.candidateFires ?? 0;
    combatFrames += row.combatFrames ?? 0;
    movementSwitches += row.movementSwitches ?? 0;
    throttleReversals += row.throttleReversals ?? 0;
    turnReversals += row.turnReversals ?? 0;
    fireOpportunityWindows += row.fireOpportunityWindows ?? 0;
    fireOpportunityCaptures += row.fireOpportunityCaptures ?? 0;
    plannedOpportunityStarts += row.plannedOpportunityStarts ?? 0;
    plannedOpportunityFires += row.plannedOpportunityFires ?? 0;
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
    activeWins,
    passiveWins,
    unattributedWins,
    activeWinRate: activeWins / Math.max(1, games),
    passiveWinRate: passiveWins / Math.max(1, games),
    zeroFireWins,
    candidateFires,
    averageFires: candidateFires / Math.max(1, games),
    movementSwitchesPer1000: 1000 * movementSwitches / Math.max(1, combatFrames),
    reversalsPer1000: 1000 * (throttleReversals + turnReversals)
      / Math.max(1, combatFrames),
    fireOpportunityWindows,
    fireOpportunityCaptures,
    fireOpportunityCaptureRate: fireOpportunityCaptures
      / Math.max(1, fireOpportunityWindows),
    plannedOpportunityStarts,
    plannedOpportunityFires,
    plannedOpportunityFireRate: plannedOpportunityFires
      / Math.max(1, plannedOpportunityStarts),
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
  // K4 physics applies to the whole world, so both tanks share it. Default
  // false keeps every existing league report describing the physics it was
  // measured on.
  wallSliding = false,
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
          wallSliding,
        });
        rows.push(row);
        onGame?.(row, rows.length, opponents.length * roundsPerSide * 2);
      }
    }
  }
  return summarizeLeague(rows, { candidate, roundsPerSide, seed, maxFrames });
}
