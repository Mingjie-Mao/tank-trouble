import { BrowserArena } from "../lib/browser-arena.js";
import { KillFieldAgent } from "../lib/killfield-runtime/src/killfield/teacher.js";
import { DODGER_TIMEOUT_REGRESSION } from "../lib/tactical-v2-regression.js";

function increment(counts, key) {
  counts[key] = (counts[key] ?? 0) + 1;
}

function cellOf(game, tank) {
  return [Math.floor(tank.x / game.scale), Math.floor(tank.y / game.scale)];
}

function mazeDistance(game) {
  const me = game.tanks[0];
  const enemy = game.tanks[1];
  const [mx, my] = cellOf(game, me);
  const [ex, ey] = cellOf(game, enemy);
  const distance = game.distMap(mx, my);
  return distance?.[ex]?.[ey] ?? NaN;
}

function summarize(caseInfo, maxFrames, candidate) {
  const opponent = "dodger-js";
  const leftPolicy = caseInfo.candidateSide === 0 ? candidate : opponent;
  const rightPolicy = caseInfo.candidateSide === 1 ? candidate : opponent;
  const arena = new BrowserArena({ seed: caseInfo.seed });
  arena.command({
    action: "mode",
    mode: "selfplay",
    left_policy: leftPolicy,
    right_policy: rightPolicy,
  });

  const view = caseInfo.candidateSide === 0 ? arena.game : arena.rightView;
  const agent = caseInfo.candidateSide === 0 ? arena.leftAgent : arena.rightAgent;
  const distances = [];
  const euclidean = [];
  const actionCounts = {};
  const decisionCounts = {};
  const cells = new Map();
  let fireEvents = 0;
  let verifiedWindows = 0;
  let ammoSaturatedFrames = 0;
  let stationaryFrames = 0;
  let wallContactFrames = 0;
  let closeFrames = 0;
  let result = "draw";
  let frames = 0;

  for (; frames < maxFrames; frames += 1) {
    const me = view.tanks[0];
    const enemy = view.tanks[1];
    if (KillFieldAgent.verifiedHit(view)) verifiedWindows += 1;
    arena.step(frames * 40);

    const distance = mazeDistance(view);
    if (Number.isFinite(distance)) distances.push(distance);
    euclidean.push(Math.hypot(me.x - enemy.x, me.y - enemy.y) / view.scale);
    if (distance <= 2) closeFrames += 1;
    if (me.bulletsFired >= view.settingsMaxBullets) ammoSaturatedFrames += 1;
    if (me.hitSomething) wallContactFrames += 1;

    const action = agent.lastAction ?? [1, 1, 0];
    increment(actionCounts, action.join(","));
    increment(decisionCounts, agent.lastDecisionKind ?? "unknown");
    if (action[0] === 1 && action[1] === 1) stationaryFrames += 1;

    const cell = cellOf(view, me).join(",");
    cells.set(cell, (cells.get(cell) ?? 0) + 1);
    for (const event of arena.lastEvents) {
      if (event[0] === "fire" && event[1] === caseInfo.candidateSide) fireEvents += 1;
      if (event[0] === "round_end") {
        result = event[1] === caseInfo.candidateSide
          ? "win"
          : event[1] === null || event[1] === undefined
            ? "double_death"
            : "loss";
      }
    }
    if (result !== "draw") {
      frames += 1;
      break;
    }
  }

  const topCells = [...cells.entries()]
    .sort((left, right) => right[1] - left[1])
    .slice(0, 5)
    .map(([cell, visits]) => ({ cell, visits }));
  const average = (values) => values.reduce((sum, value) => sum + value, 0)
    / Math.max(values.length, 1);
  return {
    ...caseInfo,
    result,
    frames,
    maze: `${view.maze.length}x${view.maze[0].length}`,
    mazeDistance: {
      first: distances[0] ?? null,
      last: distances.at(-1) ?? null,
      min: distances.length ? Math.min(...distances) : null,
      max: distances.length ? Math.max(...distances) : null,
      mean: average(distances),
    },
    euclideanMeanCells: average(euclidean),
    closeFrameRate: closeFrames / Math.max(frames, 1),
    verifiedWindows,
    fireEvents,
    ammoSaturatedRate: ammoSaturatedFrames / Math.max(frames, 1),
    stationaryRate: stationaryFrames / Math.max(frames, 1),
    wallContactRate: wallContactFrames / Math.max(frames, 1),
    uniqueCells: cells.size,
    topCells,
    actionCounts,
    decisionCounts,
    noEffectEvents: agent.noEffectEvents ?? 0,
    ownBulletGuardEvents: agent.ownBulletGuardEvents ?? 0,
    tacticalOverrides: agent.tacticalOverrides ?? 0,
  };
}

const maxFrames = Number(process.argv[2] ?? 3000);
const candidate = process.argv[3] ?? "p27-js-tactical";
const diagnosticSeed = process.argv[4] === undefined ? null : Number(process.argv[4]);
const cases = diagnosticSeed === null
  ? DODGER_TIMEOUT_REGRESSION
  : [0, 1].map((candidateSide) => ({ seed: diagnosticSeed, candidateSide }));
const reports = cases.map((caseInfo) => {
  const report = summarize(caseInfo, maxFrames, candidate);
  console.log(JSON.stringify(report));
  return report;
});

const aggregate = {
  cases: reports.length,
  verifiedWindows: reports.reduce((sum, report) => sum + report.verifiedWindows, 0),
  fireEvents: reports.reduce((sum, report) => sum + report.fireEvents, 0),
  meanAmmoSaturatedRate: reports.reduce((sum, report) => sum + report.ammoSaturatedRate, 0)
    / reports.length,
  meanCloseFrameRate: reports.reduce((sum, report) => sum + report.closeFrameRate, 0)
    / reports.length,
  meanStationaryRate: reports.reduce((sum, report) => sum + report.stationaryRate, 0)
    / reports.length,
};
console.log(JSON.stringify({ aggregate }));
