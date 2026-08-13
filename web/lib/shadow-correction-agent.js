import { makeExactSandbox } from "./exact-sandbox.js";
import { TwoStageSearchAgent } from "./two-stage-search-agent.js";
import { KillFieldAgent } from "./killfield-runtime/src/killfield/teacher.js";
import { applyAction } from "./killfield-runtime/src/killfield/sandbox.js";

const MOVE_COMMIT = 4;
const TURN_COMMIT = 2;

function sameAction(left, right) {
  return left[0] === right[0] && left[1] === right[1] && left[2] === right[2];
}

function baselineCommit(action) {
  if (action[0] !== 1) return MOVE_COMMIT;
  if (action[1] !== 1) return TURN_COMMIT;
  return 1;
}

function terminalOutcome(game, events, elapsed) {
  const ended = events.find((event) => event[0] === "round_end");
  if (ended) {
    return {
      resolved: true,
      winner: ended[1],
      rank: ended[1] === 0 ? 3 : ended[1] === 1 ? 0 : 1,
      elapsed,
    };
  }
  return null;
}

function rolloutBranch(game, firstAction, firstFrames, options) {
  const sandbox = makeExactSandbox(game);
  const maxFrames = Math.max(firstFrames, options.horizon);
  for (let frame = 0; frame < firstFrames; frame += 1) {
    applyAction(sandbox, frame === 0
      ? firstAction
      : [firstAction[0], firstAction[1], 0]);
    const events = sandbox.step();
    const outcome = terminalOutcome(sandbox, events, frame + 1);
    if (outcome) return outcome;
  }

  const continuation = new KillFieldAgent({
    seed: options.seed,
    horizon: options.searchHorizon,
    oppModel: "L2",
  });
  for (let frame = firstFrames; frame < maxFrames; frame += 1) {
    continuation.drive(sandbox);
    const events = sandbox.step();
    const outcome = terminalOutcome(sandbox, events, frame + 1);
    if (outcome) return outcome;
  }
  // If a death happened near the horizon, finish the engine's settlement
  // window so a returning bullet can still turn it into a double death.
  if (sandbox.aliveCount <= 1) {
    for (let frame = maxFrames; frame < maxFrames + 150; frame += 1) {
      continuation.drive(sandbox);
      const events = sandbox.step();
      const outcome = terminalOutcome(sandbox, events, frame + 1);
      if (outcome) return outcome;
    }
  }
  return { resolved: false, winner: null, rank: 2, elapsed: maxFrames };
}

export function evaluateExactCorrection(game, baseline, proposed, {
  candidateHold = 8,
  horizon = 150,
  searchHorizon = 36,
  seed = 0,
} = {}) {
  const started = performance.now();
  const baselineResult = rolloutBranch(
    game, baseline, baselineCommit(baseline), { horizon, searchHorizon, seed },
  );
  const proposedResult = rolloutBranch(
    game, proposed, candidateHold, { horizon, searchHorizon, seed },
  );

  let label = "unknown";
  if (baselineResult.resolved || proposedResult.resolved) {
    if (proposedResult.rank > baselineResult.rank) label = "beneficial";
    else if (proposedResult.rank < baselineResult.rank) label = "harmful";
    else label = "neutral";
  }
  return {
    label,
    baseline: baselineResult,
    proposed: proposedResult,
    verifyMs: performance.now() - started,
  };
}

function stateFeatures(agent, game, proposal) {
  const me = game.tanks[0];
  const enemy = game.tanks[1];
  const meCell = [Math.floor(me.x / game.scale), Math.floor(me.y / game.scale)];
  const enemyCell = [Math.floor(enemy.x / game.scale), Math.floor(enemy.y / game.scale)];
  const distances = game.distMap(meCell[0], meCell[1]);
  const mazeDistance = distances?.[enemyCell[0]]?.[enemyCell[1]] ?? null;
  return {
    margin: proposal.margin,
    incomingRisk: proposal.risk,
    twoStageGap: proposal.chosenTwoStage - proposal.baselineTwoStage,
    mazeDistance,
    meX: me.x / Math.max(game.scale * game.maze.length, 1),
    meY: me.y / Math.max(game.scale * game.maze[0].length, 1),
    enemyX: enemy.x / Math.max(game.scale * game.maze.length, 1),
    enemyY: enemy.y / Math.max(game.scale * game.maze[0].length, 1),
    headingSin: Math.sin((me.rotation * Math.PI) / 180),
    headingCos: Math.cos((me.rotation * Math.PI) / 180),
    ownBullets: me.bulletsFired,
    enemyBullets: enemy.bulletsFired,
    liveBullets: game.bullets.length,
    fieldValue: agent.field?.valueAt(meCell) ?? 0,
    fieldRelative: agent.field?.relativeSuccessAt(meCell) ?? 0,
    fieldGuidance: agent.field?.guidanceAt(meCell) ?? 0,
    huntChain: agent.chain?.count ?? 0,
    noEffectFrames: agent.noEffectFrames,
  };
}

/**
 * H36 remains the live controller. Two-stage search and the exact continuation
 * run in shadow and may write records, but their proposal is never executed.
 */
export class ShadowCorrectionAgent extends TwoStageSearchAgent {
  constructor(options = {}) {
    super({ ...options, shadowOnly: true });
    this.auditHorizon = Number(options.auditHorizon ?? 150);
    this.auditCooldown = Number(options.auditCooldown ?? 25);
    this.maxAuditsPerRound = Number(options.maxAuditsPerRound ?? 8);
    this.auditSeed = Number(options.auditSeed ?? ((options.seed ?? 0) ^ 0x51f15e));
    this.shadowRecords = [];
    this.lastAuditFrame = -Infinity;
    this.auditRound = null;
    this.roundAudits = 0;
  }

  act(game) {
    const baseline = super.act(game);
    const proposal = this.lastProposal;
    if (this.auditRound !== game.roundNumber) {
      this.auditRound = game.roundNumber;
      this.roundAudits = 0;
      this.lastAuditFrame = -Infinity;
    }
    if (this.lastDecisionKind !== "plan" || proposal === null
        || sameAction(baseline, proposal.chosen)
        || game.frame - this.lastAuditFrame < this.auditCooldown
        || this.roundAudits >= this.maxAuditsPerRound) return baseline;

    const verification = evaluateExactCorrection(
      game, baseline, proposal.chosen,
      {
        candidateHold: this.splitFrames,
        horizon: this.auditHorizon,
        searchHorizon: this.horizon,
        seed: (this.auditSeed ^ game.seed ^ game.frame) >>> 0,
      },
    );
    this.lastAuditFrame = game.frame;
    this.roundAudits += 1;
    this.shadowRecords.push({
      seed: game.seed,
      round: game.roundNumber,
      frame: game.frame,
      baseline: Array.from(baseline),
      proposed: Array.from(proposal.chosen),
      features: stateFeatures(this, game, proposal),
      verification,
    });
    return baseline;
  }

  drainShadowRecords() {
    const records = this.shadowRecords.slice();
    this.shadowRecords.length = 0;
    return records;
  }

  telemetry() {
    return {
      ...super.telemetry(),
      shadowAudits: this.roundAudits,
      shadowLabels: this.shadowRecords.length,
    };
  }
}
