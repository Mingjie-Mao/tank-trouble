import * as C from "./killfield-runtime/src/constants.js";
import { KillFieldAgent } from "./killfield-runtime/src/killfield/teacher.js";
import {
  CANDIDATES, LIVE_ACTION_INDICES,
} from "./killfield-runtime/src/killfield/score.js";
import { applyAction, makeSandbox } from "./killfield-runtime/src/killfield/sandbox.js";
import { incomingRisk } from "./killfield-runtime/src/killfield/risk.js";

const NO_FIRE_INDICES = LIVE_ACTION_INDICES.filter((index) => CANDIDATES[index][2] === 0);

function sameAction(left, right) {
  return left[0] === right[0] && left[1] === right[1] && left[2] === right[2];
}

function minimumBulletClearance(game) {
  const me = game.tanks[0];
  let closest = Infinity;
  for (const bullet of game.bullets) {
    if (bullet.removed) continue;
    closest = Math.min(closest, Math.hypot(bullet.x - me.x, bullet.y - me.y));
  }
  return closest / Math.max(game.scale, 1e-6);
}

function visibleBulletRollout(game, action, horizon) {
  const sandbox = makeSandbox(game, "L1", 0);
  const me = sandbox.tanks[0];
  const enemy = sandbox.tanks[1];
  // This verifier answers only whether the bullets already visible on screen
  // kill us. New speculative shots are outside this high-confidence gate.
  enemy.ai = null;
  enemy.fire = false;
  applyAction(sandbox, [action[0], action[1], action[2]]);
  let minClearance = minimumBulletClearance(sandbox);
  for (let frame = 0; frame < horizon; frame += 1) {
    if (frame === 1) me.fire = false;
    sandbox.step();
    minClearance = Math.min(minClearance, minimumBulletClearance(sandbox));
    if (!me.alive) return { survived: false, deathFrame: frame + 1, minClearance };
  }
  return { survived: true, deathFrame: horizon + 1, minClearance };
}

function continueVisibleBulletRollout(game, action, horizon, initialClearance) {
  const sandbox = makeSandbox(game, "L1", 0);
  const me = sandbox.tanks[0];
  const enemy = sandbox.tanks[1];
  enemy.ai = null;
  enemy.fire = false;
  applyAction(sandbox, [action[0], action[1], 0]);
  let minClearance = initialClearance;
  for (let frame = 0; frame < horizon; frame += 1) {
    sandbox.step();
    minClearance = Math.min(minClearance, minimumBulletClearance(sandbox));
    if (!me.alive) return { survived: false, deathFrame: frame + 1, minClearance };
  }
  return { survived: true, deathFrame: horizon + 1, minClearance };
}

function advanceVisibleBullets(game, action, frames) {
  const sandbox = makeSandbox(game, "L1", 0);
  const me = sandbox.tanks[0];
  const enemy = sandbox.tanks[1];
  enemy.ai = null;
  enemy.fire = false;
  applyAction(sandbox, [action[0], action[1], 0]);
  let minClearance = minimumBulletClearance(sandbox);
  for (let frame = 0; frame < frames; frame += 1) {
    sandbox.step();
    minClearance = Math.min(minClearance, minimumBulletClearance(sandbox));
    if (!me.alive) return { sandbox, survived: false, deathFrame: frame + 1, minClearance };
  }
  return { sandbox, survived: true, deathFrame: frames + 1, minClearance };
}

function betterVisible(left, right) {
  if (left.survived !== right.survived) return left.survived;
  if (left.deathFrame !== right.deathFrame) return left.deathFrame > right.deathFrame;
  return left.minClearance > right.minClearance;
}

function visibleBulletTwoStagePlan(game, horizon, splitFrames) {
  let best = null;
  for (const rootIndex of NO_FIRE_INDICES) {
    const root = CANDIDATES[rootIndex];
    const advanced = advanceVisibleBullets(game, root, splitFrames);
    if (!advanced.survived) {
      const candidate = { ...advanced, root: Array.from(root), continuation: Array.from(root) };
      if (best === null || betterVisible(candidate, best)) best = candidate;
      continue;
    }
    for (const continuationIndex of NO_FIRE_INDICES) {
      const continuation = CANDIDATES[continuationIndex];
      const result = continueVisibleBulletRollout(
        advanced.sandbox, continuation, Math.max(1, horizon - splitFrames),
        advanced.minClearance,
      );
      const candidate = {
        ...result,
        deathFrame: result.survived ? horizon + 1 : splitFrames + result.deathFrame,
        root: Array.from(root),
        continuation: Array.from(continuation),
      };
      if (best === null || betterVisible(candidate, best)) best = candidate;
    }
  }
  return best;
}

function advanceNoFire(game, action, frames) {
  const sandbox = makeSandbox(game, "L1", 0);
  const me = sandbox.tanks[0];
  applyAction(sandbox, [action[0], action[1], 0]);
  let minClearance = minimumBulletClearance(sandbox);
  for (let frame = 0; frame < frames; frame += 1) {
    sandbox.step();
    minClearance = Math.min(minClearance, minimumBulletClearance(sandbox));
    if (!me.alive) return { sandbox, survived: false, minClearance, deathFrame: frame + 1 };
    if (sandbox.frozen) break;
  }
  return { sandbox, survived: true, minClearance, deathFrame: frames + 1 };
}

function settleNoFire(game, action, horizon, initialClearance) {
  const sandbox = makeSandbox(game, "L1", 0);
  const me = sandbox.tanks[0];
  applyAction(sandbox, [action[0], action[1], 0]);
  let minClearance = initialClearance;
  let resolved = false;
  for (let frame = 0; frame < horizon; frame += 1) {
    const events = sandbox.step();
    minClearance = Math.min(minClearance, minimumBulletClearance(sandbox));
    if (!me.alive) {
      return { survived: false, resolved: true, deathFrame: frame + 1, minClearance };
    }
    if (sandbox.frozen || events.some((event) => event[0] === "round_end")) {
      resolved = true;
      break;
    }
  }
  return { survived: true, resolved, deathFrame: horizon + 1, minClearance };
}

function betterSettlement(left, right) {
  if (left.survived !== right.survived) return left.survived;
  if (left.resolved !== right.resolved) return left.resolved;
  if (left.deathFrame !== right.deathFrame) return left.deathFrame > right.deathFrame;
  return left.minClearance > right.minClearance;
}

function twoStageSettlementPlan(game, splitFrames) {
  const remaining = Math.max(1, game.endCount - C.NUMBEROFFRAMESFROZEN);
  const rootFrames = Math.min(splitFrames, remaining);
  let best = null;
  for (const rootIndex of NO_FIRE_INDICES) {
    const root = CANDIDATES[rootIndex];
    const advanced = advanceNoFire(game, root, rootFrames);
    if (!advanced.survived) {
      const candidate = {
        ...advanced, root: Array.from(root), continuation: Array.from(root), rootFrames,
      };
      if (best === null || betterSettlement(candidate, best)) best = candidate;
      continue;
    }
    for (const continuationIndex of NO_FIRE_INDICES) {
      const continuation = CANDIDATES[continuationIndex];
      const settled = settleNoFire(
        advanced.sandbox, continuation, Math.max(1, remaining - rootFrames),
        advanced.minClearance,
      );
      const candidate = {
        ...settled,
        root: Array.from(root),
        continuation: Array.from(continuation),
        rootFrames,
      };
      if (best === null || betterSettlement(candidate, best)) best = candidate;
    }
  }
  return best;
}

/**
 * Product-oriented safety layer over the frozen H36 controller.
 *
 * It intervenes only when an exact rollout of already-visible bullets proves
 * the baseline action dies and another action survives. After a kill, it plans
 * the whole deterministic settlement window as two stages to prevent a win
 * from degrading into a double death.
 */
export class TacticalSafetyAgent extends KillFieldAgent {
  constructor(options = {}) {
    super(options);
    this.riskGate = Number(options.riskGate ?? 0.22);
    this.safetyHorizon = Number(options.safetyHorizon ?? 36);
    this.settlementSplit = Number(options.settlementSplit ?? 8);
    this.evasionSplit = Number(options.evasionSplit ?? 4);
    this.tacticalAudits = 0;
    this.tacticalOverrides = 0;
    this.settlementPlans = 0;
    this.settlementPlan = null;
    this.settlementFrame = 0;
    this.auditMs = [];
  }

  emitVerifiedAction(game, action, kind) {
    const emitted = Array.from(action);
    this.lastDecisionKind = kind;
    this.lastAction = emitted;
    if (emitted[0] !== 1 || emitted[1] !== 1) {
      this.lastMotionAction = [emitted[0], emitted[1], 0];
    }
    const tank = game.tanks[0];
    this.effectRound = game.roundNumber;
    this.effectFrame = game.frame;
    this.effectPose = [tank.x, tank.y, tank.rotation];
    this.effectAction = emitted;
    return emitted;
  }

  act(game) {
    const baseline = super.act(game);
    if (!game.tanks[0].alive || game.frozen) return baseline;

    if (!game.tanks[1].alive) {
      if (this.settlementPlan === null) {
        this.settlementPlan = twoStageSettlementPlan(game, this.settlementSplit);
        this.settlementFrame = 0;
        this.settlementPlans += 1;
      }
      const plan = this.settlementPlan;
      if (plan?.survived) {
        const action = this.settlementFrame < plan.rootFrames
          ? plan.root : plan.continuation;
        this.settlementFrame += 1;
        this.commitRemaining = 0;
        return this.emitVerifiedAction(game, action, "settlement_two_stage");
      }
      return baseline;
    }

    const risk = incomingRisk(game, this.boxes);
    if (risk < this.riskGate || game.bullets.length === 0) return baseline;

    const started = performance.now();
    this.tacticalAudits += 1;
    const baselineResult = visibleBulletRollout(game, baseline, this.safetyHorizon);
    if (baselineResult.survived) {
      this.auditMs.push(performance.now() - started);
      return baseline;
    }

    const twoStage = visibleBulletTwoStagePlan(
      game, this.safetyHorizon, this.evasionSplit,
    );
    this.auditMs.push(performance.now() - started);
    if (!twoStage?.survived || sameAction(twoStage.root, baseline)) return baseline;

    this.tacticalOverrides += 1;
    this.commitRemaining = 0;
    return this.emitVerifiedAction(game, twoStage.root, "visible_bullet_two_stage");
  }

  telemetry() {
    const base = super.telemetry();
    const sorted = this.auditMs.slice().sort((left, right) => left - right);
    const p95 = sorted.length ? sorted[Math.floor(0.95 * (sorted.length - 1))] : 0;
    const settlement = this.settlementPlan === null ? null : {
      survived: Boolean(this.settlementPlan.survived),
      resolved: Boolean(this.settlementPlan.resolved),
      root: this.settlementPlan.root ?? null,
      continuation: this.settlementPlan.continuation ?? null,
      rootFrames: this.settlementPlan.rootFrames ?? 0,
      minClearance: this.settlementPlan.minClearance ?? null,
    };
    return {
      ...base,
      tacticalAudits: this.tacticalAudits,
      tacticalOverrides: this.tacticalOverrides,
      settlementPlans: this.settlementPlans,
      settlementPlan: settlement,
      tacticalP95Ms: p95,
    };
  }
}
