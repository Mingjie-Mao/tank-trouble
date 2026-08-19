import * as C from "./killfield-runtime/src/constants.js";
import { CANDIDATES } from "./killfield-runtime/src/killfield/score.js";
import { applyAction, makeSandbox } from "./killfield-runtime/src/killfield/sandbox.js";
import {
  chaseAction, TacticalV2Agent, topologicalDistance,
} from "./tactical-v2-agent.js";

const NO_FIRE_ACTIONS = CANDIDATES.filter((action) => action[2] === 0);

function minimumBulletClearance(game) {
  const me = game.tanks[0];
  let closest = Infinity;
  for (const bullet of game.bullets) {
    if (bullet.removed) continue;
    closest = Math.min(closest, Math.hypot(bullet.x - me.x, bullet.y - me.y));
  }
  return closest / Math.max(game.scale, 1e-6);
}

/**
 * Exact, visible-state-only check for imminent death from bullets which are
 * already on screen. Future opponent trigger presses are deliberately disabled.
 */
export function visibleBulletSurvival(game, action, horizon = 16) {
  const sandbox = makeSandbox(game, "L1", 0);
  const me = sandbox.tanks[0];
  const enemy = sandbox.tanks[1];
  enemy.ai = null;
  enemy.fire = false;
  applyAction(sandbox, action);
  let minimumClearance = minimumBulletClearance(sandbox);
  for (let frame = 0; frame < horizon; frame += 1) {
    if (frame === 1) me.fire = false;
    sandbox.step();
    minimumClearance = Math.min(minimumClearance, minimumBulletClearance(sandbox));
    if (!me.alive) {
      return {
        survived: false,
        deathFrame: frame + 1,
        minimumClearance,
        enemyAlive: enemy.alive,
      };
    }
  }
  return {
    survived: true,
    deathFrame: horizon + 1,
    minimumClearance,
    enemyAlive: enemy.alive,
  };
}

function betterSurvival(left, right) {
  if (left.result.survived !== right.result.survived) return left.result.survived;
  if (left.result.deathFrame !== right.result.deathFrame) {
    return left.result.deathFrame > right.result.deathFrame;
  }
  return left.result.minimumClearance > right.result.minimumClearance;
}

export function lastChanceVisibleBulletPlan(game, baseline, horizon = 16) {
  if (game.bullets.length === 0) return { intervened: false, reason: "no_bullets" };
  const baselineResult = visibleBulletSurvival(game, baseline, horizon);
  if (baselineResult.survived) {
    return { intervened: false, reason: "baseline_survives", baselineResult };
  }
  let best = null;
  for (const action of NO_FIRE_ACTIONS) {
    const candidate = {
      action: Array.from(action),
      result: visibleBulletSurvival(game, action, horizon),
    };
    if (best === null || betterSurvival(candidate, best)) best = candidate;
  }
  if (!best?.result.survived) {
    return { intervened: false, reason: "no_survivor", baselineResult, best };
  }
  return {
    intervened: true,
    reason: baseline[2] === 1 ? "unsafe_fire" : "unsafe_motion",
    baselineResult,
    best,
  };
}

function applyPredictedMotion(tank, model, frameOffset) {
  const action = model?.predict(frameOffset);
  if (action) {
    const [throttle, turn] = action;
    tank.forward = throttle === 2;
    tank.backup = throttle === 0;
    tank.turnLeft = turn === 0;
    tank.turnRight = turn === 2;
  }
  tank.fire = false;
}

function prepareShotEscapeOpponent(sandbox, opponentModel, opponentBehavior) {
  if (opponentBehavior !== null || opponentModel !== "L2") {
    sandbox.tanks[1].ai = null;
  }
}

function advanceShotEscapeOpponent(enemy, opponentModel, opponentBehavior, frameOffset) {
  if (opponentBehavior !== null || opponentModel !== "L2") {
    applyPredictedMotion(enemy, opponentBehavior, frameOffset);
  }
}

function fixedSettlementSurvival(game) {
  const remaining = Math.max(1, game.endCount - C.NUMBEROFFRAMESFROZEN);
  for (const action of NO_FIRE_ACTIONS) {
    const sandbox = makeSandbox(game, "L1", 0);
    const me = sandbox.tanks[0];
    sandbox.tanks[1].ai = null;
    sandbox.tanks[1].fire = false;
    applyAction(sandbox, action);
    let resolved = false;
    for (let frame = 0; frame < remaining; frame += 1) {
      const events = sandbox.step();
      if (!me.alive) break;
      if (sandbox.frozen || events.some((event) => event[0] === "round_end")) {
        resolved = true;
        break;
      }
    }
    if (me.alive && resolved) return { survived: true, resolved, action };
  }
  return { survived: false, resolved: true, action: null };
}

function betterSettlement(left, right) {
  if (left.survived !== right.survived) return left.survived;
  if (left.resolved !== right.resolved) return left.resolved;
  if (left.deathFrame !== right.deathFrame) return left.deathFrame > right.deathFrame;
  return left.minimumClearance > right.minimumClearance;
}

function advanceSettlement(game, action, frames, initialClearance = Infinity) {
  const sandbox = makeSandbox(game, "L1", 0);
  const me = sandbox.tanks[0];
  sandbox.tanks[1].ai = null;
  sandbox.tanks[1].fire = false;
  applyAction(sandbox, action);
  let minimumClearance = initialClearance;
  let resolved = false;
  for (let frame = 0; frame < frames; frame += 1) {
    const events = sandbox.step();
    minimumClearance = Math.min(minimumClearance, minimumBulletClearance(sandbox));
    if (!me.alive) {
      return {
        sandbox, survived: false, resolved: true, deathFrame: frame + 1,
        minimumClearance,
      };
    }
    if (sandbox.frozen || events.some((event) => event[0] === "round_end")) {
      resolved = true;
      return {
        sandbox, survived: true, resolved, deathFrame: frame + 1, minimumClearance,
      };
    }
  }
  return {
    sandbox, survived: true, resolved, deathFrame: frames + 1, minimumClearance,
  };
}

export function twoStageSettlementSurvival(game, splitFrames = 8) {
  const remaining = Math.max(1, game.endCount - C.NUMBEROFFRAMESFROZEN);
  const rootFrames = Math.min(splitFrames, remaining);
  let best = null;
  for (const root of NO_FIRE_ACTIONS) {
    const advanced = advanceSettlement(game, root, rootFrames);
    if (!advanced.survived || advanced.resolved) {
      const candidate = {
        ...advanced,
        root: Array.from(root),
        continuation: Array.from(root),
        rootFrames,
      };
      if (best === null || betterSettlement(candidate, best)) best = candidate;
      continue;
    }
    for (const continuation of NO_FIRE_ACTIONS) {
      const settled = advanceSettlement(
        advanced.sandbox,
        continuation,
        Math.max(1, remaining - rootFrames),
        advanced.minimumClearance,
      );
      const candidate = {
        ...settled,
        deathFrame: settled.survived
          ? remaining + 1 : rootFrames + settled.deathFrame,
        root: Array.from(root),
        continuation: Array.from(continuation),
        rootFrames,
      };
      if (best === null || betterSettlement(candidate, best)) best = candidate;
    }
  }
  return best ?? {
    survived: false,
    resolved: false,
    root: null,
    continuation: null,
    rootFrames,
  };
}

/** Verify that a shot which kills under visible motion also has a survivable settlement. */
export function auditShotSettlement(game, {
  horizon = 75,
  opponentBehavior = null,
  twoStageSettlement = false,
  settlementSplit = 8,
} = {}) {
  const liveMe = game.tanks[0];
  if (!(liveMe.alive && game.tanks[1].alive && liveMe.triggerReleased
      && game.weaponReady(liveMe))) {
    return { conclusive: false, safe: true, reason: "weapon_not_ready" };
  }
  const sandbox = makeSandbox(game, "L1", 0);
  const me = sandbox.tanks[0];
  const enemy = sandbox.tanks[1];
  enemy.ai = null;
  applyAction(sandbox, [1, 1, 1]);
  for (let frame = 0; frame < horizon; frame += 1) {
    if (frame === 1) me.fire = false;
    applyPredictedMotion(enemy, opponentBehavior, frame + 1);
    const events = sandbox.step();
    const hitEnemy = events.some((event) => (
      event[0] === "hit" && event[1] === me.number && event[2] === enemy.number
    ));
    if (hitEnemy) {
      if (!me.alive) {
        return { conclusive: true, safe: false, reason: "same_frame_double_death" };
      }
      const settlement = twoStageSettlement
        ? twoStageSettlementSurvival(sandbox, settlementSplit)
        : fixedSettlementSurvival(sandbox);
      return {
        conclusive: true,
        safe: Boolean(settlement?.survived && settlement?.resolved),
        reason: settlement?.survived && settlement?.resolved
          ? "settlement_survives" : "unsafe_settlement",
        settlement,
      };
    }
    if (!me.alive || !enemy.alive || sandbox.frozen) break;
  }
  return { conclusive: false, safe: true, reason: "no_predicted_hit" };
}

function betterShotEscape(left, right) {
  if (left.safe !== right.safe) return left.safe;
  if (left.activeHit !== right.activeHit) return left.activeHit;
  if (left.survived !== right.survived) return left.survived;
  if (left.resolved !== right.resolved) return left.resolved;
  if (left.hitFrame !== right.hitFrame) return left.hitFrame < right.hitFrame;
  return left.minimumClearance > right.minimumClearance;
}

function screenFireEscapeRoot(
  game, root, splitFrames, opponentBehavior, opponentModel,
) {
  const sandbox = makeSandbox(game, opponentModel, 0);
  const me = sandbox.tanks[0];
  const enemy = sandbox.tanks[1];
  prepareShotEscapeOpponent(sandbox, opponentModel, opponentBehavior);
  let activeHit = false;
  let minimumClearance = minimumBulletClearance(sandbox);
  for (let frame = 0; frame <= splitFrames; frame += 1) {
    applyAction(sandbox, frame === 0 ? [1, 1, 1] : root);
    if (enemy.alive) {
      advanceShotEscapeOpponent(enemy, opponentModel, opponentBehavior, frame + 1);
    }
    const events = sandbox.step();
    minimumClearance = Math.min(minimumClearance, minimumBulletClearance(sandbox));
    if (events.some((event) => (
      event[0] === "hit" && event[1] === me.number && event[2] === enemy.number
    ))) activeHit = true;
    if (!me.alive || sandbox.frozen) break;
  }
  return { root, survived: me.alive, activeHit, minimumClearance };
}

/**
 * Jointly verify `fire now -> root escape -> continuation escape`.
 *
 * Unlike auditShotSettlement(), movement begins on the frame after firing,
 * rather than after the predicted hit. This is the sequence the live agent
 * will actually execute, so a safe result proves both the shot and its escape.
 */
export function auditShotEscapePlan(game, {
  hitHorizon = 75,
  splitFrames = 8,
  opponentBehavior = null,
  opponentModel = "L1",
  maximumRoots = NO_FIRE_ACTIONS.length,
} = {}) {
  const liveMe = game.tanks[0];
  if (!(liveMe.alive && game.tanks[1].alive && liveMe.triggerReleased
      && game.weaponReady(liveMe))) {
    return { conclusive: false, safe: false, reason: "weapon_not_ready", best: null };
  }
  const screenedRoots = NO_FIRE_ACTIONS.map((root) => (
    screenFireEscapeRoot(
      game, root, splitFrames, opponentBehavior, opponentModel,
    )
  )).sort((left, right) => {
    if (left.survived !== right.survived) return left.survived ? -1 : 1;
    if (left.activeHit !== right.activeHit) return left.activeHit ? -1 : 1;
    return right.minimumClearance - left.minimumClearance;
  }).slice(0, Math.max(1, maximumRoots));
  let best = null;
  for (const screened of screenedRoots) {
    const root = screened.root;
    for (const continuation of NO_FIRE_ACTIONS) {
      const sandbox = makeSandbox(game, opponentModel, 0);
      const me = sandbox.tanks[0];
      const enemy = sandbox.tanks[1];
      prepareShotEscapeOpponent(sandbox, opponentModel, opponentBehavior);
      let activeHit = false;
      let hitFrame = Infinity;
      let resolved = false;
      let winner = null;
      let minimumClearance = minimumBulletClearance(sandbox);
      const maximumFrames = hitHorizon + C.SETTLEMENT_FRAMES + 2;
      for (let frame = 0; frame < maximumFrames; frame += 1) {
        if (frame === 0) {
          applyAction(sandbox, [1, 1, 1]);
        } else {
          const escape = frame - 1 < splitFrames ? root : continuation;
          applyAction(sandbox, escape);
        }
        if (enemy.alive) {
          advanceShotEscapeOpponent(enemy, opponentModel, opponentBehavior, frame + 1);
        }
        const events = sandbox.step();
        minimumClearance = Math.min(minimumClearance, minimumBulletClearance(sandbox));
        if (events.some((event) => (
          event[0] === "hit" && event[1] === me.number && event[2] === enemy.number
        ))) {
          activeHit = true;
          hitFrame = Math.min(hitFrame, frame + 1);
        }
        const roundEnd = events.find((event) => event[0] === "round_end");
        if (roundEnd) {
          resolved = true;
          winner = roundEnd[1];
          break;
        }
        if (!me.alive || sandbox.frozen) break;
        if (!activeHit && frame + 1 >= hitHorizon) break;
      }
      const safe = activeHit && me.alive && resolved && winner === 0;
      const candidate = {
        safe,
        activeHit,
        survived: me.alive,
        resolved,
        winner,
        hitFrame,
        minimumClearance,
        root: Array.from(root),
        continuation: Array.from(continuation),
        rootFrames: splitFrames,
      };
      if (best === null || betterShotEscape(candidate, best)) best = candidate;
    }
  }
  return {
    conclusive: best?.activeHit ?? false,
    safe: best?.safe ?? false,
    reason: best?.safe ? "fire_escape_survives" : "no_safe_fire_escape",
    rootCandidatesTested: screenedRoots.length,
    best,
  };
}

export class TacticalCandidateAgent extends TacticalV2Agent {
  constructor({
    enableShotSettlementAudit = false,
    enableLastChanceSafety = false,
    lastChanceMode = "all",
    lastChanceHorizon = 16,
    enableSafeStallChase = false,
    safeStallFrames = 180,
    safeStallNoEffectFrames = 12,
    safeStallHorizon = 36,
    safeStallMinimumDistance = 7,
    safeStallBurstFrames = 220,
    maximumSafeStallBursts = 1,
    minimumOwnBulletsForAudit = 2,
    maximumUnsafeAuditsPerRound = 4,
    ...options
  } = {}) {
    super(options);
    this.enableShotSettlementAudit = enableShotSettlementAudit;
    this.enableLastChanceSafety = enableLastChanceSafety;
    this.lastChanceMode = lastChanceMode;
    this.lastChanceHorizon = lastChanceHorizon;
    this.enableSafeStallChase = enableSafeStallChase;
    this.safeStallFrames = safeStallFrames;
    this.safeStallNoEffectFrames = safeStallNoEffectFrames;
    this.safeStallHorizon = safeStallHorizon;
    this.safeStallMinimumDistance = safeStallMinimumDistance;
    this.safeStallBurstFrames = safeStallBurstFrames;
    this.maximumSafeStallBursts = maximumSafeStallBursts;
    this.minimumOwnBulletsForAudit = minimumOwnBulletsForAudit;
    this.maximumUnsafeAuditsPerRound = maximumUnsafeAuditsPerRound;
    this.shotSettlementAudits = 0;
    this.unsafeShotsSuppressed = 0;
    this.shotAuditMs = [];
    this.suppressFireRound = null;
    this.suppressFireUntilFrame = -1;
    this.unsafeAuditRound = null;
    this.unsafeAuditsThisRound = 0;
    this.lastChanceAudits = 0;
    this.lastChanceOverrides = 0;
    this.lastChanceFireSuppressions = 0;
    this.lastChanceAuditMs = [];
    this.safeStallChecks = 0;
    this.safeStallOverrides = 0;
    this.safeStallRejected = 0;
    this.safeStallRound = null;
    this.safeStallNoEffectWindow = [];
    this.safeStallBurstRemaining = 0;
    this.safeStallCooldownUntil = -1;
    this.safeStallBursts = 0;
  }

  act(game) {
    const baseline = super.act(game);
    if (this.unsafeAuditRound !== game.roundNumber) {
      this.unsafeAuditRound = game.roundNumber;
      this.unsafeAuditsThisRound = 0;
    }
    if (this.safeStallRound !== game.roundNumber) {
      this.safeStallRound = game.roundNumber;
      this.safeStallNoEffectWindow = [];
      this.safeStallBurstRemaining = 0;
      this.safeStallCooldownUntil = -1;
      this.safeStallBursts = 0;
    }
    this.safeStallNoEffectWindow.push(this.actionNoEffect ? 1 : 0);
    if (this.safeStallNoEffectWindow.length > this.safeStallFrames) {
      this.safeStallNoEffectWindow.shift();
    }
    if (baseline[2] === 1 && this.suppressFireRound === game.roundNumber
        && game.frame <= this.suppressFireUntilFrame) {
      this.unsafeShotsSuppressed += 1;
      this.commitRemaining = 0;
      return this.emitVerifiedAction(game, [1, 1, 0], "unsafe_settlement_cached");
    }
    const stalledFor = game.frame - this.lastProgressFrame;
    const recentNoEffect = this.safeStallNoEffectWindow.reduce((sum, value) => sum + value, 0);
    const currentTopologyDistance = topologicalDistance(game);
    const opponentSaturated = game.tanks[1].bulletsFired >= game.settingsMaxBullets;
    const ownMagazineEmpty = game.tanks[0].bulletsFired === 0;
    const antiStallTrigger = stalledFor >= this.safeStallFrames
      && recentNoEffect >= this.safeStallNoEffectFrames
      && currentTopologyDistance >= this.safeStallMinimumDistance
      && opponentSaturated && ownMagazineEmpty
      && game.frame >= this.safeStallCooldownUntil;
    if (antiStallTrigger && this.safeStallBurstRemaining <= 0
        && this.safeStallBursts < this.maximumSafeStallBursts) {
      this.safeStallBurstRemaining = this.safeStallBurstFrames;
      this.safeStallCooldownUntil = game.frame + 160;
      this.safeStallBursts += 1;
    }
    if (this.enableSafeStallChase && this.safeStallBurstRemaining > 0
        && game.tanks[0].alive && game.tanks[1].alive && !game.frozen) {
      const chase = chaseAction(game);
      this.safeStallChecks += 1;
      const survival = game.bullets.length > 0
        ? visibleBulletSurvival(game, chase, this.safeStallHorizon)
        : { survived: true };
      if (survival.survived) {
        this.safeStallOverrides += 1;
        this.safeStallBurstRemaining -= 1;
        this.commitRemaining = 0;
        return this.emitVerifiedAction(game, chase, "safe_bullet_stall_chase");
      }
      this.safeStallRejected += 1;
      this.safeStallBurstRemaining = 0;
    }
    const lastChanceEligible = this.lastChanceMode === "fire" ? baseline[2] === 1 : true;
    if (this.enableLastChanceSafety && lastChanceEligible && game.bullets.length > 0
        && game.tanks[0].alive && game.tanks[1].alive && !game.frozen) {
      const started = performance.now();
      const lastChance = lastChanceVisibleBulletPlan(
        game, baseline, this.lastChanceHorizon,
      );
      this.lastChanceAuditMs.push(performance.now() - started);
      this.lastChanceAudits += 1;
      if (lastChance.intervened) {
        this.lastChanceOverrides += 1;
        if (baseline[2] === 1) this.lastChanceFireSuppressions += 1;
        this.commitRemaining = 0;
        return this.emitVerifiedAction(
          game,
          lastChance.best.action,
          lastChance.reason === "unsafe_fire"
            ? "last_chance_fire_suppressed" : "last_chance_escape",
        );
      }
    }
    const ownLiveBullets = game.bullets.filter((bullet) => (
      !bullet.removed && bullet.owner === game.tanks[0]
    )).length;
    if (!this.enableShotSettlementAudit || baseline[2] !== 1
        || ownLiveBullets < this.minimumOwnBulletsForAudit
        || this.unsafeAuditsThisRound >= this.maximumUnsafeAuditsPerRound
        || !game.tanks[0].alive || !game.tanks[1].alive || game.frozen) {
      return baseline;
    }
    const started = performance.now();
    const audit = auditShotSettlement(game, {
      opponentBehavior: this.opponentBehavior,
    });
    this.shotAuditMs.push(performance.now() - started);
    this.shotSettlementAudits += 1;
    if (!(audit.conclusive && !audit.safe)) return baseline;
    this.unsafeShotsSuppressed += 1;
    this.unsafeAuditsThisRound += 1;
    this.suppressFireRound = game.roundNumber;
    this.suppressFireUntilFrame = game.frame + 4;
    this.commitRemaining = 0;
    return this.emitVerifiedAction(game, [1, 1, 0], "unsafe_settlement_suppressed");
  }

  telemetry() {
    const sorted = this.shotAuditMs.slice().sort((a, b) => a - b);
    const lastChanceSorted = this.lastChanceAuditMs.slice().sort((a, b) => a - b);
    return {
      ...super.telemetry(),
      shotSettlementAudits: this.shotSettlementAudits,
      unsafeShotsSuppressed: this.unsafeShotsSuppressed,
      shotAuditP95Ms: sorted.length
        ? sorted[Math.floor(0.95 * (sorted.length - 1))] : 0,
      lastChanceAudits: this.lastChanceAudits,
      lastChanceOverrides: this.lastChanceOverrides,
      lastChanceFireSuppressions: this.lastChanceFireSuppressions,
      lastChanceAuditP95Ms: lastChanceSorted.length
        ? lastChanceSorted[Math.floor(0.95 * (lastChanceSorted.length - 1))] : 0,
      safeStallChecks: this.safeStallChecks,
      safeStallOverrides: this.safeStallOverrides,
      safeStallRejected: this.safeStallRejected,
      safeStallBursts: this.safeStallBursts,
    };
  }
}
