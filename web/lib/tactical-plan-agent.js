import * as C from "./killfield-runtime/src/constants.js";
import {
  applyAction, makeSandbox,
} from "./killfield-runtime/src/killfield/sandbox.js";
import {
  auditShotEscapePlan, TacticalCandidateAgent, visibleBulletSurvival,
} from "./tactical-candidate-agent.js";
import {
  evaluateAimInterceptCandidate, fieldGuidedAimCandidates, movingInterceptShot,
} from "./tactical-v2-agent.js";

const SAFETY_DECISIONS = new Set([
  "visible_bullet_two_stage",
  "visible_bullet_plan_hold",
  "settlement_two_stage",
  "unsafe_settlement_cached",
  "unsafe_settlement_suppressed",
  "last_chance_escape",
  "last_chance_fire_suppressed",
]);
const ORDINARY_DECISIONS = new Set([
  "plan",
  "hold",
  "plan:own_bullet_guard",
  "hold:own_bullet_guard",
]);

function aimFrames(plan) {
  return plan.firstTurnFrames + plan.secondTurnFrames;
}

function liveStateSnapshot(game) {
  const me = game.tanks[0];
  return {
    tank: [me.alive, me.x, me.y, me.rotation],
    bullets: game.bullets.filter((bullet) => !bullet.removed).map((bullet) => ({
      name: bullet.name,
      owner: bullet.owner.number,
      x: bullet.x,
      y: bullet.y,
      xSpeed: bullet.xSpeed,
      ySpeed: bullet.ySpeed,
      lifetime: bullet.lifetime,
      deadly: bullet.deadly,
      hasBounced: bullet.hasBounced,
      justCreated: bullet.justCreated,
    })).sort((left, right) => left.name.localeCompare(right.name)),
  };
}

function closeEnough(left, right, epsilon = 1e-7) {
  return Math.abs(left - right) <= epsilon;
}

function sameLiveState(game, expected) {
  const actual = liveStateSnapshot(game);
  if (actual.tank[0] !== expected.tank[0]
      || !closeEnough(actual.tank[1], expected.tank[1])
      || !closeEnough(actual.tank[2], expected.tank[2])
      || !closeEnough(actual.tank[3], expected.tank[3])
      || actual.bullets.length !== expected.bullets.length) return false;
  for (let index = 0; index < actual.bullets.length; index += 1) {
    const left = actual.bullets[index];
    const right = expected.bullets[index];
    if (left.name !== right.name || left.owner !== right.owner
        || left.lifetime !== right.lifetime || left.deadly !== right.deadly
        || left.hasBounced !== right.hasBounced
        || left.justCreated !== right.justCreated
        || !closeEnough(left.x, right.x) || !closeEnough(left.y, right.y)
        || !closeEnough(left.xSpeed, right.xSpeed)
        || !closeEnough(left.ySpeed, right.ySpeed)) return false;
  }
  return true;
}

function fireEscapeAction(plan, offset = 0) {
  return plan.frame + offset < plan.rootFrames ? plan.root : plan.continuation;
}

function opportunityAction(plan, offset = 0) {
  if (offset < plan.firstRemaining) return [1, plan.firstTurn, 0];
  if (offset < plan.firstRemaining + plan.secondRemaining) {
    return [1, plan.secondTurn, 0];
  }
  return [1, 1, 0];
}

function traceFireEscapeSafety(game, plan, horizon) {
  if (game.bullets.length === 0) return { survived: true, states: [] };
  const sandbox = makeSandbox(game, "L1", 0);
  const me = sandbox.tanks[0];
  sandbox.tanks[1].ai = null;
  sandbox.tanks[1].fire = false;
  const states = [];
  for (let frame = 0; frame < horizon; frame += 1) {
    applyAction(sandbox, fireEscapeAction(plan, frame));
    sandbox.step();
    states.push(liveStateSnapshot(sandbox));
    if (!me.alive) return { survived: false, states };
  }
  return { survived: true, states };
}

function traceOpportunitySafety(game, plan, horizon) {
  if (game.bullets.length === 0) return { survived: true, states: [] };
  const sandbox = makeSandbox(game, "L1", 0);
  const me = sandbox.tanks[0];
  sandbox.tanks[1].ai = null;
  sandbox.tanks[1].fire = false;
  const states = [];
  for (let frame = 0; frame < horizon; frame += 1) {
    applyAction(sandbox, opportunityAction(plan, frame));
    sandbox.step();
    states.push(liveStateSnapshot(sandbox));
    if (!me.alive) return { survived: false, states };
  }
  return { survived: true, states };
}

/**
 * Challenger controller with persistent intent and joint fire/escape plans.
 *
 * The frozen Tactical champion remains untouched. This class is selectable by
 * an explicit hidden policy name until paired regression and blind gates pass.
 */
export class TacticalPlanAgent extends TacticalCandidateAgent {
  constructor({
    opportunityScanInterval = 1,
    opportunityCandidatesPerFrame = 4,
    opportunityHorizon = 75,
    opportunityMaximumAimFrames = 24,
    opportunityMaximumKillFrames = 60,
    fireEscapeSplitFrames = 8,
    fireEscapeMaximumRoots = 3,
    fireEscapeRetryFrames = 4,
    plannedActionSafetyHorizon = 12,
    cachedPlanSafetyHorizon = 36,
    lastChanceHorizon = 16,
    continuityMargin = 0,
    ...options
  } = {}) {
    super({
      enableShotSettlementAudit: true,
      enableLastChanceSafety: true,
      lastChanceHorizon,
      persistEvasionPlan: true,
      preserveAttackIntentDuringBullets: true,
      continuityMargin,
      ...options,
    });
    this.opportunityScanInterval = opportunityScanInterval;
    this.opportunityCandidatesPerFrame = opportunityCandidatesPerFrame;
    this.opportunityHorizon = opportunityHorizon;
    this.opportunityMaximumAimFrames = opportunityMaximumAimFrames;
    this.opportunityMaximumKillFrames = opportunityMaximumKillFrames;
    this.fireEscapeSplitFrames = fireEscapeSplitFrames;
    this.fireEscapeMaximumRoots = fireEscapeMaximumRoots;
    this.fireEscapeRetryFrames = fireEscapeRetryFrames;
    this.plannedActionSafetyHorizon = plannedActionSafetyHorizon;
    this.cachedPlanSafetyHorizon = cachedPlanSafetyHorizon;
    this.opportunityPlan = null;
    this.fireEscapePlan = null;
    this.nextOpportunityScanFrame = 0;
    this.opportunityScans = 0;
    this.opportunityCandidatesEvaluated = 0;
    this.opportunityScanCursor = 0;
    this.opportunityPlanStarts = 0;
    this.opportunityPlanFrames = 0;
    this.opportunityPlanCancels = 0;
    this.opportunityFires = 0;
    this.plannedForcedFires = 0;
    this.fireEscapeFrames = 0;
    this.fireEscapeAudits = 0;
    this.fireEscapeRejected = 0;
    this.planSafetyInterrupts = 0;
    this.fireEscapeSafetyCacheHits = 0;
    this.fireEscapeSafetyRecomputes = 0;
    this.fireEscapeSafetyMismatches = 0;
    this.fastOpportunityAimFrames = 0;
    this.fastFireEscapeFrames = 0;
    this.opportunitySafetyCacheHits = 0;
    this.opportunitySafetyRecomputes = 0;
    this.opportunitySafetyMismatches = 0;
    this.opportunityScanMs = [];
    this.fireEscapeAuditMs = [];
    this.fireEscapeRetryAfter = -1;
  }

  clearOpportunityPlan(cancelled = false) {
    if (cancelled && this.opportunityPlan !== null) this.opportunityPlanCancels += 1;
    this.opportunityPlan = null;
  }

  prepareFastPlanFrame(game) {
    this.opponentBehavior?.observe(game);
    this.observeActionEffect(game);
    this.observedPreviousAction = this.effectAction ?? [1, 1, 0];
    this.observeProgress(game);
    this.observeFireOpportunity(false);
    this.commitRemaining = 0;
  }

  executeFastPlanFrame(game) {
    if (!(game.tanks[0].alive && game.tanks[1].alive && !game.frozen)
        || this.evasionPlan !== null) return null;

    // With no live projectile, a pre-verified aiming step cannot need any of
    // the visible-bullet safety layers. Avoid computing a full H36 baseline
    // which this persistent plan would immediately overwrite.
    if (this.opportunityPlan !== null
        && (this.opportunityPlan.firstRemaining > 0
          || this.opportunityPlan.secondRemaining > 0)) {
      let preverified = game.bullets.length === 0;
      if (!preverified) {
        const states = this.opportunityPlan.safetyStates;
        if (states.length === 0) return null;
        if (!sameLiveState(game, states[0])) {
          this.opportunityPlan.safetyStates = [];
          this.opportunitySafetyMismatches += 1;
          return null;
        }
        states.shift();
        this.opportunitySafetyCacheHits += 1;
        // As with fire escape, consume the final proof state but re-enter the
        // full stack before selecting an action beyond the cached horizon.
        if (states.length === 0) return null;
        preverified = true;
      }
      this.prepareFastPlanFrame(game);
      this.fastOpportunityAimFrames += 1;
      return this.executeOpportunityPlan(game, [1, 1, 0], preverified);
    }

    const plan = this.fireEscapePlan;
    if (plan === null || plan.safetyStates.length === 0
        || plan.frame > plan.expectedHitFrame + 2) return null;
    if (!sameLiveState(game, plan.safetyStates[0])) {
      plan.safetyStates = [];
      this.fireEscapeSafetyMismatches += 1;
      return null;
    }
    plan.safetyStates.shift();
    this.fireEscapeSafetyCacheHits += 1;
    // The consumed final state proves only that we reached the current frame
    // alive. Re-enter the full stack before choosing another action.
    if (plan.safetyStates.length === 0) return null;
    this.prepareFastPlanFrame(game);
    const action = fireEscapeAction(plan);
    plan.frame += 1;
    this.fireEscapeFrames += 1;
    this.fastFireEscapeFrames += 1;
    return this.emitVerifiedAction(game, action, "fire_escape_plan_cached");
  }

  safeFallback(fallback) {
    return [fallback[0], fallback[1], 0];
  }

  plannedActionSurvives(game, action) {
    return game.bullets.length === 0 || visibleBulletSurvival(
      game, action, this.plannedActionSafetyHorizon,
    ).survived;
  }

  startFireEscape(
    game, kind, fallback = [1, 1, 0], preserveFallbackFire = false,
  ) {
    const fallbackAction = preserveFallbackFire
      ? Array.from(fallback) : this.safeFallback(fallback);
    if (game.frame <= this.fireEscapeRetryAfter) {
      this.commitRemaining = 0;
      return this.emitVerifiedAction(
        game, fallbackAction, `${kind}_audit_cached`,
      );
    }
    const started = performance.now();
    const audit = auditShotEscapePlan(game, {
      hitHorizon: this.opportunityHorizon,
      splitFrames: this.fireEscapeSplitFrames,
      opponentBehavior: this.opponentBehavior,
      opponentModel: this.oppModel,
      maximumRoots: this.fireEscapeMaximumRoots,
    });
    this.fireEscapeAuditMs.push(performance.now() - started);
    this.fireEscapeAudits += 1;
    if (!(audit.conclusive && audit.safe && audit.best)) {
      this.fireEscapeRejected += 1;
      this.fireEscapeRetryAfter = game.frame + this.fireEscapeRetryFrames;
      this.clearOpportunityPlan(true);
      this.commitRemaining = 0;
      return this.emitVerifiedAction(
        game, fallbackAction, `${kind}_unsafe`,
      );
    }
    this.fireEscapePlan = {
      root: Array.from(audit.best.root),
      continuation: Array.from(audit.best.continuation),
      rootFrames: audit.best.rootFrames,
      expectedHitFrame: audit.best.hitFrame,
      frame: 0,
      safetyStates: [],
    };
    this.clearOpportunityPlan(false);
    if (kind.startsWith("opportunity_fire")) this.opportunityFires += 1;
    else this.plannedForcedFires += 1;
    this.commitRemaining = 0;
    return this.emitVerifiedAction(game, [1, 1, 1], kind);
  }

  executeFireEscape(game, fallback) {
    if (this.fireEscapePlan === null) return null;
    if (!game.tanks[1].alive || game.frozen) {
      this.fireEscapePlan = null;
      return null;
    }
    const plan = this.fireEscapePlan;
    if (plan.frame > plan.expectedHitFrame + 2) {
      // The target diverged from the visible-motion prediction. Do not keep
      // driving an obsolete escape for the rest of the bullet lifetime.
      this.fireEscapePlan = null;
      return null;
    }
    const action = plan.frame < plan.rootFrames ? plan.root : plan.continuation;
    let cacheMatched = false;
    if (plan.safetyStates.length > 0 && sameLiveState(game, plan.safetyStates[0])) {
      plan.safetyStates.shift();
      cacheMatched = true;
      this.fireEscapeSafetyCacheHits += 1;
    } else if (plan.safetyStates.length > 0) {
      plan.safetyStates = [];
      this.fireEscapeSafetyMismatches += 1;
    }
    let survived = true;
    if (game.bullets.length > 0 && (!cacheMatched || plan.safetyStates.length === 0)) {
      const safety = traceFireEscapeSafety(
        game, plan, this.cachedPlanSafetyHorizon,
      );
      this.fireEscapeSafetyRecomputes += 1;
      survived = safety.survived;
      plan.safetyStates = safety.states;
    }
    if (!survived) {
      this.fireEscapePlan = null;
      this.planSafetyInterrupts += 1;
      this.commitRemaining = 0;
      return this.emitVerifiedAction(
        game, this.safeFallback(fallback), "fire_escape_interrupted",
      );
    }
    plan.frame += 1;
    this.fireEscapeFrames += 1;
    this.commitRemaining = 0;
    return this.emitVerifiedAction(game, action, "fire_escape_plan");
  }

  executeOpportunityPlan(game, fallback, preverified = false) {
    const plan = this.opportunityPlan;
    if (plan === null) return null;
    if (!(game.tanks[0].alive && game.tanks[1].alive && !game.frozen)) {
      this.clearOpportunityPlan(true);
      return null;
    }
    if (plan.firstRemaining > 0) {
      const action = [1, plan.firstTurn, 0];
      if (!preverified && game.bullets.length > 0) {
        const safety = traceOpportunitySafety(
          game, plan, this.cachedPlanSafetyHorizon,
        );
        this.opportunitySafetyRecomputes += 1;
        plan.safetyStates = safety.survived ? safety.states : [];
        preverified = safety.survived;
      }
      if (!preverified && !this.plannedActionSurvives(game, action)) {
        this.clearOpportunityPlan(true);
        this.planSafetyInterrupts += 1;
        this.nextOpportunityScanFrame = game.frame + this.opportunityScanInterval;
        return this.emitVerifiedAction(
          game, this.safeFallback(fallback), "opportunity_aim_interrupted",
        );
      }
      plan.firstRemaining -= 1;
      this.opportunityPlanFrames += 1;
      this.commitRemaining = 0;
      return this.emitVerifiedAction(
        game,
        action,
        plan.bounces > 0 ? "opportunity_aim_ricochet" : "opportunity_aim",
      );
    }
    if (plan.secondRemaining > 0) {
      const action = [1, plan.secondTurn, 0];
      if (!preverified && game.bullets.length > 0) {
        const safety = traceOpportunitySafety(
          game, plan, this.cachedPlanSafetyHorizon,
        );
        this.opportunitySafetyRecomputes += 1;
        plan.safetyStates = safety.survived ? safety.states : [];
        preverified = safety.survived;
      }
      if (!preverified && !this.plannedActionSurvives(game, action)) {
        this.clearOpportunityPlan(true);
        this.planSafetyInterrupts += 1;
        this.nextOpportunityScanFrame = game.frame + this.opportunityScanInterval;
        return this.emitVerifiedAction(
          game, this.safeFallback(fallback), "opportunity_counter_interrupted",
        );
      }
      plan.secondRemaining -= 1;
      this.opportunityPlanFrames += 1;
      this.commitRemaining = 0;
      return this.emitVerifiedAction(
        game,
        action,
        plan.bounces > 0 ? "opportunity_counter_ricochet" : "opportunity_counter",
      );
    }
    const verified = movingInterceptShot(
      game, this.opportunityHorizon, this.opponentBehavior,
    );
    if (!(verified.hit && verified.survived)) {
      this.clearOpportunityPlan(true);
      this.nextOpportunityScanFrame = game.frame + 1;
      return null;
    }
    return this.startFireEscape(
      game,
      verified.bounces > 0 ? "opportunity_fire_ricochet" : "opportunity_fire",
      fallback,
    );
  }

  scanOpportunity(game, fallback) {
    if (game.frame < this.nextOpportunityScanFrame) return null;
    const me = game.tanks[0];
    if (!(me.alive && game.tanks[1].alive && me.triggerReleased
        && game.weaponReady(me))) return null;
    this.nextOpportunityScanFrame = game.frame + this.opportunityScanInterval;
    const started = performance.now();
    const candidates = fieldGuidedAimCandidates(game, this.field);
    let intercept = null;
    const evaluated = Math.min(this.opportunityCandidatesPerFrame, candidates.length);
    for (let offset = 0; offset < evaluated; offset += 1) {
      const index = (this.opportunityScanCursor + offset) % candidates.length;
      const result = evaluateAimInterceptCandidate(
        game, candidates[index], this.opportunityHorizon, this.opponentBehavior,
      );
      if (!(result.hit && result.survived)) continue;
      if (intercept === null || result.frame < intercept.frame
          || (result.frame === intercept.frame && result.bounces < intercept.bounces)) {
        intercept = result;
      }
    }
    if (candidates.length > 0) {
      this.opportunityScanCursor = (this.opportunityScanCursor + evaluated)
        % candidates.length;
    }
    this.opportunityScanMs.push(performance.now() - started);
    this.opportunityScans += 1;
    this.opportunityCandidatesEvaluated += evaluated;
    if (intercept === null
        || aimFrames(intercept) > this.opportunityMaximumAimFrames
        || intercept.frame > this.opportunityMaximumKillFrames) return null;
    this.opportunityPlanStarts += 1;
    if (aimFrames(intercept) === 0) {
      return this.startFireEscape(
        game,
        intercept.bounces > 0 ? "opportunity_fire_ricochet" : "opportunity_fire",
        fallback,
      );
    }
    this.opportunityPlan = {
      firstTurn: intercept.firstTurn,
      firstRemaining: intercept.firstTurnFrames,
      secondTurn: intercept.secondTurn,
      secondRemaining: intercept.secondTurnFrames,
      bounces: intercept.bounces,
      safetyStates: [],
    };
    return this.executeOpportunityPlan(game, fallback);
  }

  act(game) {
    const fastPlanAction = this.executeFastPlanFrame(game);
    if (fastPlanAction !== null) return fastPlanAction;
    const baseline = super.act(game);
    if (!(game.tanks[0].alive && game.tanks[1].alive && !game.frozen)) {
      this.clearOpportunityPlan(false);
      this.fireEscapePlan = null;
      return baseline;
    }
    if (SAFETY_DECISIONS.has(this.lastDecisionKind)) return baseline;

    const escape = this.executeFireEscape(game, baseline);
    if (escape !== null) return escape;

    // A newly verified immediate shot is folded into the same joint
    // fire/escape audit instead of bypassing the persistent plan.
    if (baseline[2] === 1) {
      // The frozen champion remains the fallback. A failed challenger audit
      // must not turn a proven baseline shot into a zero-fire/passive game.
      return this.startFireEscape(game, "planned_forced_fire", baseline, true);
    }

    const planned = this.executeOpportunityPlan(game, baseline);
    if (planned !== null) return planned;

    // Do not compete with an already active Tactical-v2 intercept or topology
    // escape. The opportunity planner owns only ordinary KillField frames.
    if (!ORDINARY_DECISIONS.has(this.lastDecisionKind)) return baseline;
    return this.scanOpportunity(game, baseline) ?? baseline;
  }

  telemetry() {
    const scan = this.opportunityScanMs.slice().sort((a, b) => a - b);
    const audit = this.fireEscapeAuditMs.slice().sort((a, b) => a - b);
    const p95 = (values) => (values.length
      ? values[Math.floor(0.95 * (values.length - 1))] : 0);
    return {
      ...super.telemetry(),
      opportunityPlanActive: this.opportunityPlan !== null,
      fireEscapePlanActive: this.fireEscapePlan !== null,
      opportunityScans: this.opportunityScans,
      opportunityCandidatesEvaluated: this.opportunityCandidatesEvaluated,
      opportunityPlanStarts: this.opportunityPlanStarts,
      opportunityPlanFrames: this.opportunityPlanFrames,
      opportunityPlanCancels: this.opportunityPlanCancels,
      opportunityFires: this.opportunityFires,
      plannedForcedFires: this.plannedForcedFires,
      fireEscapeFrames: this.fireEscapeFrames,
      fireEscapeAudits: this.fireEscapeAudits,
      fireEscapeRejected: this.fireEscapeRejected,
      planSafetyInterrupts: this.planSafetyInterrupts,
      fireEscapeSafetyCacheHits: this.fireEscapeSafetyCacheHits,
      fireEscapeSafetyRecomputes: this.fireEscapeSafetyRecomputes,
      fireEscapeSafetyMismatches: this.fireEscapeSafetyMismatches,
      fastOpportunityAimFrames: this.fastOpportunityAimFrames,
      fastFireEscapeFrames: this.fastFireEscapeFrames,
      opportunitySafetyCacheHits: this.opportunitySafetyCacheHits,
      opportunitySafetyRecomputes: this.opportunitySafetyRecomputes,
      opportunitySafetyMismatches: this.opportunitySafetyMismatches,
      opportunityScanP95Ms: p95(scan),
      fireEscapeAuditP95Ms: p95(audit),
      plannedSettlementFrames: C.SETTLEMENT_FRAMES,
    };
  }
}
