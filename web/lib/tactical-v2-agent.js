import { getShortestPathWithDistances } from "./killfield-runtime/src/maze.js";
import { CANDIDATES } from "./killfield-runtime/src/killfield/score.js";
import { applyAction, makeSandbox } from "./killfield-runtime/src/killfield/sandbox.js";
import { TacticalSafetyAgent } from "./tactical-safety-agent.js";

const NO_FIRE_ACTIONS = CANDIDATES.filter((action) => action[2] === 0);
const ESCAPE_ROOT_FRAMES = 4;
const ESCAPE_CONTINUATION_FRAMES = 8;
const INTERCEPT_HORIZON = 75;
const INTERCEPT_SCAN_INTERVAL = 12;

function cellOf(game, tank) {
  return [Math.floor(tank.x / game.scale), Math.floor(tank.y / game.scale)];
}

export function topologicalDistance(game) {
  const [mx, my] = cellOf(game, game.tanks[0]);
  const [ex, ey] = cellOf(game, game.tanks[1]);
  return game.distMap(mx, my)?.[ex]?.[ey] ?? Infinity;
}

function signedAngleDelta(target, current) {
  return ((target - current + 540) % 360) - 180;
}

function poseDelta(from, to, scale) {
  return {
    distance: Math.hypot(to.x - from.x, to.y - from.y) / Math.max(scale, 1e-6),
    rotation: Math.abs(signedAngleDelta(to.rotation, from.rotation)) / 180,
  };
}

function freezeOpponent(game) {
  const enemy = game.tanks[1];
  enemy.ai = null;
  enemy.forward = false;
  enemy.backup = false;
  enemy.turnLeft = false;
  enemy.turnRight = false;
  enemy.fire = false;
}

function applyOpponentPrediction(tank, behaviorModel, frameOffset) {
  if (behaviorModel === null) {
    tank.fire = false;
    return;
  }
  const [throttle, turn] = behaviorModel.predict(frameOffset);
  tank.forward = throttle === 2;
  tank.backup = throttle === 0;
  tank.turnLeft = turn === 0;
  tank.turnRight = turn === 2;
  // The model is for motion only. Inventing a future trigger press would add
  // bullets that are not yet visible and turn a fair predictor into a guess.
  tank.fire = false;
}

function advanceLocal(game, action, frames) {
  const sandbox = makeSandbox(game, "L1", 0);
  freezeOpponent(sandbox);
  applyAction(sandbox, action);
  for (let frame = 0; frame < frames; frame += 1) sandbox.step();
  return sandbox;
}

function predictedEnemyAfterOneFrame(game, action, behaviorModel = null) {
  const sandbox = makeSandbox(game, "L1", 0);
  sandbox.tanks[1].ai = null;
  applyOpponentPrediction(sandbox.tanks[1], behaviorModel, 1);
  applyAction(sandbox, action);
  sandbox.step();
  return { x: sandbox.tanks[1].x, y: sandbox.tanks[1].y };
}

/**
 * Exact forward check for a shot against the opponent's currently visible
 * motion. Unlike v1's stationary-target verifier this advances the tank and
 * bullet together through the real collision/reflection code. The opponent's
 * current buttons are merely held constant; no controller state is read.
 */
export function movingInterceptShot(
  game, horizon = INTERCEPT_HORIZON, behaviorModel = null,
) {
  const me = game.tanks[0];
  if (!(me.alive && game.tanks[1].alive && me.triggerReleased
      && game.weaponReady(me))) {
    return { hit: false, survived: me.alive, frame: null, bounces: 0 };
  }
  const sandbox = makeSandbox(game, "L1", 0);
  const simulatedMe = sandbox.tanks[0];
  const simulatedEnemy = sandbox.tanks[1];
  simulatedEnemy.ai = null;
  applyAction(sandbox, [1, 1, 1]);
  let bounces = 0;
  for (let frame = 0; frame < horizon; frame += 1) {
    if (frame === 1) simulatedMe.fire = false;
    applyOpponentPrediction(simulatedEnemy, behaviorModel, frame + 1);
    const events = sandbox.step();
    bounces += events.filter((event) => event[0] === "bounce").length;
    if (events.some((event) => event[0] === "hit"
        && event[1] === simulatedMe.number && event[2] === simulatedEnemy.number)) {
      return { hit: true, survived: simulatedMe.alive, frame: frame + 1, bounces };
    }
    if (!simulatedMe.alive || !simulatedEnemy.alive || sandbox.frozen) break;
  }
  return { hit: false, survived: simulatedMe.alive, frame: null, bounces };
}

function simulateAimIntercept(
  game, firstTurn, firstTurnFrames, secondTurn, secondTurnFrames, horizon,
  behaviorModel,
) {
  const sandbox = makeSandbox(game, "L1", 0);
  const me = sandbox.tanks[0];
  const enemy = sandbox.tanks[1];
  enemy.ai = null;
  let simulatedFrame = 0;
  const advance = () => {
    simulatedFrame += 1;
    applyOpponentPrediction(enemy, behaviorModel, simulatedFrame);
    return sandbox.step();
  };
  if (firstTurnFrames > 0) {
    applyAction(sandbox, [1, firstTurn, 0]);
    for (let frame = 0; frame < firstTurnFrames; frame += 1) {
      advance();
      if (!me.alive || !enemy.alive || sandbox.frozen) {
        return { hit: false, survived: me.alive, frame: null, bounces: 0 };
      }
    }
  }
  if (secondTurnFrames > 0) {
    applyAction(sandbox, [1, secondTurn, 0]);
    for (let frame = 0; frame < secondTurnFrames; frame += 1) {
      advance();
      if (!me.alive || !enemy.alive || sandbox.frozen) {
        return { hit: false, survived: me.alive, frame: null, bounces: 0 };
      }
    }
  }
  applyAction(sandbox, [1, 1, 1]);
  let bounces = 0;
  for (let frame = 0; frame < horizon; frame += 1) {
    if (frame === 1) me.fire = false;
    const events = advance();
    bounces += events.filter((event) => event[0] === "bounce").length;
    if (events.some((event) => event[0] === "hit"
        && event[1] === me.number && event[2] === enemy.number)) {
      return {
        hit: true,
        survived: me.alive,
        frame: firstTurnFrames + secondTurnFrames + frame + 1,
        flightFrame: frame + 1,
        bounces,
        firstTurn,
        firstTurnFrames,
        secondTurn,
        secondTurnFrames,
      };
    }
    if (!me.alive || !enemy.alive || sandbox.frozen) break;
  }
  return { hit: false, survived: me.alive, frame: null, bounces };
}

/** Search every physically distinct heading, including reflected paths. */
export function movingAimInterceptPlan(
  game, horizon = INTERCEPT_HORIZON, behaviorModel = null,
) {
  const me = game.tanks[0];
  if (!(me.alive && game.tanks[1].alive && me.triggerReleased
      && game.weaponReady(me))) return null;
  let best = null;
  const candidates = [{
    firstTurn: 1, firstTurnFrames: 0, secondTurn: 1, secondTurnFrames: 0,
  }];
  for (let frames = 1; frames <= 18; frames += 1) {
    candidates.push({
      firstTurn: 0, firstTurnFrames: frames, secondTurn: 1, secondTurnFrames: 0,
    });
    candidates.push({
      firstTurn: 2, firstTurnFrames: frames, secondTurn: 1, secondTurnFrames: 0,
    });
  }
  // Lead a turning target by first waiting/turning with its visible motion,
  // then counter-turning back to the firing heading. These are generic
  // symmetric trajectories, not an opponent- or seed-specific controller.
  for (const firstTurn of [0, 2]) {
    const secondTurn = firstTurn === 0 ? 2 : 0;
    for (let leadFrames = 4; leadFrames <= 24; leadFrames += 4) {
      for (let counterFrames = 1; counterFrames < leadFrames; counterFrames += 3) {
        candidates.push({
          firstTurn,
          firstTurnFrames: leadFrames,
          secondTurn,
          secondTurnFrames: counterFrames,
        });
      }
    }
  }
  for (const candidate of candidates) {
    const result = simulateAimIntercept(
      game,
      candidate.firstTurn,
      candidate.firstTurnFrames,
      candidate.secondTurn,
      candidate.secondTurnFrames,
      horizon,
      behaviorModel,
    );
    if (!(result.hit && result.survived)) continue;
    // Earliest total kill first; prefer direct fire only as a tie-breaker.
    if (best === null || result.frame < best.frame
        || (result.frame === best.frame && result.bounces < best.bounces)) {
      best = result;
    }
  }
  return best;
}

/**
 * Small field-guided version of the intercept search for ordinary combat.
 * The inverse field supplies the useful heading, so this evaluates a narrow
 * exact neighbourhood instead of scanning every turn/lead combination.
 */
export function fieldGuidedAimCandidates(game, field) {
  const me = game.tanks[0];
  if (!(me.alive && game.tanks[1].alive && me.triggerReleased
      && game.weaponReady(me) && field)) return [];
  const heading = (me.rotation - 90) * (Math.PI / 180);
  const [aim] = field.bestAimAt(cellOf(game, me), heading);
  if (aim === null) return [];
  const targetRotation = ((aim * 180) / Math.PI + 90 + 360) % 360;
  const delta = signedAngleDelta(targetRotation, me.rotation);
  const firstTurn = delta < 0 ? 0 : 2;
  const secondTurn = firstTurn === 0 ? 2 : 0;
  const centre = Math.round(Math.abs(delta) / Math.max(me.turnSpeed, 1e-6));
  const candidates = [{
    firstTurn: 1, firstTurnFrames: 0, secondTurn: 1, secondTurnFrames: 0,
  }];
  const seen = new Set(["1,0,1,0"]);
  const add = (candidate) => {
    const key = `${candidate.firstTurn},${candidate.firstTurnFrames},${candidate.secondTurn},${candidate.secondTurnFrames}`;
    if (seen.has(key)) return;
    seen.add(key);
    candidates.push(candidate);
  };
  for (let offset = -3; offset <= 3; offset += 1) {
    const frames = Math.max(1, Math.min(18, centre + offset));
    add({ firstTurn, firstTurnFrames: frames, secondTurn: 1, secondTurnFrames: 0 });
  }
  // A small symmetric lead/counter neighbourhood handles visible target
  // motion without restoring the full 80+ candidate exhaustive scan.
  for (const lead of [4, 8]) {
    const firstTurnFrames = Math.min(24, Math.max(1, centre + lead));
    for (const counterOffset of [-1, 0, 1]) {
      const secondTurnFrames = Math.max(1, lead + counterOffset);
      add({ firstTurn, firstTurnFrames, secondTurn, secondTurnFrames });
    }
  }
  return candidates;
}

export function evaluateAimInterceptCandidate(
  game, candidate, horizon = INTERCEPT_HORIZON, behaviorModel = null,
) {
  return simulateAimIntercept(
    game,
    candidate.firstTurn,
    candidate.firstTurnFrames,
    candidate.secondTurn,
    candidate.secondTurnFrames,
    horizon,
    behaviorModel,
  );
}

export function fieldGuidedAimInterceptPlan(
  game, field, horizon = INTERCEPT_HORIZON, behaviorModel = null,
) {
  const candidates = fieldGuidedAimCandidates(game, field);
  let best = null;
  for (const candidate of candidates) {
    const result = evaluateAimInterceptCandidate(
      game, candidate, horizon, behaviorModel,
    );
    if (!(result.hit && result.survived)) continue;
    if (best === null || result.frame < best.frame
        || (result.frame === best.frame && result.bounces < best.bounces)) best = result;
  }
  return best;
}

/**
 * Search a tiny two-stage, no-fire manoeuvre when a topology chase wedges the
 * tank against a wall. The score deliberately values an immediate pose change
 * before long-range progress: an escape plan must first prove that its root
 * action can physically move or rotate the real tank.
 */
function localEscapePlan(game) {
  const start = game.tanks[0];
  const enemy = game.tanks[1];
  const startTopology = topologicalDistance(game);
  let best = null;

  for (const root of NO_FIRE_ACTIONS) {
    const rootSandbox = advanceLocal(game, root, ESCAPE_ROOT_FRAMES);
    const rootPose = poseDelta(start, rootSandbox.tanks[0], game.scale);
    for (const continuation of NO_FIRE_ACTIONS) {
      const endSandbox = advanceLocal(
        rootSandbox, continuation, ESCAPE_CONTINUATION_FRAMES,
      );
      const end = endSandbox.tanks[0];
      const endPose = poseDelta(start, end, game.scale);
      const startEnemyDistance = Math.hypot(start.x - enemy.x, start.y - enemy.y);
      const endEnemyDistance = Math.hypot(end.x - enemy.x, end.y - enemy.y);
      const directProgress = (startEnemyDistance - endEnemyDistance)
        / Math.max(game.scale, 1e-6);
      const endTopology = topologicalDistance(endSandbox);
      const topologyProgress = Number.isFinite(startTopology) && Number.isFinite(endTopology)
        ? startTopology - endTopology : 0;
      const score = rootPose.distance * 12
        + rootPose.rotation * 4
        + endPose.distance * 5
        + endPose.rotation
        + directProgress * 2
        + topologyProgress * 8
        - (rootSandbox.tanks[0].hitSomething ? 0.5 : 0)
        - (end.hitSomething ? 0.25 : 0);
      if (best === null || score > best.score) {
        best = {
          root: Array.from(root),
          continuation: Array.from(continuation),
          rootFrames: ESCAPE_ROOT_FRAMES,
          score,
        };
      }
    }
  }
  return best;
}

export function chaseAction(game, targetCell = null) {
  const me = game.tanks[0];
  const enemy = game.tanks[1];
  const [mx, my] = cellOf(game, me);
  const [ex, ey] = targetCell ?? cellOf(game, enemy);
  const distances = game.distMap(mx, my);
  if (distances === null) return [1, 1, 0];
  const path = getShortestPathWithDistances(game.maze, distances, mx, my, ex, ey);
  const next = path[0];
  if (!next) return [1, 1, 0];

  const targetX = (next.x + 0.5) * game.scale;
  const targetY = (next.y + 0.5) * game.scale;
  const desired = (Math.atan2(targetX - me.x, -(targetY - me.y)) * 180) / Math.PI;
  let delta = signedAngleDelta(desired, me.rotation);
  let throttle = 2;
  if (Math.abs(delta) > 100) {
    throttle = 0;
    delta = signedAngleDelta(desired + 180, me.rotation);
  }
  const turn = Math.abs(delta) <= me.turnSpeed * 0.6 ? 1 : delta < 0 ? 0 : 2;
  if (Math.abs(delta) > 55) throttle = 1;
  return [throttle, turn, 0];
}

/**
 * Tactical v2 candidate. It leaves the frozen v1 attack and safety decisions
 * untouched until the visible match has made no topological progress for a
 * sustained window. In that narrow failure state it follows the current
 * shortest maze path toward the opponent. Any bullet, verified firing window,
 * kill, or round reset immediately returns authority to v1.
 */
export class TacticalV2Agent extends TacticalSafetyAgent {
  constructor(options = {}) {
    super(options);
    this.stallFrames = Number(options.stallFrames ?? 90);
    this.chaseBurstFrames = Number(options.chaseBurstFrames ?? 40);
    this.progressRound = null;
    this.bestDistance = Infinity;
    this.lastProgressFrame = 0;
    this.chaseRemaining = 0;
    this.chaseTarget = null;
    this.escapePlan = null;
    this.escapeFrame = 0;
    this.topologyOverrides = 0;
    this.topologyBursts = 0;
    this.escapePlans = 0;
    this.interceptChecks = 0;
    this.interceptFires = 0;
    this.interceptCancelled = 0;
    this.stationaryShotsSuppressed = 0;
    this.interceptPlan = null;
    this.nextInterceptScanFrame = 0;
    this.opponentBehavior = options.opponentBehavior ?? null;
    this.preserveAttackIntentDuringBullets = Boolean(
      options.preserveAttackIntentDuringBullets ?? false,
    );
    this.suspendedAttackFrames = 0;
  }

  resetProgress(game) {
    this.progressRound = game.roundNumber;
    this.bestDistance = topologicalDistance(game);
    this.lastProgressFrame = game.frame;
    this.chaseRemaining = 0;
    this.chaseTarget = null;
    this.escapePlan = null;
    this.escapeFrame = 0;
    this.interceptPlan = null;
    this.nextInterceptScanFrame = game.frame;
  }

  observeProgress(game) {
    if (this.progressRound !== game.roundNumber) this.resetProgress(game);
    const distance = topologicalDistance(game);
    if (distance < this.bestDistance - 1e-6) {
      this.bestDistance = distance;
      this.lastProgressFrame = game.frame;
      this.chaseRemaining = 0;
      this.chaseTarget = null;
    }
    return distance;
  }

  act(game) {
    this.opponentBehavior?.observe(game);
    const baseline = super.act(game);
    if (!game.tanks[0].alive || !game.tanks[1].alive || game.frozen) return baseline;
    const distance = this.observeProgress(game);

    // v1 remains authoritative while any live projectile is already visible.
    if (game.bullets.length > 0) {
      if (!this.preserveAttackIntentDuringBullets) {
        this.chaseRemaining = 0;
        this.chaseTarget = null;
      } else if (this.chaseRemaining > 0 || this.chaseTarget !== null) {
        this.suspendedAttackFrames += 1;
      }
      this.escapePlan = null;
      this.escapeFrame = 0;
      // A partially executed aim sequence cannot be resumed at the old frame,
      // but the higher-level chase target survives and can be reacquired.
      this.interceptPlan = null;
      return baseline;
    }
    if (!Number.isFinite(distance)) return baseline;

    const stalled = game.frame - this.lastProgressFrame >= this.stallFrames;
    const antiEvasionActive = stalled || this.chaseRemaining > 0
      || (distance <= 2 && game.frame - this.lastProgressFrame >= Math.floor(this.stallFrames / 2));
    if (antiEvasionActive) {
      if (this.interceptPlan !== null) {
        const liveEnemy = game.tanks[1];
        const drift = Math.hypot(
          liveEnemy.x - this.interceptPlan.expectedEnemyX,
          liveEnemy.y - this.interceptPlan.expectedEnemyY,
        ) / Math.max(game.scale, 1e-6);
        if (drift > 0.35) {
          this.interceptPlan = null;
          this.interceptCancelled += 1;
          this.nextInterceptScanFrame = game.frame;
        }
      }
      if (this.interceptPlan !== null) {
        if (this.interceptPlan.remaining > 0) {
          this.interceptPlan.remaining -= 1;
          const action = [1, this.interceptPlan.firstTurn, 0];
          const expected = predictedEnemyAfterOneFrame(game, action, this.opponentBehavior);
          this.interceptPlan.expectedEnemyX = expected.x;
          this.interceptPlan.expectedEnemyY = expected.y;
          this.commitRemaining = 0;
          return this.emitVerifiedAction(game, action,
            this.interceptPlan.bounces > 0
              ? "aim_ricochet_intercept" : "aim_moving_intercept");
        }
        if (this.interceptPlan.secondRemaining > 0) {
          this.interceptPlan.secondRemaining -= 1;
          const action = [1, this.interceptPlan.secondTurn, 0];
          const expected = predictedEnemyAfterOneFrame(game, action, this.opponentBehavior);
          this.interceptPlan.expectedEnemyX = expected.x;
          this.interceptPlan.expectedEnemyY = expected.y;
          this.commitRemaining = 0;
          return this.emitVerifiedAction(game, action,
            this.interceptPlan.bounces > 0
              ? "counter_aim_ricochet" : "counter_aim_intercept");
        }
        const verified = movingInterceptShot(game, INTERCEPT_HORIZON, this.opponentBehavior);
        this.interceptPlan = null;
        if (verified.hit && verified.survived) {
          this.interceptFires += 1;
          this.chaseRemaining = 0;
          this.chaseTarget = null;
          this.commitRemaining = 0;
          return this.emitVerifiedAction(game, [1, 1, 1],
            verified.bounces > 0 ? "moving_ricochet_intercept" : "moving_intercept");
        }
      }
      let intercept = null;
      if (game.frame >= this.nextInterceptScanFrame) {
        this.interceptChecks += 1;
        this.nextInterceptScanFrame = game.frame + INTERCEPT_SCAN_INTERVAL;
        intercept = movingAimInterceptPlan(game, INTERCEPT_HORIZON, this.opponentBehavior);
      }
      if (intercept !== null) {
        this.commitRemaining = 0;
        if (intercept.firstTurnFrames === 0 && intercept.secondTurnFrames === 0) {
          this.interceptFires += 1;
          this.chaseRemaining = 0;
          this.chaseTarget = null;
          return this.emitVerifiedAction(game, [1, 1, 1],
            intercept.bounces > 0 ? "moving_ricochet_intercept" : "moving_intercept");
        }
        const firstAction = [1, intercept.firstTurnFrames > 0
          ? intercept.firstTurn : intercept.secondTurn, 0];
        const expected = predictedEnemyAfterOneFrame(
          game, firstAction, this.opponentBehavior,
        );
        this.interceptPlan = {
          firstTurn: intercept.firstTurn,
          remaining: intercept.firstTurnFrames - 1,
          secondTurn: intercept.secondTurn,
          secondRemaining: intercept.secondTurnFrames,
          bounces: intercept.bounces,
          expectedEnemyX: expected.x,
          expectedEnemyY: expected.y,
        };
        if (intercept.firstTurnFrames > 0) {
          return this.emitVerifiedAction(game, [1, intercept.firstTurn, 0],
            intercept.bounces > 0 ? "aim_ricochet_intercept" : "aim_moving_intercept");
        }
        return this.emitVerifiedAction(game, [1, intercept.secondTurn, 0],
          intercept.bounces > 0 ? "counter_aim_ricochet" : "counter_aim_intercept");
      }
      if (this.lastDecisionKind === "forced_fire") {
        this.stationaryShotsSuppressed += 1;
        this.commitRemaining = 0;
      }
    } else if (this.lastDecisionKind === "forced_fire") {
      return baseline;
    }

    if (this.chaseRemaining <= 0 && stalled) {
      this.chaseRemaining = this.chaseBurstFrames;
      this.chaseTarget = cellOf(game, game.tanks[1]);
      this.topologyBursts += 1;
    }
    if (this.chaseRemaining <= 0) return baseline;

    if (this.escapePlan === null && (this.actionNoEffect || game.tanks[0].hitSomething)) {
      this.escapePlan = localEscapePlan(game);
      this.escapeFrame = 0;
      this.escapePlans += 1;
    }
    let action = chaseAction(game, this.chaseTarget);
    if (this.escapePlan !== null) {
      const plan = this.escapePlan;
      action = this.escapeFrame < plan.rootFrames ? plan.root : plan.continuation;
      this.escapeFrame += 1;
      if (this.escapeFrame >= plan.rootFrames + ESCAPE_CONTINUATION_FRAMES) {
        this.escapePlan = null;
        this.escapeFrame = 0;
      }
    }
    this.chaseRemaining -= 1;
    this.topologyOverrides += 1;
    this.commitRemaining = 0;
    return this.emitVerifiedAction(game, action, "topology_chase");
  }

  telemetry() {
    return {
      ...super.telemetry(),
      topologyOverrides: this.topologyOverrides,
      topologyBursts: this.topologyBursts,
      escapePlans: this.escapePlans,
      interceptChecks: this.interceptChecks,
      interceptFires: this.interceptFires,
      interceptCancelled: this.interceptCancelled,
      stationaryShotsSuppressed: this.stationaryShotsSuppressed,
      suspendedAttackFrames: this.suspendedAttackFrames,
      bestTopologicalDistance: this.bestDistance,
      ...(this.opponentBehavior?.telemetry() ?? {}),
    };
  }
}
