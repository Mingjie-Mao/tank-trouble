/**
 * The search agent.
 *
 * A receding-horizon controller wrapped in a small amount of hand-written
 * machinery that exists because pure per-frame replanning plays badly:
 *
 *   - Commitment. A chosen move is held for a few frames, so the tank drives
 *     in a line instead of dithering between near-tied candidates.
 *   - Forced fire. If the engine's own ballistics says the current heading
 *     connects, take the shot immediately and override everything else. There
 *     is no confidence threshold — a verified hit is a hit.
 *   - Own-bullet guard. Plans can predate a bullet we just fired, so any
 *     movement that would drive into our own shot is replaced.
 *   - Stuck detection. If a commanded move produced no motion at all, the
 *     whole (throttle, turn) pair is penalised so we stop grinding a wall.
 *
 * The agent always drives tank 0.
 */

import * as C from "../constants.js";
import { LaikaAI } from "../laika.js";
import { HuntChainState } from "./chain.js";
import {
  InverseDensityFieldBuilder, DEFAULT_RAYS, DEFAULT_BOUNCES, DEFAULT_FLIGHT_FRAMES,
} from "./field.js";
import {
  CANDIDATES, LIVE_ACTION_INDICES, ROLLOUT_PLANS, STATIONARY_FIRE_ACTION,
  MPC_HORIZON, MPC_HOLD,
  COMMIT_MOVE_FRAMES, COMMIT_TURN_FRAMES, OWN_BULLET_GUARD_HORIZON,
  NO_EFFECT_REPEAT_PENALTY, densityRollout, postKillSurvivalScores,
  actionSelfHits, maskMovingFireScores, actionIndex, argmax,
} from "./score.js";
import { Rng } from "../rng.js";

function cellOf(game, tank) {
  return [Math.floor(tank.x / game.scale), Math.floor(tank.y / game.scale)];
}

export class KillFieldAgent {
  constructor({
    seed = 0,
    rayCount = DEFAULT_RAYS,
    maxBounces = DEFAULT_BOUNCES,
    maxFlightFrames = DEFAULT_FLIGHT_FRAMES,
    horizon = MPC_HORIZON,
    hold = MPC_HOLD,
    oppModel = "L2",
    continuityMargin = 0,
    commitMoveFrames = COMMIT_MOVE_FRAMES,
    wallContactReplan = false,
    fireContinuation = false,
  } = {}) {
    this.rng = new Rng(seed);
    this.rayCount = rayCount;
    this.maxBounces = maxBounces;
    this.maxFlightFrames = maxFlightFrames;
    this.horizon = horizon;
    this.hold = hold;
    // How many frames a chosen translation is held before the search runs
    // again. Commitment is also broken early by `hitSomething`, which fires on
    // ~22% of frames under the original collision model but only ~3% under K4
    // wall sliding — so a world with K4 replans far less often at the same
    // value here, and this becomes the knob that restores planning cadence.
    this.commitMoveFrames = Number(commitMoveFrames);
    // Restore the selective replan trigger K4 removed. Under the original
    // collision model `hitSomething` fired on ~23% of frames and broke
    // commitment whenever a wall altered the motion; under K4 it fires on
    // ~3%, because most contacts now resolve as slides. `wallSliding` marks
    // exactly those resolved contacts, so it carries the signal the planner
    // lost — but it is denser than the original (~32%), so it is a
    // replacement to be measured, not a like-for-like restoration.
    this.wallContactReplan = Boolean(wallContactReplan);
    // K1. Search 18 plans instead of 10 actions and stop forcing verified
    // shots, so firing is scored against its own follow-up movement.
    this.fireContinuation = Boolean(fireContinuation);
    // Optional, safety-preserving tie-break for challengers. The frozen
    // KillField/Tactical policies leave this at zero. When enabled, a current
    // no-fire movement may be retained only when it is already within this
    // many score points of the freshly searched optimum.
    this.continuityMargin = Number(continuityMargin);
    // Which controller the lookahead sandbox assumes tank 1 is running.
    // "L2" plays out the real Laika script — only sound when the opponent
    // actually is Laika. "L1" just freezes their current buttons, which is
    // the honest assumption against a human: we cannot script their play,
    // so pretending we can (and imagining them dying on schedule) is how the
    // agent ends up standing still against a live opponent who never died.
    this.oppModel = oppModel;
    this.reset();
  }

  reset() {
    this.game = null;
    this.roundNumber = null;
    this.builder = null;
    this.fieldCache = new Map();
    this.field = null;
    this.boxes = [];

    this.commitRemaining = 0;
    this.committedAction = [1, 1, 0];
    this.lastMotionAction = [1, 1, 0];
    this.lastAction = [1, 1, 0];
    this.lastDecisionKind = "none";

    this.chain = new HuntChainState();
    this.chainTotal = 0.0;
    this.lastChainGain = 0.0;
    this.chainRound = null;
    this.chainTarget = null;
    this.chainCell = null;

    this.actionNoEffect = false;
    this.noEffectFrames = 0;
    this.observedPreviousAction = [1, 1, 0];
    this.effectRound = null;
    this.effectFrame = null;
    this.effectPose = null;
    this.effectAction = null;

    // Telemetry
    this.fieldBuilds = 0;
    this.fieldBuildMs = 0.0;
    this.ownBulletGuardEvents = 0;
    this.noEffectEvents = 0;
    this.planMs = [];
    this.continuityHolds = 0;
    this.currentFireOpportunity = false;
    this.fireOpportunityWindows = 0;
    this.fireOpportunityCaptures = 0;
    this.fireOpportunityFrames = 0;
    this.missedFireOpportunityFrames = 0;
    this.executedCombatFrames = 0;
    this.movementSwitches = 0;
    this.throttleReversals = 0;
    this.turnReversals = 0;
    this.previousExecutedMovement = null;
  }

  observeFireOpportunity(available) {
    const next = Boolean(available);
    if (next && !this.currentFireOpportunity) this.fireOpportunityWindows += 1;
    this.currentFireOpportunity = next;
  }

  observeExecutedAction(game, action) {
    if (this.currentFireOpportunity) {
      this.fireOpportunityFrames += 1;
      if (action[2] === 1) {
        this.fireOpportunityCaptures += 1;
        // A held trigger cannot capture the same window twice.
        this.currentFireOpportunity = false;
      } else {
        this.missedFireOpportunityFrames += 1;
      }
    }
    if (!(game.tanks[0].alive && game.tanks[1].alive && !game.frozen)) return;
    this.executedCombatFrames += 1;
    const movement = [action[0], action[1]];
    const previous = this.previousExecutedMovement;
    if (previous !== null
        && (movement[0] !== previous[0] || movement[1] !== previous[1])) {
      this.movementSwitches += 1;
    }
    if (previous !== null
        && ((movement[0] === 0 && previous[0] === 2)
          || (movement[0] === 2 && previous[0] === 0))) {
      this.throttleReversals += 1;
    }
    if (previous !== null
        && ((movement[1] === 0 && previous[1] === 2)
          || (movement[1] === 2 && previous[1] === 0))) {
      this.turnReversals += 1;
    }
    this.previousExecutedMovement = movement;
  }

  /** Did the last command actually move us? Detects grinding against a wall. */
  observeActionEffect(game) {
    const tank = game.tanks[0];
    if (this.effectRound !== game.roundNumber) {
      this.actionNoEffect = false;
      this.noEffectFrames = 0;
      this.effectRound = game.roundNumber;
      this.effectFrame = game.frame;
      this.effectPose = [tank.x, tank.y, tank.rotation];
      this.effectAction = null;
      return;
    }
    if (this.effectFrame === null || game.frame === this.effectFrame) return;
    const previous = this.effectPose;
    const action = this.effectAction;
    if (previous === null || action === null) return;

    const displacement = Math.hypot(tank.x - previous[0], tank.y - previous[1]);
    const rotationDelta = Math.abs(
      ((tank.rotation - previous[2] + 180.0) % 360.0 + 360.0) % 360.0 - 180.0,
    );
    const requestedTranslation = action[0] !== 1;
    const requestedTurn = action[1] !== 1;
    const moved = displacement > Math.max(1e-4, game.scale * 1e-4);
    const turned = rotationDelta > 1e-3;
    this.actionNoEffect = (requestedTranslation || requestedTurn) && !moved && !turned;
    this.noEffectFrames = this.actionNoEffect ? this.noEffectFrames + 1 : 0;
    if (this.actionNoEffect) this.noEffectEvents += 1;
  }

  emitAction(game, action, kind) {
    if (action[2] && !(action[0] === 1 && action[1] === 1)) action = [1, 1, 1];
    if (action[2] === 0 && !(action[0] === 1 && action[1] === 1)
        && actionSelfHits(game, action)) {
      const safety = postKillSurvivalScores(game, OWN_BULLET_GUARD_HORIZON);
      const picked = CANDIDATES[argmax(safety)];
      action = [picked[0], picked[1], 0];
      this.commitRemaining = 0;
      this.committedAction = action;
      this.ownBulletGuardEvents += 1;
      kind = `${kind}:own_bullet_guard`;
    }
    this.lastDecisionKind = kind;
    this.lastAction = action;
    if (action[0] !== 1 || action[1] !== 1) {
      this.lastMotionAction = [action[0], action[1], 0];
    }
    const tank = game.tanks[0];
    this.effectRound = game.roundNumber;
    this.effectFrame = game.frame;
    this.effectPose = [tank.x, tank.y, tank.rotation];
    this.effectAction = action;
    return action;
  }

  /** Fields are cached per enemy cell; a new round throws the cache away. */
  ensureField(game) {
    if (game !== this.game || game.roundNumber !== this.roundNumber) {
      this.game = game;
      this.roundNumber = game.roundNumber;
      this.builder = new InverseDensityFieldBuilder(
        game, this.rayCount, this.maxBounces, this.maxFlightFrames,
      );
      this.boxes = this.builder.boxes;
      this.fieldCache = new Map();
      this.commitRemaining = 0;
    }
    const target = cellOf(game, game.tanks[1]);
    const key = `${target[0]},${target[1]}`;
    if (!this.fieldCache.has(key)) {
      const started = performance.now();
      this.fieldCache.set(key, this.builder.build(target));
      this.fieldBuildMs += performance.now() - started;
      this.fieldBuilds += 1;
      this.commitRemaining = 0;
    }
    this.field = this.fieldCache.get(key);
    return this.field;
  }

  updateLiveChain(game, field) {
    const currentCell = cellOf(game, game.tanks[0]);
    const target = field.targetCell;
    if (this.chainRound !== game.roundNumber) {
      this.chain = new HuntChainState();
      this.chainRound = game.roundNumber;
      this.chainTarget = target;
      this.chainCell = currentCell;
      this.lastChainGain = 0.0;
      return;
    }
    this.chain.advance();
    const stable = this.chainTarget !== null
      && target[0] === this.chainTarget[0] && target[1] === this.chainTarget[1];
    const gain = this.chain.collectAscent(field, this.chainCell, currentCell, stable);
    this.lastChainGain = gain;
    this.chainTotal += gain;
    this.chainTarget = target;
    this.chainCell = currentCell;
  }

  /** The engine's own ballistics simulator is the sole firing authority. */
  static verifiedHit(game) {
    const me = game.tanks[0];
    if (!(me.alive && game.tanks[1].alive && me.triggerReleased
        && game.weaponReady(me))) return false;
    return new LaikaAI(game, me).checkBulletPath(me.rotation).result === "HIT";
  }

  scores(game) {
    const field = this.ensureField(game);
    const seed = this.rng.randrange(1 << 30);
    const values = new Float64Array(CANDIDATES.length);
    this.bestFireContinuation = null;
    if (this.fireContinuation) {
      // K1. Eighteen plans collapsing onto the same ten real first actions:
      // nine persistent no-fire controls, plus nine "fire this frame, then
      // move" continuations that all share the stationary-fire first action.
      values.fill(-1e9);
      const me = game.tanks[0];
      const canFire = me.triggerReleased && game.weaponReady(me);
      for (const plan of ROLLOUT_PLANS) {
        if (plan.kind === "fire_then_move" && !canFire) continue;
        const value = densityRollout(game, plan.firstAction, field, seed, {
          boxes: this.boxes,
          chainState: this.chain,
          horizon: this.horizon,
          hold: this.hold,
          oppModel: this.oppModel,
          continuationAction: plan.continuationAction,
        });
        const index = actionIndex(plan.firstAction);
        if (value > values[index]) {
          values[index] = value;
          if (plan.firstAction === STATIONARY_FIRE_ACTION) {
            this.bestFireContinuation = plan.continuationAction;
          }
        }
      }
    } else {
      // Only the ten selectable candidates are rolled out. The eight
      // move-and-shoot columns get masked unconditionally, so simulating them
      // would be 44% of the work for a value that is overwritten anyway.
      for (const index of LIVE_ACTION_INDICES) {
        values[index] = densityRollout(game, CANDIDATES[index], field, seed, {
          boxes: this.boxes,
          chainState: this.chain,
          horizon: this.horizon,
          hold: this.hold,
          oppModel: this.oppModel,
        });
      }
    }
    maskMovingFireScores(values);
    if (this.actionNoEffect && this.observedPreviousAction !== null) {
      const failed = this.observedPreviousAction;
      for (let i = 0; i < CANDIDATES.length; i++) {
        if (CANDIDATES[i][0] === failed[0] && CANDIDATES[i][1] === failed[1]) {
          values[i] -= NO_EFFECT_REPEAT_PENALTY;
        }
      }
    }
    return values;
  }

  /**
   * Decide this frame's move.
   * @returns {number[]} a [throttle, turn, fire] triple
   */
  act(game) {
    const started = performance.now();
    this.lastDecisionKind = "none";
    if (!game.tanks[0].alive) {
      this.observeFireOpportunity(false);
      return [1, 1, 0];
    }
    this.observeActionEffect(game);
    this.observedPreviousAction = this.effectAction ?? [1, 1, 0];

    try {
      // Post-kill: the world is still live and our own bullets can still kill
      // us, so keep making explicit no-fire survival decisions.
      if (!game.tanks[1].alive) {
        this.observeFireOpportunity(false);
        if (this.actionNoEffect) this.commitRemaining = 0;
        if (this.commitRemaining > 0 && !game.tanks[0].hitSomething) {
          this.commitRemaining -= 1;
          const held = [this.committedAction[0], this.committedAction[1], 0];
          return this.emitAction(game, held, "post_kill_hold");
        }
        const values = postKillSurvivalScores(
          game, C.NUMBEROFFRAMESBEFOREEND - C.NUMBEROFFRAMESFROZEN,
        );
        const picked = CANDIDATES[argmax(values)];
        const action = [picked[0], picked[1], 0];
        this.committedAction = action;
        this.commitRemaining = Math.min(1,
          action[0] !== 1 ? this.commitMoveFrames
            : action[1] !== 1 ? COMMIT_TURN_FRAMES : 0);
        return this.emitAction(game, action, "post_kill_plan");
      }

      const field = this.ensureField(game);
      this.updateLiveChain(game, field);
      if (this.actionNoEffect) this.commitRemaining = 0;

      const verifiedHit = KillFieldAgent.verifiedHit(game);
      this.observeFireOpportunity(verifiedHit);
      // K1. A verified firing window cancels commitment so the search runs
      // again, but no longer forces the shot: firing competes with the nine
      // movement plans in the same score and may lose to evasion or to a
      // better shooting position.
      if (verifiedHit && this.fireContinuation) {
        this.commitRemaining = 0;
      } else if (verifiedHit) {
        this.commitRemaining = 0;
        return this.emitAction(game, [1, 1, 1], "forced_fire");
      }

      const contactBrokeCommitment = game.tanks[0].hitSomething
        || (this.wallContactReplan && game.tanks[0].wallSliding);
      if (this.commitRemaining > 0 && !contactBrokeCommitment) {
        this.commitRemaining -= 1;
        return this.emitAction(game, this.committedAction, "hold");
      }

      const values = this.scores(game);
      let pickedIndex = argmax(values);
      if (this.continuityMargin > 0 && CANDIDATES[pickedIndex][2] === 0) {
        const previousIndex = actionIndex(this.lastMotionAction);
        const previous = CANDIDATES[previousIndex];
        if (previous?.[2] === 0
            && values[previousIndex] >= values[pickedIndex] - this.continuityMargin
            && !actionSelfHits(game, previous)) {
          pickedIndex = previousIndex;
          this.continuityHolds += 1;
        }
      }
      const action = CANDIDATES[pickedIndex];
      if (action[2] === 0) {
        this.committedAction = action;
        this.commitRemaining = action[0] !== 1 ? this.commitMoveFrames
          : action[1] !== 1 ? COMMIT_TURN_FRAMES : 0;
      }
      return this.emitAction(game, action, "plan");
    } finally {
      this.planMs.push(performance.now() - started);
      if (this.planMs.length > 600) this.planMs.shift();
    }
  }

  /** Decide and write the result onto tank 0. */
  drive(game) {
    const [throttle, turn, fire] = this.act(game);
    this.observeExecutedAction(game, [throttle, turn, fire]);
    const me = game.tanks[0];
    me.forward = throttle === 2;
    me.backup = throttle === 0;
    me.turnLeft = turn === 0;
    me.turnRight = turn === 2;
    me.fire = fire === 1;
  }

  telemetry() {
    const sorted = this.planMs.slice().sort((a, b) => a - b);
    const at = (q) => (sorted.length ? sorted[Math.min(sorted.length - 1,
      Math.floor(q * sorted.length))] : 0);
    return {
      decision: this.lastDecisionKind,
      action: this.lastAction,
      fieldBuilds: this.fieldBuilds,
      meanFieldBuildMs: this.fieldBuildMs / Math.max(this.fieldBuilds, 1),
      cachedTargetCells: this.fieldCache.size,
      huntChain: this.chain.count,
      huntChainTotal: this.chainTotal,
      ownBulletGuardEvents: this.ownBulletGuardEvents,
      noEffectEvents: this.noEffectEvents,
      continuityHolds: this.continuityHolds,
      fireOpportunityWindows: this.fireOpportunityWindows,
      fireOpportunityCaptures: this.fireOpportunityCaptures,
      fireOpportunityCaptureRate: this.fireOpportunityCaptures
        / Math.max(1, this.fireOpportunityWindows),
      fireOpportunityFrames: this.fireOpportunityFrames,
      missedFireOpportunityFrames: this.missedFireOpportunityFrames,
      movementSwitches: this.movementSwitches,
      movementSwitchesPer1000: 1000 * this.movementSwitches
        / Math.max(1, this.executedCombatFrames),
      throttleReversals: this.throttleReversals,
      turnReversals: this.turnReversals,
      reversalsPer1000: 1000 * (this.throttleReversals + this.turnReversals)
        / Math.max(1, this.executedCombatFrames),
      planMedianMs: at(0.5),
      planP95Ms: at(0.95),
    };
  }
}

export { CANDIDATES, actionIndex };
