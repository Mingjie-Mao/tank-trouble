import * as C from "./killfield-runtime/src/constants.js";
import { CANDIDATES } from "./killfield-runtime/src/killfield/score.js";
import { applyAction, makeSandbox } from "./killfield-runtime/src/killfield/sandbox.js";
import { TacticalV2Agent } from "./tactical-v2-agent.js";

const NO_FIRE_ACTIONS = CANDIDATES.filter((action) => action[2] === 0);

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

/** Verify that a shot which kills under visible motion also has a survivable settlement. */
export function auditShotSettlement(game, {
  horizon = 75,
  opponentBehavior = null,
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
      const settlement = fixedSettlementSurvival(sandbox);
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

export class TacticalCandidateAgent extends TacticalV2Agent {
  constructor({
    enableShotSettlementAudit = false,
    minimumOwnBulletsForAudit = 2,
    maximumUnsafeAuditsPerRound = 4,
    ...options
  } = {}) {
    super(options);
    this.enableShotSettlementAudit = enableShotSettlementAudit;
    this.minimumOwnBulletsForAudit = minimumOwnBulletsForAudit;
    this.maximumUnsafeAuditsPerRound = maximumUnsafeAuditsPerRound;
    this.shotSettlementAudits = 0;
    this.unsafeShotsSuppressed = 0;
    this.shotAuditMs = [];
    this.suppressFireRound = null;
    this.suppressFireUntilFrame = -1;
    this.unsafeAuditRound = null;
    this.unsafeAuditsThisRound = 0;
  }

  act(game) {
    const baseline = super.act(game);
    if (this.unsafeAuditRound !== game.roundNumber) {
      this.unsafeAuditRound = game.roundNumber;
      this.unsafeAuditsThisRound = 0;
    }
    if (baseline[2] === 1 && this.suppressFireRound === game.roundNumber
        && game.frame <= this.suppressFireUntilFrame) {
      this.unsafeShotsSuppressed += 1;
      this.commitRemaining = 0;
      return this.emitVerifiedAction(game, [1, 1, 0], "unsafe_settlement_cached");
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
    return {
      ...super.telemetry(),
      shotSettlementAudits: this.shotSettlementAudits,
      unsafeShotsSuppressed: this.unsafeShotsSuppressed,
      shotAuditP95Ms: sorted.length
        ? sorted[Math.floor(0.95 * (sorted.length - 1))] : 0,
    };
  }
}
