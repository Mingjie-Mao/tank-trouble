import * as C from "./killfield-runtime/src/constants.js";
import { LaikaAI } from "./killfield-runtime/src/laika.js";

function clearInputs(tank) {
  tank.forward = false;
  tank.backup = false;
  tank.turnLeft = false;
  tank.turnRight = false;
  tank.fire = false;
}
function makeLaika(game) {
  return new LaikaAI(game, game.tanks[0]);
}

/**
 * Adapter that lets the native Laika script participate on either side of a
 * BrowserArena self-play match. The mirror view always exposes the controlled
 * tank as tanks[0], so this controller never receives privileged identity or
 * future opponent actions.
 */
export class LaikaJsAgent {
  constructor({ profile = "laika" } = {}) {
    this.profile = profile;
    this.roundNumber = null;
    this.controller = null;
    this.lastDecisionKind = profile;
  }

  reset() {
    this.roundNumber = null;
    this.controller = null;
  }

  configure(controller) {
    if (this.profile !== "hunter") return;
    controller.AGGRESIVENESS = 1.0;
    controller.currentAggresiveness = 1.0;
    controller.COWARDNESS = 0.25;
    controller.LONGESTPATHTOSHOOT = 12;
    controller.LONGESTPATHTONOTHESITATETOSHOOT = 5;
    controller.LONGESTPATHTORUN = 5;
    controller.IDLEDRIVETOWARDENEMYPRIORITY = 0.45;
  }

  ensureController(game) {
    if (this.controller && this.roundNumber === game.roundNumber) return;
    this.roundNumber = game.roundNumber;
    this.controller = makeLaika(game);
    this.configure(this.controller);
  }

  drive(game) {
    const tank = game.tanks[0];
    if (!tank.alive) {
      clearInputs(tank);
      return;
    }
    this.ensureController(game);
    if (this.controller.makeDecisionsAndUpdateGoal()) {
      this.controller.decideActionsToAchieveGoal();
    }
    this.controller.setInputToDoActions();
    this.lastDecisionKind = this.profile;
  }

  telemetry() {
    return { decision: this.profile, planMedianMs: 0, planP95Ms: 0 };
  }
}

/**
 * A deliberately defensive visible-state opponent. It reuses Laika's public
 * trajectory scanner and path compiler, but always climbs the maze-distance
 * field away from the enemy when no immediate bullet needs dodging. It does
 * not know the opposing policy and does not simulate future inputs.
 */
export class DodgerJsAgent {
  constructor() {
    this.roundNumber = null;
    this.controller = null;
    this.replanIn = 0;
    this.lastDecisionKind = "dodger";
  }

  reset() {
    this.roundNumber = null;
    this.controller = null;
    this.replanIn = 0;
  }

  ensureController(game) {
    if (this.controller && this.roundNumber === game.roundNumber) return;
    this.roundNumber = game.roundNumber;
    this.controller = makeLaika(game);
    this.replanIn = 0;
  }

  chooseGoal(game) {
    const my = game.tanks[0];
    const scale = game.scale;
    const fx = Math.floor(my.x / scale);
    const fy = Math.floor(my.y / scale);
    const threat = this.controller.dodgeTrajectories(
      fx,
      fy,
      game.bullets,
      100,
      5 * scale,
      (100 * C.BULLETSPEED) / 50,
      C.BULLETHITCHECKINTERVALS,
      true,
    );
    if (threat.priority > 0) {
      this.controller.myGoal = threat;
      this.replanIn = 2;
      return;
    }

    const enemy = game.tanks[1];
    const ex = Math.floor(enemy.x / scale);
    const ey = Math.floor(enemy.y / scale);
    const distance = game.distMap(ex, ey);
    this.controller.myGoal = distance === null
      ? { goal: "idle", period: 1, priority: 0, updateContinuously: false, id: -1 }
      : {
          goal: "runAway",
          dist: distance,
          period: 8,
          priority: 1,
          updateContinuously: false,
          id: -1,
        };
    this.replanIn = 8;
  }

  drive(game) {
    const tank = game.tanks[0];
    if (!tank.alive) {
      clearInputs(tank);
      return;
    }
    this.ensureController(game);
    this.replanIn -= 1;
    if (this.replanIn <= 0 || this.controller.myActions.length === 0) {
      this.chooseGoal(game);
      this.controller.decideActionsToAchieveGoal();
    }
    this.controller.setInputToDoActions();
    this.lastDecisionKind = this.controller.myGoal.goal === "dodgeBullet"
      ? "dodger:bullet"
      : "dodger:distance";
  }

  telemetry() {
    return { decision: this.lastDecisionKind, planMedianMs: 0, planP95Ms: 0 };
  }
}
