/**
 * Forward-simulation sandbox.
 *
 * The planner needs to roll the world forward without touching it, and without
 * cheating. So the sandbox shares everything a player could see on screen —
 * the maze, wall geometry, distance maps, both poses, every bullet — copies
 * the mutable state, and scrubs the two things that would be hidden knowledge:
 *
 *   - the real random stream is replaced with an independently seeded one
 *   - the opponent's controller is rebuilt from scratch, so its internal goal
 *     stack does not leak across
 *
 * Two opponent models:
 *   L2  runs the scripted AI's algorithm with fresh state. White-box knowledge
 *       of that specific opponent, appropriate when the opponent really is it.
 *   L1  freezes whatever buttons the opponent is currently holding.
 */

import { Game, Tank, Bullet } from "../game.js";
import { LaikaAI } from "../laika.js";
import { Rng } from "../rng.js";

function copyTank(tank, sandbox, visibleIndex) {
  const copy = Object.create(Tank.prototype);
  Object.assign(copy, tank);
  copy.game = sandbox;
  // A sandbox is always ego-relative: tanks[0] is the controlled tank and
  // tanks[1] is its opponent. Game.destroyTank() indexes by this number, so a
  // mirrored rollout must use view-relative numbers as well as array order.
  copy.number = visibleIndex;
  copy.ai = null;
  return copy;
}

function copyBullet(bullet, sandbox, visibleTanks) {
  const copy = Object.create(Bullet.prototype);
  Object.assign(copy, bullet);
  copy.game = sandbox;
  const ownerIndex = visibleTanks.indexOf(bullet.owner);
  if (ownerIndex === -1) throw new Error(`bullet ${bullet.name} owner is not visible`);
  copy.owner = sandbox.tanks[ownerIndex];
  return copy;
}

/**
 * @param {Game} game
 * @param {"L1"|"L2"} oppModel
 * @param {number} rngSeed
 * @returns {Game} a stepping-compatible clone
 */
export function makeSandbox(game, oppModel = "L2", rngSeed = 0) {
  const sb = Object.create(Game.prototype);

  // Shared, read-only for the duration of a round.
  sb.maze = game.maze;
  sb.walls = game.walls;
  sb.wallHalfT = game.wallHalfT;
  sb.scale = game.scale;
  sb.wallGrid = game.wallGrid;
  sb.distancesForMaze = game.distancesForMaze;
  sb.deadEnds = game.deadEnds;
  sb.reachable = game.reachable;
  sb.reachableIndex = game.reachableIndex;
  sb.settingsMaxBullets = game.settingsMaxBullets;
  sb.tanksCount = game.tanksCount;
  sb.aiFactory = null;

  // Hidden information scrubbed.
  sb.rng = new Rng(rngSeed);

  // Mutable state copied.
  // Resolve bullet ownership by object position in the supplied view. Tank
  // numbers are physical identities and therefore are intentionally not
  // swapped by mirrorView(); indexing with number made right-side rollouts
  // attribute every live bullet to the wrong tank.
  const visibleTanks = game.tanks;
  sb.tanks = visibleTanks.map((t, index) => copyTank(t, sb, index));
  sb.bullets = game.bullets.map((b) => copyBullet(b, sb, visibleTanks));
  sb.tankFields = visibleTanks.map((tank) => {
    const field = game.tankFields[tank.number];
    return { x: field.x, y: field.y };
  });
  sb.events = [];
  sb.frame = game.frame;
  sb.aliveCount = game.aliveCount;
  sb.endCount = game.endCount;
  sb.resetCount = game.resetCount;
  sb.frozen = game.frozen;
  sb.shake = game.shake;
  sb.crateTimer = game.crateTimer;
  sb.bulletDepth = game.bulletDepth;
  sb.roundNumber = game.roundNumber;
  sb.scores = game.scores.slice();

  if (oppModel === "L2") {
    sb.tanks[1].ai = new LaikaAI(sb, sb.tanks[1]);
  }
  return sb;
}

/** Write a (throttle, turn, fire) triple onto the sandbox's own tank. */
export function applyAction(sandbox, action) {
  const [throttle, turn, fire] = action;
  const me = sandbox.tanks[0];
  me.forward = throttle === 2;
  me.backup = throttle === 0;
  me.turnLeft = turn === 0;
  me.turnRight = turn === 2;
  me.fire = fire === 1;
}
