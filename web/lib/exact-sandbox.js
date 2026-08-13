/**
 * Privileged, state-exact clone of the browser game.
 *
 * Unlike KillField's deployable sandbox, this deliberately copies Laika's
 * private goal/action stack and the live RNG state. It is therefore suitable
 * only for offline labels, regression tests, and shadow evaluation. Never use
 * it to choose the action that is sent to the live game.
 */

import { Bullet, Game, Tank } from "./killfield-runtime/src/game.js";
import { LaikaAI } from "./killfield-runtime/src/laika.js";
import { Rng } from "./killfield-runtime/src/rng.js";

function cloneTank(tank, sandbox) {
  const copy = Object.create(Tank.prototype);
  Object.assign(copy, tank);
  copy.game = sandbox;
  copy.ai = null;
  return copy;
}

function cloneBullet(bullet, sandbox) {
  const copy = Object.create(Bullet.prototype);
  Object.assign(copy, bullet);
  copy.game = sandbox;
  copy.owner = sandbox.tanks[bullet.owner.number];
  return copy;
}

function cloneAiValue(value, source, target, seen = new Map()) {
  if (value === null || typeof value !== "object") return value;
  if (value === source) return target;
  const tankIndex = source.tanks.indexOf(value);
  if (tankIndex >= 0) return target.tanks[tankIndex];
  const bulletIndex = source.bullets.indexOf(value);
  if (bulletIndex >= 0) return target.bullets[bulletIndex];
  if (seen.has(value)) return seen.get(value);

  const copy = Array.isArray(value) ? [] : Object.create(Object.getPrototypeOf(value));
  seen.set(value, copy);
  for (const [key, item] of Object.entries(value)) {
    copy[key] = cloneAiValue(item, source, target, seen);
  }
  return copy;
}

function cloneLaika(ai, source, target) {
  const copy = Object.create(LaikaAI.prototype);
  for (const [key, value] of Object.entries(ai)) {
    if (key === "game") copy.game = target;
    else if (key === "myTank") copy.myTank = target.tanks[value.number];
    else copy[key] = cloneAiValue(value, source, target);
  }
  return copy;
}

/**
 * Copy every mutable continuation variable, including privileged opponent
 * internals. Shared maze geometry is immutable until the next round.
 */
export function makeExactSandbox(game) {
  const sandbox = Object.create(Game.prototype);

  sandbox.seed = game.seed;
  sandbox.tanksCount = game.tanksCount;
  sandbox.aiFactory = game.aiFactory;
  sandbox.settingsMaxBullets = game.settingsMaxBullets;

  sandbox.maze = game.maze;
  sandbox.scale = game.scale;
  sandbox.walls = game.walls;
  sandbox.wallHalfT = game.wallHalfT;
  sandbox.wallGrid = game.wallGrid;
  sandbox.reachable = game.reachable;
  sandbox.reachableIndex = game.reachableIndex;
  sandbox.distancesForMaze = game.distancesForMaze;
  sandbox.deadEnds = game.deadEnds;

  sandbox.rng = new Rng(game.rng.seed);
  sandbox.rng.state = game.rng.state;

  sandbox.tanks = game.tanks.map((tank) => cloneTank(tank, sandbox));
  sandbox.bullets = game.bullets.map((bullet) => cloneBullet(bullet, sandbox));
  for (let index = 0; index < game.tanks.length; index += 1) {
    if (game.tanks[index].ai instanceof LaikaAI) {
      sandbox.tanks[index].ai = cloneLaika(game.tanks[index].ai, game, sandbox);
    }
  }

  sandbox.tankFields = game.tankFields.map((field) => ({ x: field.x, y: field.y }));
  sandbox.aliveCount = game.aliveCount;
  sandbox.endCount = game.endCount;
  sandbox.resetCount = game.resetCount;
  sandbox.frozen = game.frozen;
  sandbox.shake = game.shake;
  sandbox.crateTimer = game.crateTimer;
  sandbox.scores = game.scores.slice();
  sandbox.roundNumber = game.roundNumber;
  sandbox.frame = game.frame;
  sandbox.events = game.events.map((event) => Array.from(event));
  sandbox.bulletDepth = game.bulletDepth;
  return sandbox;
}
