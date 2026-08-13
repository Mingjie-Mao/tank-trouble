import { CANDIDATES, LIVE_ACTION_INDICES } from "./killfield-runtime/src/killfield/score.js";

export const SEARCH_ACTIONS = Object.freeze(
  LIVE_ACTION_INDICES.map((index) => Object.freeze(Array.from(CANDIDATES[index]))),
);

function cellOf(game, tank) {
  return [Math.floor(tank.x / game.scale), Math.floor(tank.y / game.scale)];
}

function signedAngleDelta(target, current) {
  return ((target - current + 540) % 360) - 180;
}

function wallProbe(game, tank, angleDegrees) {
  const radians = ((angleDegrees - 90) * Math.PI) / 180;
  const step = game.scale * 0.2;
  const limit = game.scale * 2;
  for (let distance = step; distance <= limit; distance += step) {
    if (game.wallHit(
      tank.x + Math.cos(radians) * distance,
      tank.y + Math.sin(radians) * distance,
    )) return distance / limit;
  }
  return 1;
}

/** Cheap, visible-state-only inputs shared by collection and browser inference. */
export function searchFeatures(game, agent) {
  const me = game.tanks[0];
  const enemy = game.tanks[1];
  const width = game.maze.length;
  const height = game.maze[0].length;
  const diagonal = Math.max(1, Math.hypot(width, height));
  const [mx, my] = cellOf(game, me);
  const [ex, ey] = cellOf(game, enemy);
  const topology = game.distMap(mx, my)?.[ex]?.[ey];
  const field = agent.field;
  const [aim, concentration] = field?.bestAimAt(
    [mx, my], ((me.rotation - 90) * Math.PI) / 180,
  ) ?? [null, 0];
  const heading = (me.rotation * Math.PI) / 180;
  const enemyHeading = (enemy.rotation * Math.PI) / 180;
  const targetHeading = Math.atan2(enemy.x - me.x, -(enemy.y - me.y));
  const aimDegrees = aim === null ? me.rotation : (aim * 180) / Math.PI + 90;
  const last = agent.lastMotionAction ?? [1, 1, 0];
  const bullets = game.bullets.filter((bullet) => !bullet.removed);
  let closestBullet = diagonal;
  for (const bullet of bullets) {
    closestBullet = Math.min(
      closestBullet,
      Math.hypot(bullet.x - me.x, bullet.y - me.y) / game.scale,
    );
  }

  return [
    me.x / Math.max(game.scale * width, 1),
    me.y / Math.max(game.scale * height, 1),
    enemy.x / Math.max(game.scale * width, 1),
    enemy.y / Math.max(game.scale * height, 1),
    (enemy.x - me.x) / Math.max(game.scale * width, 1),
    (enemy.y - me.y) / Math.max(game.scale * height, 1),
    Math.hypot(enemy.x - me.x, enemy.y - me.y) / (game.scale * diagonal),
    Number.isFinite(topology) ? topology / diagonal : 1,
    Math.sin(heading), Math.cos(heading),
    Math.sin(enemyHeading), Math.cos(enemyHeading),
    Math.sin((targetHeading * 180 / Math.PI - me.rotation) * Math.PI / 180),
    Math.cos((targetHeading * 180 / Math.PI - me.rotation) * Math.PI / 180),
    Math.sin(signedAngleDelta(aimDegrees, me.rotation) * Math.PI / 180),
    Math.cos(signedAngleDelta(aimDegrees, me.rotation) * Math.PI / 180),
    concentration,
    (field?.valueAt([mx, my]) ?? 0) / 128,
    field?.guidanceAt([mx, my]) ?? 0,
    field?.relativeSuccessAt([mx, my]) ?? 0,
    (me.x / game.scale) % 1,
    (me.y / game.scale) % 1,
    (enemy.x / game.scale) % 1,
    (enemy.y / game.scale) % 1,
    wallProbe(game, me, me.rotation),
    wallProbe(game, me, me.rotation + 90),
    wallProbe(game, me, me.rotation + 180),
    wallProbe(game, me, me.rotation - 90),
    me.hitSomething ? 1 : 0,
    agent.actionNoEffect ? 1 : 0,
    last[0] / 2,
    last[1] / 2,
    me.bulletsFired / Math.max(game.settingsMaxBullets, 1),
    enemy.bulletsFired / Math.max(game.settingsMaxBullets, 1),
    Math.min(1, bullets.length / 10),
    Math.min(1, closestBullet / diagonal),
  ];
}

export function searchActionClass(action) {
  return SEARCH_ACTIONS.findIndex((candidate) => (
    candidate[0] === action[0] && candidate[1] === action[1] && candidate[2] === action[2]
  ));
}

/** Base state plus an exact one-hot of the previously executed search action. */
export function searchPriorFeatures(game, agent) {
  const features = searchFeatures(game, agent);
  const previousClass = searchActionClass(agent.lastMotionAction ?? [1, 1, 0]);
  for (let index = 0; index < SEARCH_ACTIONS.length; index += 1) {
    features.push(index === previousClass ? 1 : 0);
  }
  return features;
}
