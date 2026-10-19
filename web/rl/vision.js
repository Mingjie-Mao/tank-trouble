/**
 * What tank 0's controller sees, for the arena's "AI view" overlay.
 *
 * Every value is computed from the visible game state with the engine's own
 * rules: the nine-move `dodgeSafety` lookahead the policy observes, the
 * ricochet path of a shot fired now, and the bullets forecast to hit tank 0.
 * The policy's action distribution is added when the controller is the
 * league PPO network.
 */

import * as C from "../lib/engine/constants.js";
import { CANDIDATES } from "../lib/killfield-runtime/src/killfield/score.js";
import { DODGE_DIM, dodgeSafety } from "./duel-env.js";

const SUBSTEPS = C.BULLETHITCHECKINTERVALS;
const SHOT_FRAMES = Math.floor(C.BULLETLIFETIME / 3);
const THREAT_FRAMES = 75;

/** Move a point one substep, reflecting off walls as Bullet.update does. */
function substep(game, p) {
  const prevX = p.x;
  const prevY = p.y;
  p.x += p.xs;
  p.y += p.ys;
  if (!game.wallHit(p.x, p.y)) return false;
  const hitXInv = game.wallHit(prevX - p.xs, prevY + p.ys);
  const hitYInv = game.wallHit(prevX + p.xs, prevY - p.ys);
  if (hitXInv && !hitYInv) p.ys = -p.ys;
  else if (hitYInv && !hitXInv) p.xs = -p.xs;
  else { p.xs = -p.xs; p.ys = -p.ys; }
  p.x = prevX + p.xs;
  p.y = prevY + p.ys;
  return true;
}

const hits = (tank, x, y) => tank.alive && tank.pointInBbox(x, y) && tank.pointInShape(x, y);

/** The polyline of a shot fired now from `tank`'s barrel, and what it hits. */
function shotPath(game, tank) {
  const rad = ((tank.rotation - 90) * Math.PI) / 180;
  const scale = game.scale;
  const p = {
    x: tank.x + Math.cos(rad) * scale * 4.5 / 16,
    y: tank.y + Math.sin(rad) * scale * 4.5 / 16,
    xs: Math.cos(rad) * C.BULLETSPEED / SUBSTEPS * (scale / 50),
    ys: Math.sin(rad) * C.BULLETSPEED / SUBSTEPS * (scale / 50),
  };
  const points = [[p.x, p.y]];
  for (let frame = 0; frame < SHOT_FRAMES; frame++) {
    for (let s = 0; s < SUBSTEPS; s++) {
      if (substep(game, p)) points.push([p.x, p.y]);
    }
    for (let i = 0; i < game.tanks.length; i++) {
      if (hits(game.tanks[i], p.x, p.y)) {
        points.push([p.x, p.y]);
        return { points, hit: i === 0 ? "self" : "opponent" };
      }
    }
  }
  points.push([p.x, p.y]);
  return { points, hit: "none" };
}

/** Ids of live bullets whose flight reaches tank 0 within THREAT_FRAMES. */
function threats(game) {
  const me = game.tanks[0];
  if (!me.alive) return [];
  const out = [];
  for (const bullet of game.bullets) {
    if (bullet.removed) continue;
    const p = { x: bullet.x, y: bullet.y, xs: bullet.xSpeed, ys: bullet.ySpeed };
    for (let frame = 0; frame < Math.min(THREAT_FRAMES, bullet.lifetime); frame++) {
      for (let s = 0; s < SUBSTEPS; s++) substep(game, p);
      if (hits(me, p.x, p.y)) {
        out.push(bullet.name);
        break;
      }
    }
  }
  return out;
}

/** The policy's three most likely actions after the shield, as [throttle, turn, fire, p]. */
function topActions(probs) {
  return Array.from(probs.keys())
    .sort((a, b) => probs[b] - probs[a])
    .slice(0, 3)
    .filter((a) => probs[a] > 0.005)
    .map((a) => [...CANDIDATES[a], probs[a]]);
}

export function visionFor(game, agent) {
  const me = game.tanks[0];
  if (!me.alive || game.frozen) return null;
  const dodge = new Float32Array(DODGE_DIM);
  dodgeSafety(game, 0, dodge, 0);
  const vision = {
    // [throttle 0 back / 1 none / 2 forward][turn 0 left / 1 none / 2 right]: survives?
    dodge: Array.from(dodge, (value) => value >= 0),
    shot: shotPath(game, me),
    threats: threats(game),
  };
  if (agent?.lastProbs) {
    vision.actions = topActions(agent.lastProbs);
    vision.shieldVetoed = Boolean(agent.lastOverridden);
  }
  return vision;
}
