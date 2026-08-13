import assert from "node:assert/strict";
import test from "node:test";

import { makeExactSandbox } from "../lib/exact-sandbox.js";
import { Game } from "../lib/killfield-runtime/src/game.js";
import { LaikaAI } from "../lib/killfield-runtime/src/laika.js";
import { applyAction } from "../lib/killfield-runtime/src/killfield/sandbox.js";

function frameAction(frame) {
  return [frame % 17 < 9 ? 2 : 1, frame % 23 < 8 ? 0 : 2, frame % 41 === 0 ? 1 : 0];
}

function snapshot(game) {
  const ai = game.tanks[1].ai;
  return {
    rng: game.rng.state,
    frame: game.frame,
    aliveCount: game.aliveCount,
    endCount: game.endCount,
    resetCount: game.resetCount,
    frozen: game.frozen,
    crateTimer: game.crateTimer,
    scores: game.scores,
    tankFields: game.tankFields,
    tanks: game.tanks.map((tank) => ({
      x: tank.x,
      y: tank.y,
      rotation: tank.rotation,
      alive: tank.alive,
      triggerReleased: tank.triggerReleased,
      bulletsFired: tank.bulletsFired,
      hitSomething: tank.hitSomething,
      input: [tank.forward, tank.backup, tank.turnLeft, tank.turnRight, tank.fire],
    })),
    bullets: game.bullets.map((bullet) => ({
      name: bullet.name,
      owner: bullet.owner.number,
      x: bullet.x,
      y: bullet.y,
      xSpeed: bullet.xSpeed,
      ySpeed: bullet.ySpeed,
      lifetime: bullet.lifetime,
      deadly: bullet.deadly,
      removed: bullet.removed,
      justCreated: bullet.justCreated,
      hasBounced: bullet.hasBounced,
    })),
    ai: ai ? {
      stuckTime: ai.stuckTime,
      currentAggresiveness: ai.currentAggresiveness,
      goalId: ai.goalId,
      goal: ai.myGoal.goal,
      goalPeriod: ai.myGoal.period,
      goalPriority: ai.myGoal.priority,
      actions: ai.myActions.map((action) => ({ ...action })),
    } : null,
  };
}

test("privileged exact sandbox preserves Laika and RNG continuation", () => {
  const live = new Game({
    seed: 970017,
    aiFactory: (game, tank) => new LaikaAI(game, tank),
  });
  for (let frame = 0; frame < 93; frame += 1) {
    applyAction(live, frameAction(frame));
    live.step();
  }
  const clone = makeExactSandbox(live);
  assert.deepEqual(snapshot(clone), snapshot(live));

  for (let frame = 93; frame < 393; frame += 1) {
    const action = frameAction(frame);
    applyAction(live, action);
    applyAction(clone, action);
    const liveEvents = live.step();
    const cloneEvents = clone.step();
    assert.deepEqual(cloneEvents, liveEvents, `events diverged at frame ${frame}`);
    assert.deepEqual(snapshot(clone), snapshot(live), `state diverged at frame ${frame}`);
  }
});
