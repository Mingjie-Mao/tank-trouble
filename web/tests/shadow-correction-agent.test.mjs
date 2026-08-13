import assert from "node:assert/strict";
import test from "node:test";

import { Game } from "../lib/killfield-runtime/src/game.js";
import { LaikaAI } from "../lib/killfield-runtime/src/laika.js";
import { KillFieldAgent } from "../lib/killfield-runtime/src/killfield/teacher.js";
import { ShadowCorrectionAgent } from "../lib/shadow-correction-agent.js";

function fingerprint(game) {
  return {
    rng: game.rng.state,
    frame: game.frame,
    alive: game.tanks.map((tank) => tank.alive),
    poses: game.tanks.map((tank) => [tank.x, tank.y, tank.rotation]),
    bullets: game.bullets.map((bullet) => [
      bullet.owner.number, bullet.x, bullet.y, bullet.xSpeed, bullet.ySpeed, bullet.lifetime,
    ]),
  };
}

test("shadow correction never changes the frozen H36 live trajectory", () => {
  const makeGame = () => new Game({
    seed: 970001,
    aiFactory: (game, tank) => new LaikaAI(game, tank),
  });
  const baselineGame = makeGame();
  const shadowGame = makeGame();
  const baseline = new KillFieldAgent({ seed: 970102, oppModel: "L2" });
  const shadow = new ShadowCorrectionAgent({
    seed: 970102,
    oppModel: "L2",
    auditCooldown: 1000,
    maxAuditsPerRound: 1,
    auditHorizon: 20,
  });

  for (let frame = 0; frame < 80; frame += 1) {
    baseline.drive(baselineGame);
    shadow.drive(shadowGame);
    assert.deepEqual(shadow.lastAction, baseline.lastAction, `action diverged at ${frame}`);
    baselineGame.step();
    shadowGame.step();
    assert.deepEqual(fingerprint(shadowGame), fingerprint(baselineGame),
      `world diverged at ${frame}`);
  }
});
