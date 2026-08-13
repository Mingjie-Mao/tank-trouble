import assert from "node:assert/strict";
import test from "node:test";

import { Game } from "../lib/killfield-runtime/src/game.js";
import { mirrorView } from "../lib/killfield-runtime/src/killfield/mirror.js";
import { makeSandbox } from "../lib/killfield-runtime/src/killfield/sandbox.js";

test("mirrored sandbox keeps visible bullet ownership aligned with tank order", () => {
  const game = new Game({ seed: 1300811 });
  game.fireWeapon(game.tanks[1]);

  const normal = makeSandbox(game, "L1", 1);
  assert.equal(normal.bullets[0].owner, normal.tanks[1]);
  assert.equal(normal.bullets[0].owner.number, 1);

  const mirrored = makeSandbox(mirrorView(game), "L1", 1);
  assert.equal(mirrored.bullets[0].owner, mirrored.tanks[0]);
  assert.equal(mirrored.bullets[0].owner.number, 0);
  assert.deepEqual(mirrored.tanks.map((tank) => tank.number), [0, 1]);
  assert.deepEqual(mirrored.tankFields, [game.tankFields[1], game.tankFields[0]]);

  mirrored.destroyTank(0);
  assert.equal(mirrored.tanks[0].alive, false);
  assert.equal(mirrored.tanks[1].alive, true);
});
