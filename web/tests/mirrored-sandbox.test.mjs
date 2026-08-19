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

test("mirrored rollouts inherit the world's collision model", () => {
  // A planner that rolls out under different physics than the engine runs is
  // predicting a different game. This was a real defect: `wallSliding` was
  // missing from mirrorView's passthrough list, so the right-hand agent planned
  // against the original collision model while the world slid. It cost 21pp of
  // win rate on the mirrored side alone and was invisible on the left side.
  for (const wallSliding of [false, true]) {
    const game = new Game({ seed: 1300811, wallSliding });
    const view = mirrorView(game);
    assert.equal(view.wallSliding, wallSliding);
    assert.equal(makeSandbox(game, "L1", 0).wallSliding, wallSliding);
    assert.equal(makeSandbox(view, "L1", 0).wallSliding, wallSliding);
  }
});

test("every game field the planner reads survives mirroring", () => {
  // Guards the general shape of the bug above: a new property added to Game
  // that the sandbox copies must also be mirrored, or one side silently
  // diverges.
  const game = new Game({ seed: 1300811, wallSliding: true });
  const view = mirrorView(game);
  const direct = makeSandbox(game, "L1", 0);
  const mirrored = makeSandbox(view, "L1", 0);
  for (const key of Object.keys(direct)) {
    if (["tanks", "bullets", "tankFields", "scores", "rng", "events"].includes(key)) {
      continue; // Deliberately reordered or freshly seeded by the sandbox.
    }
    assert.deepEqual(mirrored[key], direct[key], `sandbox field "${key}" lost in mirror`);
  }
});
