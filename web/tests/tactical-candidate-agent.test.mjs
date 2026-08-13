import assert from "node:assert/strict";
import test from "node:test";

import { BrowserArena } from "../lib/browser-arena.js";
import { VisibleOpponentModel } from "../lib/visible-opponent-model.js";

for (const policy of [
  "p27-js-tactical-dd-safe",
  "p27-js-tactical-opponent-model",
  "p27-js-tactical-combined",
]) {
  test(`${policy} runs as a hidden evaluation candidate`, () => {
    const arena = new BrowserArena({ seed: 2500042 });
    arena.command({
      action: "mode",
      mode: "selfplay",
      left_policy: policy,
      right_policy: "laika-js",
    });
    for (let frame = 0; frame < 25; frame += 1) arena.step(frame * 40);
    assert.equal(arena.state().left_policy, policy);
  });
}

test("visible opponent model learns generic action persistence and transition", () => {
  const model = new VisibleOpponentModel({ minimumTransitions: 1 });
  const enemy = {
    backup: false, forward: true, turnLeft: false, turnRight: false, fire: false,
  };
  const game = { roundNumber: 1, frame: 1, tanks: [{}, enemy] };
  model.observe(game);
  game.frame = 4;
  model.observe(game);
  enemy.forward = false;
  enemy.turnLeft = true;
  game.frame = 5;
  model.observe(game);
  enemy.forward = true;
  enemy.turnLeft = false;
  game.frame = 7;
  model.observe(game);
  assert.equal(model.telemetry().visibleOpponentModelReady, true);
  assert.deepEqual(model.predict(5), [1, 0, 0]);
});

test("shot settlement audit repairs a real double death without a seed rule", () => {
  const play = (policy) => {
    const arena = new BrowserArena({ seed: 2700002 });
    arena.command({
      action: "mode",
      mode: "selfplay",
      left_policy: policy,
      right_policy: "dodger-js",
    });
    for (let frame = 0; frame < 3000; frame += 1) {
      arena.step(frame * 40);
      const roundEnd = arena.lastEvents.find((event) => event[0] === "round_end");
      if (roundEnd) return { roundEnd, telemetry: arena.leftAgent.telemetry() };
    }
    throw new Error("round did not finish");
  };

  assert.deepEqual(play("p27-js-tactical-dd-safe-m0").roundEnd, ["round_end", null]);
  const repaired = play("p27-js-tactical-dd-safe");
  assert.deepEqual(repaired.roundEnd, ["round_end", 0]);
  assert.ok(repaired.telemetry.unsafeShotsSuppressed >= 1);
  assert.ok(repaired.telemetry.shotAuditP95Ms < 15);
});
