import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";
import { gunzipSync } from "node:zlib";

import { Game } from "../lib/engine/game.js";
import { LaikaAI } from "../lib/engine/laika.js";

// Recorded from the Python port by tools/record_engine_parity.py: per frame,
// a digest of the discrete state (exact) and the continuous state rounded to
// 1e-6. Python's libm and V8 round sin/cos differently in the last bit for
// some angles, so floats are matched within a tolerance and everything that
// can branch — counters, flags, ammo, buttons, events — is matched exactly.
const fixture = JSON.parse(gunzipSync(readFileSync(
  new URL("./fixtures/engine-parity.json.gz", import.meta.url))).toString());
const TOLERANCE = 1e-6;

/** The recorder's xorshift32 input stream for tank 0, exactly. */
class Driver {
  constructor(seed) {
    this.s = Number((BigInt(seed) * 2654435761n + 1n) & 0xffffffffn) || 1;
    this.hold = 0;
    this.buttons = [false, false, false, false, false];
  }

  next32() {
    let s = this.s;
    s = (s ^ (s << 13)) >>> 0;
    s = (s ^ (s >>> 17)) >>> 0;
    s = (s ^ (s << 5)) >>> 0;
    this.s = s;
    return s;
  }

  step() {
    if (this.hold <= 0) {
      const r = this.next32();
      const throttle = r % 3;
      const turn = Math.floor(r / 3) % 3;
      this.buttons = [throttle === 2, throttle === 0, turn === 0, turn === 2, Math.floor(r / 9) % 6 === 0];
      this.hold = 3 + (Math.floor(r / 54) % 8);
    }
    this.hold -= 1;
    return this.buttons;
  }
}

const bit = (v) => (v ? "1" : "0");

function snapshot(g, events) {
  const parts = [String(g.frame), String(g.roundNumber), String(g.aliveCount), String(g.endCount),
    String(g.resetCount), bit(g.frozen), g.crateTimer.toFixed(9), g.scores.join(",")];
  const floats = [];
  for (const t of g.tanks) {
    parts.push(`T${[bit(t.alive), String(t.bulletsFired), bit(t.triggerReleased), bit(t.hitSomething),
      bit(t.forward), bit(t.backup), bit(t.turnLeft), bit(t.turnRight), bit(t.fire)].join(",")}`);
    floats.push(t.x, t.y, t.rotation);
  }
  for (const b of g.bullets) {
    parts.push(`B${[b.name, String(b.owner.number), String(b.lifetime), bit(b.hasExitedOwner)].join(",")}`);
    floats.push(b.x, b.y, b.xSpeed, b.ySpeed);
  }
  for (const e of events) parts.push(`E${e.map((v) => (v === null ? "null" : String(v))).join(",")}`);
  return [createHash("sha1").update(parts.join("|")).digest("hex").slice(0, 16), floats];
}

for (const { seed, rounds, frames } of fixture.cases) {
  test(`seed ${seed}: ${fixture.frames} frames, ${rounds} rounds, matches the Python port`, () => {
    const g = new Game({ seed, aiFactory: (game, tank) => new LaikaAI(game, tank) });
    const drive = new Driver(seed);
    let events = [];
    for (let frame = 0; frame <= fixture.frames; frame++) {
      if (frame > 0) {
        if (!g.frozen) {
          const t = g.tanks[0];
          [t.forward, t.backup, t.turnLeft, t.turnRight, t.fire] = drive.step();
        }
        events = g.step();
      }
      const [discrete, floats] = snapshot(g, events);
      const [expectedDiscrete, expectedFloats] = frames[frame];
      assert.equal(discrete, expectedDiscrete, `discrete state differs at frame ${frame}`);
      assert.equal(floats.length, expectedFloats.length, `frame ${frame}`);
      for (let i = 0; i < floats.length; i++) {
        assert.ok(Math.abs(floats[i] - expectedFloats[i]) <= TOLERANCE,
          `frame ${frame} value ${i}: ${floats[i]} vs ${expectedFloats[i]}`);
      }
    }
    assert.equal(g.roundNumber, rounds);
  });
}
