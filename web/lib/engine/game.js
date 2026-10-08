/**
 * The simulation, translated from `tank_trouble_original/game.py`.
 *
 * Fixed 25 FPS; one `step()` is one Flash enterFrame dispatch: the root
 * round logic, then tanks in creation order, then bullets in creation order,
 * with a bullet fired this frame not moving until the next one. The Python
 * port is the reference: `web/tests/engine-parity.test.mjs` replays recorded
 * Python trajectories through this file and requires them to match.
 */

import * as C from "./constants.js";
import { Rng } from "./rng.js";
import {
  buildWallSegments, calcDistances, calcReachable, createMaze, findDeadEnds, WallGrid,
} from "./maze.js";

const DEG = C.DEG;

/** Flash `_rotation` semantics: normalise to (-180, 180] on every write. */
export function normRot(deg) {
  deg %= 360.0; // fmod, as in Python's math.fmod
  if (deg > 180.0) deg -= 360.0;
  else if (deg <= -180.0) deg += 360.0;
  return deg;
}

// ============================================================== Tank

export class Tank {
  constructor(game, number, cell, scale, rng) {
    this.game = game;
    this.number = number;
    // deployTank: centre of the cell, heading random(32) * 11.25
    this.x = (cell.x + 0.5) * scale;
    this.y = (cell.y + 0.5) * scale;
    this.rotation = normRot(Math.floor(rng.random() * 32) * 11.25);
    this.forwardSpeed = C.TANK_FORWARD_SPEED_BASE * (scale / 50.0);
    this.backupSpeed = C.TANK_BACKUP_SPEED_BASE * (scale / 50.0);
    this.turnSpeed = C.TANK_TURN_SPEED;
    this.displayScale = C.TANK_DISPLAY_SCALE_FACTOR * scale;

    this.triggerReleased = true;
    this.bulletsFired = 0;
    this.alive = true;
    this.currentWeapon = C.STARTWEAPON;
    this.hitSomething = false;
    // The original collision model has no wall sliding; kept for readers.
    this.wallSliding = false;

    this.forward = false;
    this.backup = false;
    this.turnLeft = false;
    this.turnRight = false;
    this.fire = false;
    this.ai = null;

    // Wall probes in local sprite units (tank sprite :58-82). The front row
    // has no centre point and the rear row does; that asymmetry is original.
    const bw = C.TANK_BASE_WIDTH;
    const bh = C.TANK_BASE_HEIGHT;
    const tw = C.TANK_TURRET_WIDTH;
    const th = C.TANK_TURRET_HEIGHT;
    this.hitPointsFront = [
      [-bw / 2, -bh / 2], [-bw / 4, -bh / 2], [bw / 4, -bh / 2], [bw / 2, -bh / 2],
      [-tw / 6, (-th / 16) * 11], [tw / 6, (-th / 16) * 11],
    ];
    this.hitPointsRear = [
      [-bw / 2, bh / 2], [-bw / 4, bh / 2], [0, bh / 2], [bw / 4, bh / 2], [bw / 2, bh / 2],
    ];
    this.hitPointsRight = [
      [bw / 2, (-bh / 6) * 2], [bw / 2, -bh / 6], [bw / 2, 0], [bw / 2, bh / 6], [bw / 2, (bh / 6) * 2],
    ];
    this.hitPointsLeft = [
      [-bw / 2, (-bh / 6) * 2], [-bw / 2, -bh / 6], [-bw / 2, 0], [-bw / 2, bh / 6], [-bw / 2, (bh / 6) * 2],
    ];
  }

  /** localToGlobal: scale, rotate, translate. */
  localToGlobal(lx, ly) {
    const s = this.displayScale;
    const th = this.rotation * DEG;
    const c = Math.cos(th);
    const sn = Math.sin(th);
    return [this.x + s * (lx * c - ly * sn), this.y + s * (lx * sn + ly * c)];
  }

  /**
   * tank:12-26 — does any probe point touch a wall? Same arithmetic as
   * localToGlobal per point, with the heading's sine and cosine taken once:
   * this runs up to six times per tank per frame and inside every rollout.
   */
  hitCheck(points, factor = 1) {
    const g = this.game;
    const s = this.displayScale;
    const th = this.rotation * DEG;
    const c = Math.cos(th);
    const sn = Math.sin(th);
    for (let i = 0; i < points.length; i++) {
      const lx = points[i][0] * factor;
      const ly = points[i][1] * factor;
      if (g.wallHit(this.x + s * (lx * c - ly * sn), this.y + s * (lx * sn + ly * c))) return true;
    }
    return false;
  }

  /** tank:27-41 — the same with the probes pushed outwards (Laika's backAway). */
  expandedHitCheck(points, factor) {
    return this.hitCheck(points, factor);
  }

  anySideHit() {
    return this.hitCheck(this.hitPointsFront) || this.hitCheck(this.hitPointsRear)
      || this.hitCheck(this.hitPointsLeft) || this.hitCheck(this.hitPointsRight);
  }

  /** hitTest(x, y, true): hull rectangle union barrel rectangle. */
  pointInShape(px, py) {
    const s = this.displayScale;
    const th = this.rotation * DEG;
    const c = Math.cos(th);
    const sn = Math.sin(th);
    const dx = px - this.x;
    const dy = py - this.y;
    const lx = (dx * c + dy * sn) / s;
    const ly = (-dx * sn + dy * c) / s;
    const bw2 = C.TANK_BASE_WIDTH / 2;
    const bh2 = C.TANK_BASE_HEIGHT / 2;
    if (-bw2 <= lx && lx <= bw2 && -bh2 <= ly && ly <= bh2) return true;
    return Math.abs(lx) <= C.TANK_SHAPE_BARREL_HALF_WIDTH
      && C.TANK_SHAPE_BARREL_TIP_Y <= ly && ly <= 0;
  }

  /** hitTest(x, y, false): the axis-aligned box around the rotated bounds. */
  pointInBbox(px, py) {
    const s = this.displayScale;
    const th = this.rotation * DEG;
    const c = Math.cos(th);
    const sn = Math.sin(th);
    const [xmin, ymin, xmax, ymax] = C.TANK_BOUNDS_LOCAL;
    let loX = Infinity;
    let hiX = -Infinity;
    let loY = Infinity;
    let hiY = -Infinity;
    for (const [lx, ly] of [[xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]]) {
      const gx = this.x + s * (lx * c - ly * sn);
      const gy = this.y + s * (lx * sn + ly * c);
      if (gx < loX) loX = gx;
      if (gx > hiX) hiX = gx;
      if (gy < loY) loY = gy;
      if (gy > hiY) hiY = gy;
    }
    return loX <= px && px <= hiX && loY <= py && py <= hiY;
  }

  /** tank onEnterFrame :138-433. */
  update() {
    const g = this.game;
    if (g.frozen) return;
    if (!(this.alive && !g.lockedControl(this))) return;

    const oldX = this.x;
    const oldY = this.y;
    const oldRot = this.rotation;

    // The AI writes this tank's input before anything moves (tank :314-321).
    if (this.ai !== null) {
      if (this.ai.makeDecisionsAndUpdateGoal()) this.ai.decideActionsToAchieveGoal();
      this.ai.setInputToDoActions();
    }

    const STEPS = C.TANK_MOVE_STEPS;
    let moveSize = 0.0;
    let turnSize = 0.0;
    if (this.forward) moveSize = this.forwardSpeed / STEPS;
    if (this.backup) moveSize -= this.backupSpeed / STEPS;
    if (this.turnLeft) turnSize = -this.turnSpeed / STEPS;
    if (this.turnRight) turnSize += this.turnSpeed / STEPS;

    this.hitSomething = false;
    // Optimistic pass: all five substeps, walls ignored.
    for (let i = 0; i < STEPS; i++) {
      this.rotation = normRot(this.rotation + turnSize);
      const rad = (this.rotation - 90) * DEG;
      this.x += Math.cos(rad) * moveSize;
      this.y += Math.sin(rad) * moveSize;
    }

    // Ended inside a wall: roll back and redo it substep by substep, turning
    // back on contact and testing only the leading probe row for motion.
    if (this.anySideHit()) {
      this.x = oldX;
      this.y = oldY;
      this.rotation = oldRot;
      for (let i = 0; i < STEPS; i++) {
        const stepOldRot = this.rotation;
        this.rotation = normRot(this.rotation + turnSize);
        if (this.anySideHit()) {
          this.rotation = stepOldRot;
          this.hitSomething = true;
        }
        const stepOldX = this.x;
        const stepOldY = this.y;
        const rad = (this.rotation - 90) * DEG;
        this.x += Math.cos(rad) * moveSize;
        this.y += Math.sin(rad) * moveSize;
        if (moveSize > 0 && this.hitCheck(this.hitPointsFront)) {
          this.x = stepOldX;
          this.y = stepOldY;
          this.hitSomething = true;
        } else if (moveSize < 0 && this.hitCheck(this.hitPointsRear)) {
          this.x = stepOldX;
          this.y = stepOldY;
          this.hitSomething = true;
        }
      }
    }

    // Snap the heading to a multiple of the turn rate (tank :399-410).
    const offset = (360 + this.rotation) % this.turnSpeed;
    if (!this.hitSomething && turnSize !== 0 && offset !== 0) {
      if (offset < this.turnSpeed / 2) this.rotation = normRot(this.rotation - offset);
      else this.rotation = normRot(this.rotation + (this.turnSpeed - offset));
    }

    // Edge-triggered fire (tank :411-419).
    if (this.fire && this.triggerReleased && g.weaponReady(this)) {
      this.triggerReleased = false;
      g.fireWeapon(this);
    } else if (!this.fire) {
      this.triggerReleased = true;
    }
  }
}

// ============================================================== Bullet

export class Bullet {
  constructor(game, name, owner, scale) {
    this.game = game;
    this.name = name;
    this.owner = owner;
    const rad = (owner.rotation - 90) * DEG;
    // fireBullet (frame_53:1259-1282)
    this.x = owner.x + Math.cos(rad) * scale * 4.5 / 16;
    this.y = owner.y + Math.sin(rad) * scale * 4.5 / 16;
    this.xSpeed = Math.cos(rad) * C.BULLETSPEED / C.BULLETHITCHECKINTERVALS * (scale / 50.0);
    this.ySpeed = Math.sin(rad) * C.BULLETSPEED / C.BULLETHITCHECKINTERVALS * (scale / 50.0);
    this.lifetime = C.BULLETLIFETIME;
    this.deadly = C.BULLETDEADLY;
    this.removed = false;
    this.justCreated = false;
    // The muzzle point lies inside the shooter's own hit shape. The shooter
    // is exempt until the bullet has been outside that shape once; a
    // ricochet almost always leaves before it returns, so this only spares
    // a tank driving straight after its own shot.
    this.hasExitedOwner = false;
    // Not a rule, a fact: whether this bullet has come off a wall yet.
    this.hasBounced = false;
  }

  /** bullet onEnterFrame. */
  update() {
    const g = this.game;
    if (g.frozen) return;
    for (let step = 0; step < C.BULLETHITCHECKINTERVALS; step++) {
      const prevX = this.x;
      const prevY = this.y;
      this.x += this.xSpeed;
      this.y += this.ySpeed;
      if (g.wallHit(this.x, this.y)) {
        g.events.push(["bounce", this.name]);
        this.hasBounced = true;
        // The two inversion probes (bullet :31-58) look asymmetric because
        // they are; that is what fixes every ricochet angle in the game.
        const hitOnXInvert = g.wallHit(prevX - this.xSpeed, prevY + this.ySpeed);
        const hitOnYInvert = g.wallHit(prevX + this.xSpeed, prevY - this.ySpeed);
        if (hitOnXInvert && !hitOnYInvert) {
          this.ySpeed = -this.ySpeed;
        } else if (hitOnYInvert && !hitOnXInvert) {
          this.xSpeed = -this.xSpeed;
        } else {
          this.xSpeed = -this.xSpeed;
          this.ySpeed = -this.ySpeed;
        }
        this.x = prevX + this.xSpeed;
        this.y = prevY + this.ySpeed;
      }
    }

    // One hit test per frame, after the substeps (bullet :90-104). No break:
    // a bullet can hit two tanks on the same frame.
    if (this.deadly === 0) {
      if (!this.hasExitedOwner && !this.owner.pointInShape(this.x, this.y)) {
        this.hasExitedOwner = true;
      }
      for (let i = 0; i < g.tanksCount; i++) {
        const tank = g.tanks[i];
        if (tank === this.owner && !this.hasExitedOwner) continue;
        if (tank.alive && tank.pointInShape(this.x, this.y)) {
          g.registerHit(this.owner, tank);
          this.owner.bulletsFired -= 1;
          g.destroyTank(i);
          this.removed = true;
        }
      }
    }
    if (this.deadly > 0) this.deadly -= 1;

    this.lifetime -= 1;
    if (this.lifetime <= 0 && !this.removed) {
      this.owner.bulletsFired -= 1;
      this.removed = true;
    }
  }
}

// ============================================================== Game

export class Game {
  /**
   * @param {object} opts
   * @param {number|null} opts.seed        map seed; null draws one at random
   * @param {number}      opts.tanks       tank count
   * @param {Function|null} opts.aiFactory (game, tank) => controller for tank 1, every round
   */
  constructor({ seed = null, tanks = 2, aiFactory = null, wallSliding = false } = {}) {
    if (wallSliding) {
      throw new Error("this engine implements the original collision model only");
    }
    this.rng = new Rng(seed);
    this.seed = this.rng.seed;
    this.tanksCount = tanks;
    this.aiFactory = aiFactory;
    this.wallSliding = false;
    this.settingsMaxBullets = C.SETTINGS_MAX_BULLETS;

    this.aliveCount = 0;
    this.endCount = -1;
    this.resetCount = -1;
    this.frozen = false;
    this.shake = 0.0;
    // Crates never spawn against Laika, but the timer runs and draws from the RNG.
    this.crateTimer = (C.CRATESPAWNTIMEBASE + this.rng.randrange(C.CRATESPAWNTIMERANDOM))
      * C.SETTINGS_CRATE_SPAWN_MODIFIER;

    this.scores = new Array(tanks).fill(0);
    this.roundNumber = 0;
    this.frame = 0;
    this.events = [];

    this.maze = null;
    this.scale = 50.0;
    this.walls = [];
    this.wallHalfT = 3;
    this.wallGrid = null;
    this.reachable = [];
    this.reachableIndex = null;
    this.distancesForMaze = null;
    this.deadEnds = null;
    this.tankFields = [];
    this.tanks = [];
    this.bullets = [];
    this.bulletDepth = 0;
    this.setupBattle();
  }

  /** frame_53 setupBattle + setupStandardMaze + deployTank. */
  setupBattle() {
    this.roundNumber += 1;
    const rng = this.rng;
    const TANKS = this.tanksCount;

    const spawnCells = new Array(TANKS).fill(null);
    this.reachable = [];
    while (this.reachable.length < 2 * TANKS) {
      const width = rng.randrange(9) + 4; // 4..12
      const height = rng.randrange(7) + 4; // 4..10
      this.scale = Math.min(
        (C.MOVIEHEIGHT - C.HEIGHTTOBOTTOM) / (height + 0.125),
        C.MOVIEWIDTH / (width + 0.125),
      );
      this.maze = createMaze(width, height, rng);
      spawnCells[0] = { x: Math.floor(rng.random() * width), y: Math.floor(rng.random() * height) };
      const r = calcReachable(this.maze, spawnCells[0].x, spawnCells[0].y);
      this.reachable = r.reachable;
      this.reachableIndex = r.index;
    }
    this.reachable[0].used = true;
    for (let i = 1; i < TANKS;) {
      const k = Math.floor(rng.random() * this.reachable.length);
      if (!this.reachable[k].used) {
        spawnCells[i] = { x: this.reachable[k].x, y: this.reachable[k].y };
        this.reachable[k].used = true;
        i++;
      }
    }
    for (const cell of this.reachable) cell.used = false;

    this.walls = buildWallSegments(this.maze, this.scale);
    this.wallHalfT = Math.floor(this.scale / 16);
    this.wallGrid = new WallGrid(this.walls, this.wallHalfT, this.scale);

    // Fresh tanks and a fresh AI every round; only the score carries over.
    this.tanks = [];
    this.bullets = [];
    this.bulletDepth = 0;
    for (let n = 0; n < TANKS; n++) this.tanks.push(new Tank(this, n, spawnCells[n], this.scale, rng));
    if (this.aiFactory) this.tanks[1].ai = this.aiFactory(this, this.tanks[1]);
    this.aliveCount = TANKS;

    const w = this.maze.length;
    const h = this.maze[0].length;
    this.distancesForMaze = Array.from({ length: w }, () => new Array(h).fill(null));
    for (const cell of this.reachable) {
      this.distancesForMaze[cell.x][cell.y] = calcDistances(this.maze, cell.x, cell.y);
    }
    this.tankFields = spawnCells.map((cell) => ({ x: cell.x, y: cell.y }));
    this.deadEnds = findDeadEnds(this.maze, this.reachable, C.MAXDEADENDPENALTY);
    this.events.push(["new_round", this.roundNumber]);
  }

  wallHit(px, py) {
    return this.wallGrid.hit(px, py);
  }

  /** distancesForMaze[fx][fy], or null for an unreachable or out-of-range cell. */
  distMap(fx, fy) {
    if (this.distancesForMaze && fx >= 0 && fx < this.distancesForMaze.length
        && fy >= 0 && fy < this.distancesForMaze[fx].length) {
      return this.distancesForMaze[fx][fy];
    }
    return null;
  }

  /** frame_53:1077-1090; always false for the bullet. */
  lockedControl() {
    return false;
  }

  /** frame_53:1091-1115, bullet only: the duel never spawns another weapon. */
  weaponReady(tank) {
    return tank.currentWeapon === "bullet" && tank.bulletsFired < this.settingsMaxBullets;
  }

  /** frame_53:1227-1258 */
  fireWeapon(tank) {
    if (tank.currentWeapon !== "bullet") return;
    this.bulletDepth += 1;
    const b = new Bullet(this, `bullet${this.bulletDepth}`, tank, this.scale);
    b.justCreated = true; // Flash: an attached clip's first enterFrame is next frame
    this.bullets.push(b);
    tank.bulletsFired += 1;
    this.events.push(["fire", tank.number]);
  }

  /** frame_53:1950-2013 */
  registerHit(owner, victim) {
    this.events.push(["hit", owner.number, victim.number]);
  }

  /** frame_53:900-911. A second death re-arms the window: that is a double KO. */
  destroyTank(number) {
    const tank = this.tanks[number];
    tank.alive = false;
    this.aliveCount -= 1;
    this.endCount = C.NUMBEROFFRAMESBEFOREEND;
    this.shake = Math.max(C.MAXSHAKE, this.shake + 7);
    this.events.push(["destroy", number]);
  }

  /** frame_53:2015-2050, local branch: the survivor scores. */
  assignPoints() {
    let winner = null;
    for (let i = 0; i < this.tanksCount; i++) {
      if (this.tanks[i].alive) {
        this.scores[i] += 1;
        winner = i;
      }
    }
    this.events.push(["round_end", winner]);
  }

  /** frame_53:1942-1949 */
  cleanUpBattle() {
    this.bullets = [];
  }

  /** Advance one frame (1/25 s) and return this frame's events. */
  step() {
    this.frame += 1;
    this.events = [];
    const rng = this.rng;

    // root onEnterFrame (frame_53:2366-2536)
    for (let i = 0; i < this.tanksCount; i++) {
      this.tankFields[i] = {
        x: Math.floor(this.tanks[i].x / this.scale),
        y: Math.floor(this.tanks[i].y / this.scale),
      };
    }
    if (!this.frozen) this.crateTimer -= 1;
    if (!this.frozen && this.crateTimer <= 0) {
      this.crateTimer = (C.CRATESPAWNTIMEBASE + rng.randrange(C.CRATESPAWNTIMERANDOM)
        + C.CRATESPAWNMAZESIZESCALE / this.reachable.length) * C.SETTINGS_CRATE_SPAWN_MODIFIER;
    }
    if (this.shake >= 0) this.shake -= 0.5;

    // Round state machine. It runs before the tanks move, and setupBattle()
    // may fire mid-frame: the new tanks then update later this same frame.
    if (this.aliveCount <= 1) {
      if (this.endCount >= 0) this.endCount -= 1;
      if (this.endCount === C.NUMBEROFFRAMESFROZEN) {
        this.frozen = true;
        this.assignPoints();
      }
      if (this.endCount === 0) {
        this.cleanUpBattle();
        this.resetCount = C.NUMBEROFFRAMESBEFORERESET;
      }
    }
    if (this.resetCount >= 0) this.resetCount -= 1;
    if (this.resetCount === 0) {
      this.endCount = C.NUMBEROFFRAMESBEFOREEND + C.NUMBEROFFRAMESFROZEN;
      this.frozen = false;
      this.setupBattle();
    }

    for (const tank of this.tanks) tank.update();

    for (const b of this.bullets.slice()) {
      if (b.justCreated) {
        b.justCreated = false;
        continue;
      }
      if (!b.removed) b.update();
    }
    this.bullets = this.bullets.filter((b) => !b.removed);
    return this.events;
  }
}
