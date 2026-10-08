/**
 * Laika, the original game's scripted AI, translated from
 * `tank_trouble_original/laika.py` (DefineSprite_186_tankTroubleAI).
 *
 * Three hooks run every frame inside the tank's own update:
 *   makeDecisionsAndUpdateGoal()   count down the goal's period; when it
 *                                  expires re-score every candidate goal
 *   decideActionsToAchieveGoal()   goal -> action stack
 *   setInputToDoActions()          action stack -> buttons
 * The action stack is LIFO. Laika is rebuilt every round; nothing carries over.
 * Numeric literals, float artefacts included, are those of the decompiled source.
 */

import * as C from "./constants.js";
import {
  followGradientPathWithDistancesAndDeadEnds, getShortestPathWithDistances,
} from "./maze.js";

const PI = 3.141592653589793;

/** Python's round(): nearest integer, ties to even. */
function pyRound(x) {
  const f = Math.floor(x);
  const diff = x - f;
  if (diff > 0.5) return f + 1;
  if (diff < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/** Heading, in the game's convention, of the vector (dx, dy); `fallback` for zero. */
function headingOf(dx, dy, fallback) {
  if (dx !== 0) {
    return dx > 0
      ? 90 + Math.atan(dy / dx) * 180 / PI
      : -90 + Math.atan(dy / dx) * 180 / PI;
  }
  if (dy > 0) return 180;
  if (dy < 0) return 0;
  return fallback;
}

export class LaikaAI {
  constructor(game, myTank) {
    this.game = game;
    this.myTank = myTank;
    const scale = game.scale;
    // AI :1355-1385; several depend on this round's cell size.
    this.AGGRESIVENESS = 0.5;
    this.COWARDNESS = 0.7000000000000001;
    this.GREEDY = 1;
    this.LONGESTPATHTOSHOOT = 7;
    this.LONGESTPATHTONOTHESITATETOSHOOT = 2;
    this.LONGESTPATHTORUN = 10;
    this.MAXSTUCKTIME = 1;
    this.stuckTime = 0;
    this.currentAggresiveness = this.AGGRESIVENESS;
    this.IDLEDRIVETOWARDENEMYPRIORITY = 0.1;
    this.IDLEDRIVEPRIORITY = 0.1;
    this.MAXCLOSESTCELLDISTANCE = 2;
    this.MAXCLOSESTDISTANCE = scale * this.MAXCLOSESTCELLDISTANCE;
    this.MAXTIMETODODGEBULLET = 75;
    this.MAXDISTTODODGEBULLET = 4 * scale;
    this.MAXCELLDISTTODODGEBULLET = (this.MAXTIMETODODGEBULLET * C.BULLETSPEED) / 50;
    this.MAXCELLDISTTOGOFORCRATE = 10;
    this.goalId = 1;
    this.myGoal = { goal: "idle", priority: 0, period: 15, id: 0, updateContinuously: true };
    this.myActions = [];
  }

  /** AS2 random(n) */
  rand(n) {
    return Math.floor(this.game.rng.random() * n);
  }

  /** distancesForMaze[fx][fy][cx][cy]; missing is NaN, which compares false. */
  cellDist(fx, fy, cx, cy) {
    const dm = this.game.distMap(fx, fy);
    if (dm === null) return NaN;
    if (cx >= 0 && cx < dm.length && cy >= 0 && cy < dm[cx].length) {
      const v = dm[cx][cy];
      return v === null || v === undefined ? NaN : v;
    }
    return NaN;
  }

  /** AI :1-7 */
  updateGoal(temp) {
    if (this.myGoal.priority < temp.priority) this.myGoal = temp;
  }

  // ------------------------------------------------ ballistics

  /**
   * AI :165-229. Walk a straight line; on the first wall, bounce and report.
   * Returns {x, y, xSpeed, ySpeed, t} or null.
   */
  checkPathForCollision(x, y, xSpeed, ySpeed, hitCheckInterval, maxtime, lifetime) {
    const g = this.game;
    lifetime = Math.min(maxtime, lifetime);
    let t = 0;
    while (lifetime > 0) {
      for (let i = 0; i < hitCheckInterval; i++) {
        const prevX = x;
        const prevY = y;
        x += xSpeed;
        y += ySpeed;
        if (g.wallHit(x, y)) {
          const hitXInv = g.wallHit(prevX - xSpeed, prevY + ySpeed);
          const hitYInv = g.wallHit(prevX + xSpeed, prevY - ySpeed);
          if (hitXInv && !hitYInv) ySpeed = -ySpeed;
          else if (hitYInv && !hitXInv) xSpeed = -xSpeed;
          else { xSpeed = -xSpeed; ySpeed = -ySpeed; }
          return { x: prevX + xSpeed, y: prevY + ySpeed, xSpeed, ySpeed, t };
        }
      }
      lifetime -= 1;
      t += 1;
    }
    return null;
  }

  /**
   * AI :230-336. Fly one bullet from the muzzle at `angle`: HIT, SUICIDE or
   * NOTHING. Deliberately coarse — one substep per frame, a third of a real
   * lifetime — Laika aims with a worse model of ballistics than the engine.
   */
  checkBulletPath(angle) {
    const g = this.game;
    const scale = g.scale;
    const my = this.myTank;
    const rad = ((angle - 90) * PI) / 180;
    let x = my.x + Math.cos(rad) * scale * 4.5 / 16;
    let y = my.y + Math.sin(rad) * scale * 4.5 / 16;
    let xs = Math.cos(rad) * C.BULLETSPEED * (scale / 50);
    let ys = Math.sin(rad) * C.BULLETSPEED * (scale / 50);
    let life = C.BULLETLIFETIME / 3; // fractional, as in the source
    let deadly = C.BULLETDEADLY;
    let closest = C.MOVIEWIDTH + C.MOVIEHEIGHT;
    while (life > 0) {
      const prevX = x;
      const prevY = y;
      x += xs;
      y += ys;
      if (g.wallHit(x, y)) {
        const hitXInv = g.wallHit(prevX - xs, prevY + ys);
        const hitYInv = g.wallHit(prevX + xs, prevY - ys);
        if (hitXInv && !hitYInv) ys = -ys;
        else if (hitYInv && !hitXInv) xs = -xs;
        else { xs = -xs; ys = -ys; }
        x = prevX + xs;
        y = prevY + ys;
      }
      if (deadly === 0) {
        for (let i = 0; i < g.tanksCount; i++) {
          const tank = g.tanks[i];
          if (tank.alive && tank.pointInBbox(x, y)) {
            if (tank.pointInShape(x, y)) {
              return { result: tank === my ? "SUICIDE" : "HIT", time: C.BULLETLIFETIME / 3 - life };
            }
          } else if (tank.alive && tank !== my) {
            // Manhattan, and only near that tank in maze terms.
            const d = Math.abs(tank.x - x) + Math.abs(tank.y - y);
            if (d < this.MAXCLOSESTDISTANCE) {
              const tf = g.tankFields[i];
              if (this.cellDist(tf.x, tf.y, Math.floor(x / scale), Math.floor(y / scale))
                  <= this.MAXCLOSESTCELLDISTANCE && d < closest) {
                closest = d;
              }
            }
          }
        }
      }
      if (deadly > 0) deadly -= 1;
      life -= 1;
    }
    return { result: "NOTHING", time: C.BULLETLIFETIME / 3, closest };
  }

  /** A goal object for a bullet passing within `dist` at parameter t. */
  dodgeGoal(b, cx, cy, dist, t, dirX, dirY, maxTime, maxDist) {
    const goal = {
      goal: "dodgeBullet", x: b.x, y: b.y, closest: { x: cx, y: cy }, dist, t,
      dir: { x: dirX, y: dirY }, maxTime, maxDist,
      period: 10, priority: 1, updateContinuously: false, id: this.goalId,
    };
    this.goalId += 1;
    return goal;
  }

  /** Straight-line wall check between two points, as the source does it. */
  clearLine(fromX, fromY, toX, toY) {
    const dx = toX - fromX;
    const dy = toY - fromY;
    const d = Math.sqrt(dx * dx + dy * dy);
    if (!(d > 0)) return true;
    return this.checkPathForCollision(fromX, fromY, dx / d, dy / d, 1, Math.ceil(d), Math.ceil(d)) === null;
  }

  /**
   * AI :8-88. For each bullet near me in maze terms, find its closest approach
   * — directly and after its next bounce — and raise a dodge goal if that
   * approach is near and unobstructed. Ownership is not checked: Laika dodges
   * its own shots.
   */
  dodgeTrajectories(fieldx, fieldy, bullets, maxTimeToDodge, maxDistToDodge,
    maxCellDistToDodge, hitCheckInterval, checkBounce) {
    const scale = this.game.scale;
    const my = this.myTank;
    let bestDist = maxDistToDodge;
    let result = { priority: 0 };
    for (const b of bullets) {
      const bx = b.x;
      const by = b.y;
      if (!(this.cellDist(fieldx, fieldy, Math.floor(bx / scale), Math.floor(by / scale))
          <= maxCellDistToDodge)) continue;
      let x2 = b.x + b.xSpeed * hitCheckInterval;
      let y2 = b.y + b.ySpeed * hitCheckInterval;
      const tx = my.x;
      const ty = my.y;
      let segSq = (x2 - bx) * (x2 - bx) + (y2 - by) * (y2 - by);
      let t = segSq ? ((tx - bx) * (x2 - bx) + (ty - by) * (y2 - by)) / segSq : 0.0;
      if (-1 < t && t < maxTimeToDodge) {
        const cx = bx + t * (x2 - bx);
        const cy = by + t * (y2 - by);
        const dx = tx - cx;
        const dy = ty - cy;
        const dist = Math.sqrt(dx * dx + dy * dy);
        let col = dist > 0
          ? this.checkPathForCollision(cx, cy, dx / dist, dy / dist, 1, Math.ceil(dist), Math.ceil(dist))
          : null;
        if (col === null && dist < bestDist) {
          const dx2 = x2 - cx;
          const dy2 = y2 - cy;
          const d2 = Math.sqrt(dx2 * dx2 + dy2 * dy2);
          col = d2 > 0
            ? this.checkPathForCollision(cx, cy, dx2 / d2, dy2 / d2, 1, Math.ceil(d2), Math.ceil(d2))
            : null;
          if (col === null) {
            bestDist = Math.min(bestDist, dist);
            result = this.dodgeGoal(b, cx, cy, dist, t, x2 - bx, y2 - by, maxTimeToDodge, maxDistToDodge);
          }
        }
      }
      // The threat after the bullet's next bounce (AI :50-83).
      if (bestDist > scale / 4 && checkBounce) {
        const col5 = this.checkPathForCollision(bx, by, b.xSpeed, b.ySpeed, hitCheckInterval, 12, b.lifetime);
        if (col5 !== null) {
          const bx2 = col5.x;
          const by2 = col5.y;
          x2 = col5.x + col5.xSpeed * hitCheckInterval;
          y2 = col5.y + col5.ySpeed * hitCheckInterval;
          segSq = (x2 - bx2) * (x2 - bx2) + (y2 - by2) * (y2 - by2);
          t = segSq ? ((tx - bx2) * (x2 - bx2) + (ty - by2) * (y2 - by2)) / segSq : 0.0;
          if (0 < t && t < maxTimeToDodge - col5.t) {
            const cx = bx2 + t * (x2 - bx2);
            const cy = by2 + t * (y2 - by2);
            const dx = tx - cx;
            const dy = ty - cy;
            const dist = Math.sqrt(dx * dx + dy * dy);
            let col = dist > 0
              ? this.checkPathForCollision(cx, cy, dx / dist, dy / dist, 1, Math.ceil(dist), Math.ceil(dist))
              : null;
            if (col === null && dist < bestDist) {
              const dx2 = cx - bx2;
              const dy2 = cy - by2;
              const d2 = Math.sqrt(dx2 * dx2 + dy2 * dy2);
              col = d2 > 0
                ? this.checkPathForCollision(bx2, by2, dx2 / d2, dy2 / d2, 1, Math.ceil(d2), Math.ceil(d2))
                : null;
              if (col === null) {
                bestDist = Math.min(bestDist, dist);
                result = this.dodgeGoal(b, cx, cy, dist, t + col5.t, x2 - bx2, y2 - by2,
                  maxTimeToDodge, maxDistToDodge);
              }
            }
          }
        }
      }
    }
    return result;
  }

  /** AI :89-164, bullet branch: fire back while dodging if the barrel is on. */
  tryToRetaliate() {
    const g = this.game;
    const my = this.myTank;
    if (this.currentAggresiveness < this.AGGRESIVENESS / 2) return;
    if (my.currentWeapon !== "bullet" || my.bulletsFired >= g.settingsMaxBullets) return;
    let found = false;
    let closest = C.MOVIEWIDTH + C.MOVIEHEIGHT;
    const res = this.checkBulletPath(my.rotation);
    if (res.result === "HIT") found = true;
    else if (res.result === "NOTHING" && res.closest < closest) closest = res.closest;
    if (found || closest < this.MAXCLOSESTDISTANCE / 2) {
      this.myActions.push({ action: "fireWeapon", delay: 1 });
      this.currentAggresiveness = Math.max(0, this.currentAggresiveness - 0.2);
    }
  }

  /**
   * AI :337-375. LIFO: push driveToField from the far end down to path[1],
   * then driveToPos(path[0]) last so it runs first.
   */
  pushActionsToFollowPath(path) {
    const scale = this.game.scale;
    for (let i = path.length - 1; i > 0; i--) {
      this.myActions.push({ action: "driveToField", x: path[i].x, y: path[i].y });
    }
    if (path.length) {
      this.myActions.push({
        action: "driveToPos",
        x: (path[0].x + 0.5) * scale,
        y: (path[0].y + 0.5) * scale,
        canReverse: path.length <= 2,
      });
    }
  }

  // ------------------------------------------------ goal selection

  /** AI :376-721. Returns true when the action stack must be rebuilt. */
  makeDecisionsAndUpdateGoal() {
    const g = this.game;
    const scale = g.scale;
    const my = this.myTank;

    if (this.myGoal.period > 0) {
      this.myGoal.period -= 1;
      return this.myGoal.updateContinuously;
    }

    this.myGoal.priority *= 0.9000000000000002;
    const oldGoal = this.myGoal;
    const fx = Math.floor(my.x / scale);
    const fy = Math.floor(my.y / scale);

    // goForCrate (AI :387-414): no crates in the duel, so the scan is empty
    // and the zero-priority goal it offers never wins.

    // dodgeBullet (AI :415-424)
    this.updateGoal(this.dodgeTrajectories(
      fx, fy, g.bullets, this.MAXTIMETODODGEBULLET, this.MAXDISTTODODGEBULLET,
      this.MAXCELLDISTTODODGEBULLET, C.BULLETHITCHECKINTERVALS, true));

    // shootAfter (AI :526-616)
    if (my.currentWeapon === "bullet" && my.bulletsFired < g.settingsMaxBullets) {
      for (let i = 0; i < g.tanksCount; i++) {
        const t = g.tanks[i];
        if (!(t.alive && t !== my)) continue;
        const dm = g.distMap(fx, fy);
        if (dm === null) continue;
        const path = getShortestPathWithDistances(
          g.maze, dm, fx, fy, g.tankFields[i].x, g.tankFields[i].y);
        if (path.length < this.LONGESTPATHTOSHOOT) {
          const priority = path.length <= this.LONGESTPATHTONOTHESITATETOSHOOT
            ? 1
            : ((this.LONGESTPATHTOSHOOT - path.length) / this.LONGESTPATHTOSHOOT)
              * this.currentAggresiveness;
          const goal = {
            goal: "shootAfter", target: t, period: 10, priority,
            updateContinuously: false, id: this.goalId,
          };
          this.goalId += 1;
          this.updateGoal(goal);
        }
      }
    }

    // runAway (AI :617-662), once the magazine is empty
    if (g.aliveCount > 1 && my.currentWeapon === "bullet"
        && my.bulletsFired === g.settingsMaxBullets) {
      const w = g.maze.length;
      const h = g.maze[0].length;
      // The source's off-by-one: (W-1) x (H-1); the missing row and column read NaN.
      const summed = Array.from({ length: w - 1 }, () => new Array(h - 1).fill(0.0));
      for (let i = 0; i < g.tanksCount; i++) {
        const t = g.tanks[i];
        if (t.alive && t !== my && t.bulletsFired !== g.settingsMaxBullets) {
          const dm = g.distMap(g.tankFields[i].x, g.tankFields[i].y);
          for (let xx = 0; xx < w - 1; xx++) {
            for (let yy = 0; yy < h - 1; yy++) {
              if (dm === null || dm[xx][yy] === null) summed[xx][yy] = NaN;
              else summed[xx][yy] += dm[xx][yy];
            }
          }
        }
      }
      const here = fx < w - 1 && fy < h - 1 ? summed[fx][fy] : NaN;
      if (here < this.LONGESTPATHTORUN) {
        const goal = {
          goal: "runAway", dist: summed, period: 10,
          priority: ((this.LONGESTPATHTORUN - here) / this.LONGESTPATHTORUN) * this.COWARDNESS
            * (my.bulletsFired / g.settingsMaxBullets),
          updateContinuously: false, id: this.goalId,
        };
        this.goalId += 1;
        this.updateGoal(goal);
      }
    }

    // backAway (AI :663-672)
    this.stuckTime = my.hitSomething ? Math.min(this.stuckTime + 1, this.MAXSTUCKTIME) : 0;
    this.updateGoal({
      goal: "backAway", period: 5, priority: this.stuckTime / (this.MAXSTUCKTIME - 0.1),
      updateContinuously: false, id: this.goalId++,
    });

    // Idle pursuit (AI :673-685)
    if (g.aliveCount > 1) {
      let k = this.rand(g.tanksCount);
      let guard = 0;
      while ((g.tanks[k] === my || !g.tanks[k].alive) && guard < 1000) {
        k = this.rand(g.tanksCount);
        guard += 1;
      }
      if (g.tanks[k] !== my) {
        this.updateGoal({
          goal: "driveTo", period: 10, priority: this.IDLEDRIVETOWARDENEMYPRIORITY,
          x: g.tankFields[k].x, y: g.tankFields[k].y,
          updateContinuously: false, id: this.goalId++,
        });
      }
    }

    // Goal switch (AI :686-720)
    if (oldGoal.id !== this.myGoal.id) {
      if (this.myGoal.goal === "shootAfter") {
        this.currentAggresiveness = Math.max(0, this.currentAggresiveness - 0.2);
      }
      return true;
    }
    this.currentAggresiveness = Math.min(
      this.AGGRESIVENESS, this.currentAggresiveness + this.AGGRESIVENESS / 50);
    return this.myGoal.updateContinuously;
  }

  // ------------------------------------------------ goal -> actions

  /** AI :722-1051 */
  decideActionsToAchieveGoal() {
    const g = this.game;
    const scale = g.scale;
    const my = this.myTank;
    this.myActions = [];
    const fx = Math.floor(my.x / scale);
    const fy = Math.floor(my.y / scale);
    const goal = this.myGoal;

    switch (goal.goal) {
      case "shootAfter": {
        let bestAngle = my.rotation;
        let found = false;
        let bestTime = C.BULLETLIFETIME;
        let closest = C.MOVIEWIDTH + C.MOVIEHEIGHT;
        let angle = my.rotation;
        // Direct line of sight (AI :735-767)
        const dx = goal.target.x - my.x;
        const dy = goal.target.y - my.y;
        const d = Math.sqrt(dx * dx + dy * dy);
        const col = d > 0
          ? this.checkPathForCollision(my.x, my.y, dx / d, dy / d, 1, Math.ceil(d), Math.ceil(d))
          : null;
        if (col === null) {
          found = true;
          closest = 0;
          bestAngle = headingOf(dx, dy, angle);
        }
        if (!found) {
          // Probe +- turnSpeed * k^2 (AI :768-810)
          for (let k = 1; k < 4; k++) {
            const res = this.checkBulletPath(angle);
            if (res.result === "HIT") {
              found = true;
              if (res.time < bestTime) {
                bestTime = res.time;
                closest = 0;
                bestAngle = angle;
              }
            } else if (res.result === "NOTHING" && !found && res.closest < closest) {
              closest = res.closest;
              bestAngle = angle;
            }
            if (g.rng.random() < 0.5) angle += my.turnSpeed * k * k;
            else angle -= my.turnSpeed * k * k;
            if (angle < -180) angle = 360 + angle;
            if (angle > 180) angle -= 360;
          }
        }
        if (found || closest < this.MAXCLOSESTDISTANCE) {
          this.myActions.push({ action: "fireWeapon", delay: 5 });
          this.myActions.push({ action: "turnTo", angle: bestAngle });
        } else if (bestAngle !== my.rotation) {
          this.myActions.push({ action: "turnTo", angle: bestAngle });
        } else {
          let a = my.rotation + 180;
          if (a > 180) a -= 360;
          this.myActions.push({ action: "turnTo", angle: a });
        }
        break;
      }

      case "driveTo": {
        const dm = g.distMap(fx, fy);
        if (dm !== null) {
          this.pushActionsToFollowPath(getShortestPathWithDistances(g.maze, dm, fx, fy, goal.x, goal.y));
        }
        break;
      }

      case "runAway":
        this.pushActionsToFollowPath(
          followGradientPathWithDistancesAndDeadEnds(g.maze, goal.dist, g.deadEnds, fx, fy, 5));
        break;

      case "backAway": {
        // AI :909-951
        this.myActions.push({
          action: "driveToPos", x: (fx + 0.5) * scale, y: (fy + 0.5) * scale, canReverse: false,
        });
        const front = my.expandedHitCheck(my.hitPointsFront, 1.1);
        const rear = my.expandedHitCheck(my.hitPointsRear, 1.1);
        if (front) {
          if (rear) {
            const left = my.expandedHitCheck(my.hitPointsLeft, 1.3000000000000005);
            this.myActions.push({ action: "backupAndTurn", dist: 5, dir: left ? "left" : "right" });
          } else {
            this.myActions.push({ action: "backup", dist: 3 });
          }
        } else if (rear) {
          // `front` is already known false here, so the source's nested
          // re-check of the front probes always takes its else branch.
          this.myActions.push({ action: "forward", dist: 3 });
        } else {
          this.myActions.push({ action: "backup", dist: 3 });
        }
        break;
      }

      case "dodgeBullet": {
        // AI :952-1019
        const dm = g.distMap(Math.floor(goal.x / scale), Math.floor(goal.y / scale));
        const path = dm !== null
          ? followGradientPathWithDistancesAndDeadEnds(g.maze, dm, g.deadEnds, fx, fy, 5)
          : [];
        const closeCall = goal.t < goal.maxTime / 3 && goal.dist < goal.maxDist / 5;
        if (closeCall || path.length <= 1) {
          // Cornered or out of time: turn parallel to the trajectory (AI :962-995).
          const cur = my.rotation;
          const gd = goal.dir;
          let a = headingOf(gd.x, gd.y, cur);
          if (Math.abs(a - cur) > 90 && Math.abs(a - cur) < 270) {
            a += 180;
            if (a > 180) a -= 360;
          }
          a = pyRound(a / my.turnSpeed) * my.turnSpeed;
          this.myActions.push({ action: "turnTo", angle: a });
          if (goal.dist < scale / 4) {
            // Sidestep perpendicular to the trajectory (AI :996-1012).
            const dl = Math.sqrt(gd.x * gd.x + gd.y * gd.y);
            if (dl > 0) {
              const px = -gd.y / dl;
              const py = gd.x / dl;
              const p1 = { x: goal.closest.x + (px * scale) / 2, y: goal.closest.y + (py * scale) / 2 };
              const p2 = { x: goal.closest.x - (px * scale) / 2, y: goal.closest.y - (py * scale) / 2 };
              const d1 = Math.hypot(my.x - p1.x, my.y - p1.y);
              const d2 = Math.hypot(my.x - p2.x, my.y - p2.y);
              const p = d1 < d2 ? p1 : p2;
              this.myActions.push({ action: "driveToPos", x: p.x, y: p.y, canReverse: true });
            }
          }
        } else {
          this.pushActionsToFollowPath(path);
        }
        this.tryToRetaliate();
        break;
      }

      case "idle":
        this.myActions.push({ action: "idle" });
        break;

      default:
        break;
    }
  }

  // ------------------------------------------------ actions -> buttons

  /** AI :1052-1354 — pop and re-push unfinished actions, then drive by the top one. */
  setInputToDoActions() {
    const scale = this.game.scale;
    const my = this.myTank;
    const fx = Math.floor(my.x / scale);
    const fy = Math.floor(my.y / scale);

    // Part one (AI :1056-1113)
    let action = this.myActions.length ? this.myActions.pop() : null;
    if (action !== null) {
      switch (action.action) {
        case "driveToField":
          if (Math.abs(my.x - (action.x + 0.5) * scale) > scale / 3
              || Math.abs(my.y - (action.y + 0.5) * scale) > scale / 3) this.myActions.push(action);
          break;
        case "turnTo":
          if (Math.abs(my.rotation - action.angle) >= my.turnSpeed) this.myActions.push(action);
          break;
        case "fireWeapon":
          if (action.delay !== 0) {
            action.delay -= 1;
            this.myActions.push(action);
          }
          break;
        case "driveToPos":
          if (Math.abs(my.x - action.x) > scale / 4 || Math.abs(my.y - action.y) > scale / 4) {
            this.myActions.push(action);
          }
          break;
        case "forward":
        case "backup":
        case "backupAndTurn":
          if (action.dist !== 0) {
            action.dist -= 1;
            this.myActions.push(action);
          }
          break;
        case "idle":
          this.myActions.push(action);
          break;
        default:
          break;
      }
    }

    // Part two (AI :1114-1353)
    action = this.myActions.length ? this.myActions[this.myActions.length - 1] : null;
    if (action === null) {
      my.turnLeft = my.turnRight = false;
      my.forward = my.backup = my.fire = false;
      this.myGoal.period = 0;
      return;
    }

    switch (action.action) {
      case "driveToField": {
        const cur = my.rotation;
        let target;
        if (fx > action.x) target = -90;
        else if (fx < action.x) target = 90;
        else if (fy > action.y) target = 0;
        else if (fy < action.y) target = 180;
        else target = cur;
        this.turnToward(target, cur);
        my.forward = !(Math.abs(target - cur) > 90 && Math.abs(target - cur) < 270);
        my.backup = false;
        my.fire = false;
        break;
      }
      case "turnTo":
        this.turnToward(action.angle, my.rotation);
        my.forward = false;
        my.backup = false;
        my.fire = false;
        break;
      case "fireWeapon":
        my.turnLeft = my.turnRight = false;
        my.forward = my.backup = false;
        my.fire = true;
        break;
      case "driveToPos": {
        const cur = my.rotation;
        let reverse = false;
        let target = headingOf(action.x - my.x, action.y - my.y, cur);
        target = my.turnSpeed * pyRound(target / my.turnSpeed);
        if (action.canReverse && Math.abs(target - cur) > 90 && Math.abs(target - cur) < 270) {
          reverse = true;
          target += 180;
          if (target > 180) target -= 360;
        }
        // Turning with a dead band (AI :1268-1298)
        const gap = Math.abs(target - cur);
        if (target > cur) {
          if (gap > 180) {
            my.turnLeft = gap < 360 - my.turnSpeed;
            my.turnRight = false;
          } else {
            my.turnLeft = false;
            my.turnRight = gap > my.turnSpeed;
          }
        } else if (target < cur) {
          if (gap > 180) {
            my.turnLeft = false;
            my.turnRight = gap < 360 - my.turnSpeed;
          } else {
            my.turnLeft = gap > my.turnSpeed;
            my.turnRight = false;
          }
        } else {
          my.turnLeft = false;
          my.turnRight = false;
        }
        if (gap > 45 && gap < 315) {
          my.forward = false;
          my.backup = false;
        } else {
          my.forward = !reverse;
          my.backup = reverse;
        }
        my.fire = false;
        break;
      }
      case "forward":
        my.turnLeft = my.turnRight = false;
        my.forward = true;
        my.backup = false;
        my.fire = false;
        break;
      case "backup":
        my.turnLeft = my.turnRight = false;
        my.forward = false;
        my.backup = true;
        my.fire = false;
        break;
      case "backupAndTurn":
        my.turnLeft = action.dir === "left";
        my.turnRight = action.dir === "right";
        my.forward = false;
        my.backup = true;
        my.fire = false;
        break;
      case "idle":
        my.turnLeft = my.turnRight = false;
        my.forward = my.backup = my.fire = false;
        break;
      default:
        my.turnLeft = my.turnRight = false;
        my.forward = my.backup = my.fire = false;
        this.myGoal.period = 0;
        break;
    }
  }

  /** AI :1140-1216 — turn the nearer way, no dead band. */
  turnToward(target, cur) {
    const my = this.myTank;
    if (target > cur) {
      my.turnLeft = Math.abs(target - cur) > 180;
      my.turnRight = !my.turnLeft;
    } else if (target < cur) {
      my.turnRight = Math.abs(target - cur) > 180;
      my.turnLeft = !my.turnRight;
    } else {
      my.turnLeft = false;
      my.turnRight = false;
    }
  }
}

