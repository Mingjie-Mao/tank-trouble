/**
 * Maze generation, reachability, distances and wall geometry, translated from
 * `tank_trouble_original/maze.py` (itself from frame_53 of the Flash source).
 *
 * maze[x][y] = [ground, bottomWall, leftWall]; the outer boundary is drawn
 * separately and is always closed. Out-of-range reads behave like AS2's
 * `undefined`: an edge is closed and a distance is NaN, which compares false.
 */

const SQRT2 = 1.4142135623730951; // the source's literal

/** True when the cell's bottom edge is open; out of range counts as walled. */
export function hOpen(maze, x, y) {
  if (x >= 0 && x < maze.length && y >= 0 && y < maze[x].length) return maze[x][y][1] === 0;
  return false;
}

/** True when the cell's left edge is open; out of range counts as walled. */
export function vOpen(maze, x, y) {
  if (x >= 0 && x < maze.length && y >= 0 && y < maze[x].length) return maze[x][y][2] === 0;
  return false;
}

function dAt(distances, x, y) {
  if (x >= 0 && x < distances.length && y >= 0 && y < distances[x].length) {
    const v = distances[x][y];
    return v === null || v === undefined ? NaN : v;
  }
  return NaN;
}

/**
 * frame_53:1-42. A (xsize+1) x (ysize+1) grid of random(4) is reduced to wall
 * flags; the result can be disconnected, which the caller handles by rerolling.
 */
export function createMaze(xsize, ysize, rng) {
  const temp = [];
  for (let x = 0; x <= xsize; x++) {
    const col = [];
    for (let y = 0; y <= ysize; y++) col.push(rng.randrange(4));
    temp.push(col);
  }
  const maze = [];
  for (let x = 0; x < xsize; x++) {
    const col = [];
    for (let y = 0; y < ysize; y++) {
      const bottom = temp[x][y + 1] === 2 || temp[x + 1][y + 1] === 0;
      const left = temp[x][y] === 1 || temp[x][y + 1] === 3;
      col.push([1, bottom ? 1 : 0, left ? 1 : 0]);
    }
    maze.push(col);
  }
  return maze;
}

/** frame_53:43-96. Stack DFS; push order left, right, up, down fixes cell order. */
export function calcReachable(maze, startx, starty) {
  const w = maze.length;
  const h = maze[0].length;
  const index = Array.from({ length: w }, () => new Array(h).fill(null));
  const visited = new Uint8Array(w * h);
  const out = [];
  const stack = [[startx, starty]];
  const push = (x, y) => {
    if (!visited[x * h + y]) {
      visited[x * h + y] = 1;
      stack.push([x, y]);
    }
  };
  while (stack.length) {
    const [cx, cy] = stack.pop();
    index[cx][cy] = out.length;
    out.push({ x: cx, y: cy, used: false });
    visited[cx * h + cy] = 1;
    if (vOpen(maze, cx, cy) && cx > 0) push(cx - 1, cy);
    if (vOpen(maze, cx + 1, cy) && cx < w - 1) push(cx + 1, cy);
    if (hOpen(maze, cx, cy - 1) && cy > 0) push(cx, cy - 1);
    if (hOpen(maze, cx, cy) && cy < h - 1) push(cx, cy + 1);
  }
  return { reachable: out, index };
}

/**
 * frame_53:97-170. null = unreachable, 0 = ordinary, 1..maxPenalty = inside a
 * dead-end corridor (lower is deeper).
 */
export function findDeadEnds(maze, reachable, maxPenalty) {
  const w = maze.length;
  const h = maze[0].length;
  const de = Array.from({ length: w }, () => new Array(h).fill(null));
  const stack = [];
  for (const cell of reachable) {
    stack.push([cell.x, cell.y]);
    de[cell.x][cell.y] = 0;
  }
  const val = (x, y) => (x >= 0 && x < w && y >= 0 && y < h ? de[x][y] : null);
  while (stack.length) {
    const [cx, cy] = stack.pop();
    if (de[cx][cy]) continue; // AS2: !undefined and !0 are both true
    let next = null;
    let open = 0;
    let penalty = maxPenalty;
    if (vOpen(maze, cx, cy) && cx > 0 && !val(cx - 1, cy)) { next = [cx - 1, cy]; open++; }
    else if (vOpen(maze, cx, cy) && cx > 0) penalty = Math.max(1, Math.min(de[cx - 1][cy] - 1, penalty));
    if (vOpen(maze, cx + 1, cy) && cx < w - 1 && !val(cx + 1, cy)) { next = [cx + 1, cy]; open++; }
    else if (vOpen(maze, cx + 1, cy) && cx < w - 1) penalty = Math.max(1, Math.min(de[cx + 1][cy] - 1, penalty));
    if (hOpen(maze, cx, cy - 1) && cy > 0 && !val(cx, cy - 1)) { next = [cx, cy - 1]; open++; }
    else if (hOpen(maze, cx, cy - 1) && cy > 0) penalty = Math.max(1, Math.min(de[cx][cy - 1] - 1, penalty));
    if (hOpen(maze, cx, cy) && cy < h - 1 && !val(cx, cy + 1)) { next = [cx, cy + 1]; open++; }
    else if (hOpen(maze, cx, cy) && cy < h - 1) penalty = Math.max(1, Math.min(de[cx][cy + 1] - 1, penalty));
    if (open === 1) {
      de[cx][cy] = penalty;
      stack.push(next);
    }
    if (open === 0) de[cx][cy] = penalty;
  }
  return de;
}

/**
 * frame_53:171-264. First-come FIFO flood: four orthogonal steps at 1, four
 * diagonals at sqrt(2), never relaxed — so neighbour order is part of the
 * result: left, right, up, down, down-left, down-right, up-left, up-right.
 */
export function calcDistances(maze, startx, starty) {
  const w = maze.length;
  const h = maze[0].length;
  const dist = Array.from({ length: w }, () => new Array(h).fill(NaN));
  const visited = new Uint8Array(w * h);
  const queue = [[startx, starty]];
  let head = 0;
  dist[startx][starty] = 0.0;
  while (head < queue.length) {
    const [cx, cy] = queue[head++];
    visited[cx * h + cy] = 1;
    const tryAdd = (nx, ny, cost) => {
      if (!visited[nx * h + ny]) {
        visited[nx * h + ny] = 1;
        dist[nx][ny] = dist[cx][cy] + cost;
        queue.push([nx, ny]);
      }
    };
    if (vOpen(maze, cx, cy) && cx > 0) tryAdd(cx - 1, cy, 1);
    if (vOpen(maze, cx + 1, cy) && cx < w - 1) tryAdd(cx + 1, cy, 1);
    if (hOpen(maze, cx, cy - 1) && cy > 0) tryAdd(cx, cy - 1, 1);
    if (hOpen(maze, cx, cy) && cy < h - 1) tryAdd(cx, cy + 1, 1);
    if (hOpen(maze, cx, cy) && vOpen(maze, cx, cy) && hOpen(maze, cx - 1, cy)
        && vOpen(maze, cx, cy + 1) && cx > 0 && cy < h - 1) tryAdd(cx - 1, cy + 1, SQRT2);
    if (hOpen(maze, cx, cy) && vOpen(maze, cx + 1, cy) && hOpen(maze, cx + 1, cy)
        && vOpen(maze, cx + 1, cy + 1) && cx < w - 1 && cy < h - 1) tryAdd(cx + 1, cy + 1, SQRT2);
    if (vOpen(maze, cx, cy) && hOpen(maze, cx, cy - 1) && vOpen(maze, cx, cy - 1)
        && hOpen(maze, cx - 1, cy - 1) && cx > 0 && cy > 0) tryAdd(cx - 1, cy - 1, SQRT2);
    if (vOpen(maze, cx + 1, cy) && hOpen(maze, cx, cy - 1) && hOpen(maze, cx + 1, cy - 1)
        && vOpen(maze, cx + 1, cy - 1) && cx < w - 1 && cy > 0) tryAdd(cx + 1, cy - 1, SQRT2);
  }
  return dist;
}

/**
 * frame_53:270-338. Walk downhill from the end cell back to the start; returns
 * the path start-adjacent first, end last, start excluded. Check order: the
 * four diagonals, then the four orthogonals. A do-while in the source, so a
 * start equal to the end still yields one element; a step cap stands in for
 * the source's reliance on connectivity.
 */
export function getShortestPathWithDistances(maze, distances, startx, starty, endx, endy) {
  const w = maze.length;
  const h = maze[0].length;
  const path = [];
  let cx = endx;
  let cy = endy;
  let best = dAt(distances, cx, cy);
  let nx = endx;
  let ny = endy;
  let safety = w * h * 4 + 8;
  for (;;) {
    path.push({ x: cx, y: cy });
    const take = (x, y) => {
      if (dAt(distances, x, y) < best) { best = dAt(distances, x, y); nx = x; ny = y; }
    };
    if (hOpen(maze, cx, cy) && vOpen(maze, cx, cy) && hOpen(maze, cx - 1, cy)
        && vOpen(maze, cx, cy + 1) && cx > 0 && cy < h - 1) take(cx - 1, cy + 1);
    if (hOpen(maze, cx, cy) && vOpen(maze, cx + 1, cy) && hOpen(maze, cx + 1, cy)
        && vOpen(maze, cx + 1, cy + 1) && cx < w - 1 && cy < h - 1) take(cx + 1, cy + 1);
    if (vOpen(maze, cx, cy) && hOpen(maze, cx, cy - 1) && vOpen(maze, cx, cy - 1)
        && hOpen(maze, cx - 1, cy - 1) && cx > 0 && cy > 0) take(cx - 1, cy - 1);
    if (vOpen(maze, cx + 1, cy) && hOpen(maze, cx, cy - 1) && hOpen(maze, cx + 1, cy - 1)
        && vOpen(maze, cx + 1, cy - 1) && cx < w - 1 && cy > 0) take(cx + 1, cy - 1);
    if (vOpen(maze, cx, cy) && cx > 0) take(cx - 1, cy);
    if (vOpen(maze, cx + 1, cy) && cx < w - 1) take(cx + 1, cy);
    if (hOpen(maze, cx, cy - 1) && cy > 0) take(cx, cy - 1);
    if (hOpen(maze, cx, cy) && cy < h - 1) take(cx, cy + 1);
    if ((nx === cx && ny === cy) || safety <= 0) break;
    cx = nx;
    cy = ny;
    safety--;
    if (cx === startx && cy === starty) break;
  }
  path.reverse();
  return path;
}

/** frame_53:265-269 */
export function getShortestPath(maze, startx, starty, endx, endy) {
  return getShortestPathWithDistances(
    maze, calcDistances(maze, startx, starty), startx, starty, endx, endy);
}

/**
 * The shared body of followGradientPath*: climb `value` (escape), same check
 * order as the shortest-path walk; a do-while, so it always yields a cell.
 */
function gradientWalk(maze, value, startx, starty, maxLength) {
  const w = maze.length;
  const h = maze[0].length;
  const path = [];
  let cx = startx;
  let cy = starty;
  let best = value(cx, cy);
  for (;;) {
    let found = false;
    let nx = cx;
    let ny = cy;
    const take = (x, y) => {
      if (value(x, y) > best) { best = value(x, y); nx = x; ny = y; found = true; }
    };
    if (hOpen(maze, cx, cy) && vOpen(maze, cx, cy) && hOpen(maze, cx - 1, cy)
        && vOpen(maze, cx, cy + 1) && cx > 0 && cy < h - 1) take(cx - 1, cy + 1);
    if (hOpen(maze, cx, cy) && vOpen(maze, cx + 1, cy) && hOpen(maze, cx + 1, cy)
        && vOpen(maze, cx + 1, cy + 1) && cx < w - 1 && cy < h - 1) take(cx + 1, cy + 1);
    if (vOpen(maze, cx, cy) && hOpen(maze, cx, cy - 1) && vOpen(maze, cx, cy - 1)
        && hOpen(maze, cx - 1, cy - 1) && cx > 0 && cy > 0) take(cx - 1, cy - 1);
    if (vOpen(maze, cx + 1, cy) && hOpen(maze, cx, cy - 1) && hOpen(maze, cx + 1, cy - 1)
        && vOpen(maze, cx + 1, cy - 1) && cx < w - 1 && cy > 0) take(cx + 1, cy - 1);
    if (vOpen(maze, cx, cy) && cx > 0) take(cx - 1, cy);
    if (vOpen(maze, cx + 1, cy) && cx < w - 1) take(cx + 1, cy);
    if (hOpen(maze, cx, cy - 1) && cy > 0) take(cx, cy - 1);
    if (hOpen(maze, cx, cy) && cy < h - 1) take(cx, cy + 1);
    cx = nx;
    cy = ny;
    path.push({ x: cx, y: cy });
    maxLength--;
    if (!(found && maxLength > 0)) break;
  }
  return path;
}

/** frame_53:428-505 */
export function followGradientPathWithDistances(maze, distances, startx, starty, maxLength) {
  return gradientWalk(maze, (x, y) => dAt(distances, x, y), startx, starty, maxLength);
}

/** frame_53:506-586. value = distance - deadEnd; undefined makes it NaN. */
export function followGradientPathWithDistancesAndDeadEnds(
  maze, distances, deadEnds, startx, starty, maxLength,
) {
  const value = (x, y) => {
    const d = dAt(distances, x, y);
    let de = NaN;
    if (x >= 0 && x < deadEnds.length && y >= 0 && y < deadEnds[x].length) {
      const v = deadEnds[x][y];
      de = v === null || v === undefined ? NaN : v;
    }
    return d - de;
  };
  return gradientWalk(maze, value, startx, starty, maxLength);
}

/**
 * Wall segments per drawMaze (frame_53:587-692): floor-rounded grid lines,
 * bottom and left walls per cell, and the four outer edges always.
 */
export function buildWallSegments(maze, scale) {
  const w = maze.length;
  const h = maze[0].length;
  const fl = Math.floor;
  const segs = [];
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      if (maze[x][y][1] !== 0) segs.push([fl(x * scale), fl((y + 1) * scale), fl((x + 1) * scale), fl((y + 1) * scale)]);
      if (maze[x][y][2] !== 0) segs.push([fl(x * scale), fl(y * scale), fl(x * scale), fl((y + 1) * scale)]);
    }
  }
  for (let x = 0; x < w; x++) {
    segs.push([fl(x * scale), 0, fl((x + 1) * scale), 0]);
    segs.push([fl(x * scale), fl(h * scale), fl((x + 1) * scale), fl(h * scale)]);
  }
  for (let y = 0; y < h; y++) {
    segs.push([0, fl((y + 1) * scale), 0, fl(y * scale)]);
    segs.push([fl(w * scale), fl((y + 1) * scale), fl(w * scale), fl(y * scale)]);
  }
  return segs;
}

/**
 * Point-vs-wall test (mazemc.hitTest with shapeFlag). An axis-aligned stroke
 * of half-thickness t with square caps is exactly its segment's bounding box
 * grown by t, so the test is point-in-rectangle. Rectangles are bucketed on a
 * dense integer grid; the answer does not depend on the bucketing.
 */
export class WallGrid {
  constructor(walls, halfT, bucketSize = 64.0) {
    const rects = walls.map(([x1, y1, x2, y2]) => [
      Math.min(x1, x2) - halfT, Math.min(y1, y2) - halfT,
      Math.max(x1, x2) + halfT, Math.max(y1, y2) + halfT,
    ]);
    let bx0 = Infinity;
    let bx1 = -Infinity;
    let by0 = Infinity;
    let by1 = -Infinity;
    for (const r of rects) {
      bx0 = Math.min(bx0, Math.floor(r[0] / bucketSize));
      bx1 = Math.max(bx1, Math.floor(r[2] / bucketSize));
      by0 = Math.min(by0, Math.floor(r[1] / bucketSize));
      by1 = Math.max(by1, Math.floor(r[3] / bucketSize));
    }
    this.cell = bucketSize;
    this.bx0 = bx0;
    this.by0 = by0;
    this.nx = bx1 - bx0 + 1;
    this.ny = by1 - by0 + 1;
    const lists = Array.from({ length: this.nx * this.ny }, () => []);
    for (const r of rects) {
      for (let bx = Math.floor(r[0] / bucketSize); bx <= Math.floor(r[2] / bucketSize); bx++) {
        for (let by = Math.floor(r[1] / bucketSize); by <= Math.floor(r[3] / bucketSize); by++) {
          lists[(bx - bx0) * this.ny + (by - by0)].push(...r);
        }
      }
    }
    this.buckets = lists.map((list) => (list.length ? Float64Array.from(list) : null));
  }

  hit(px, py) {
    const bx = Math.floor(px / this.cell) - this.bx0;
    const by = Math.floor(py / this.cell) - this.by0;
    if (bx < 0 || bx >= this.nx || by < 0 || by >= this.ny) return false;
    const r = this.buckets[bx * this.ny + by];
    if (r === null) return false;
    for (let i = 0; i < r.length; i += 4) {
      if (r[i] <= px && px <= r[i + 2] && r[i + 1] <= py && py <= r[i + 3]) return true;
    }
    return false;
  }
}
