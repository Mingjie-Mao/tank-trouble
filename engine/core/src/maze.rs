//! Maze generation, reachability, distances and wall geometry, from
//! `tank_trouble_original/maze.py` (frame_53 of the Flash source).
//!
//! A cell is [ground, bottom wall, left wall]; the outer boundary is always
//! closed. Out-of-range reads behave like AS2's `undefined`: an edge is closed
//! and a distance is NaN, which compares false.

use crate::rng::Rng;

/// The source's literal for a diagonal step.
const SQRT2: f64 = 1.4142135623730951;

#[derive(Clone, Debug)]
pub struct Maze {
    pub w: usize,
    pub h: usize,
    /// [bottom wall, left wall] per cell, index x * h + y.
    cells: Vec<[bool; 2]>,
}

impl Maze {
    /// frame_53:1-42. A (w+1) x (h+1) grid of random(4) is reduced to wall
    /// flags; the result can be disconnected, which the caller rerolls.
    pub fn generate(w: usize, h: usize, rng: &mut Rng) -> Maze {
        let mut temp = vec![0u32; (w + 1) * (h + 1)];
        for x in 0..=w {
            for y in 0..=h {
                temp[x * (h + 1) + y] = rng.randrange(4);
            }
        }
        let t = |x: usize, y: usize| temp[x * (h + 1) + y];
        let mut cells = Vec::with_capacity(w * h);
        for x in 0..w {
            for y in 0..h {
                let bottom = t(x, y + 1) == 2 || t(x + 1, y + 1) == 0;
                let left = t(x, y) == 1 || t(x, y + 1) == 3;
                cells.push([bottom, left]);
            }
        }
        Maze { w, h, cells }
    }

    fn inside(&self, x: i32, y: i32) -> bool {
        x >= 0 && (x as usize) < self.w && y >= 0 && (y as usize) < self.h
    }

    /// The cell's bottom edge is open; out of range counts as walled.
    pub fn h_open(&self, x: i32, y: i32) -> bool {
        self.inside(x, y) && !self.cells[x as usize * self.h + y as usize][0]
    }

    /// The cell's left edge is open; out of range counts as walled.
    pub fn v_open(&self, x: i32, y: i32) -> bool {
        self.inside(x, y) && !self.cells[x as usize * self.h + y as usize][1]
    }

    pub fn bottom_wall(&self, x: usize, y: usize) -> bool {
        self.cells[x * self.h + y][0]
    }

    pub fn left_wall(&self, x: usize, y: usize) -> bool {
        self.cells[x * self.h + y][1]
    }
}

/// A w x h grid of f64 with NaN for "unknown", read like an AS2 array.
#[derive(Clone, Debug)]
pub struct Grid {
    pub w: usize,
    pub h: usize,
    pub v: Vec<f64>,
}

impl Grid {
    pub fn new(w: usize, h: usize, fill: f64) -> Grid {
        Grid { w, h, v: vec![fill; w * h] }
    }

    /// The value at (x, y), or NaN out of range.
    pub fn at(&self, x: i32, y: i32) -> f64 {
        if x >= 0 && (x as usize) < self.w && y >= 0 && (y as usize) < self.h {
            self.v[x as usize * self.h + y as usize]
        } else {
            f64::NAN
        }
    }

    pub fn set(&mut self, x: usize, y: usize, value: f64) {
        self.v[x * self.h + y] = value;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cell {
    pub x: i32,
    pub y: i32,
}

/// frame_53:43-96. Stack DFS; push order left, right, up, down fixes the order.
pub fn calc_reachable(maze: &Maze, sx: i32, sy: i32) -> Vec<Cell> {
    let (w, h) = (maze.w as i32, maze.h as i32);
    let mut visited = vec![false; maze.w * maze.h];
    let idx = |x: i32, y: i32| (x * h + y) as usize;
    let mut out = Vec::new();
    let mut stack = vec![(sx, sy)];
    while let Some((cx, cy)) = stack.pop() {
        out.push(Cell { x: cx, y: cy });
        visited[idx(cx, cy)] = true;
        let mut push = |x: i32, y: i32, stack: &mut Vec<(i32, i32)>| {
            if !visited[idx(x, y)] {
                visited[idx(x, y)] = true;
                stack.push((x, y));
            }
        };
        if maze.v_open(cx, cy) && cx > 0 {
            push(cx - 1, cy, &mut stack);
        }
        if maze.v_open(cx + 1, cy) && cx < w - 1 {
            push(cx + 1, cy, &mut stack);
        }
        if maze.h_open(cx, cy - 1) && cy > 0 {
            push(cx, cy - 1, &mut stack);
        }
        if maze.h_open(cx, cy) && cy < h - 1 {
            push(cx, cy + 1, &mut stack);
        }
    }
    out
}

/// frame_53:97-170. NaN = unreachable, 0 = ordinary, 1..max = inside a
/// dead-end corridor (lower is deeper).
pub fn find_dead_ends(maze: &Maze, reachable: &[Cell], max_penalty: i32) -> Grid {
    let (w, h) = (maze.w as i32, maze.h as i32);
    let mut de: Vec<Option<i32>> = vec![None; maze.w * maze.h];
    let idx = |x: i32, y: i32| (x * h + y) as usize;
    let mut stack = Vec::new();
    for c in reachable {
        stack.push((c.x, c.y));
        de[idx(c.x, c.y)] = Some(0);
    }
    // AS2 truthiness: undefined and 0 are both false.
    let val = |de: &Vec<Option<i32>>, x: i32, y: i32| -> Option<i32> {
        if x >= 0 && x < w && y >= 0 && y < h { de[idx(x, y)] } else { None }
    };
    let truthy = |v: Option<i32>| matches!(v, Some(n) if n != 0);
    while let Some((cx, cy)) = stack.pop() {
        if truthy(de[idx(cx, cy)]) {
            continue;
        }
        let mut next = (0, 0);
        let mut open = 0;
        let mut penalty = max_penalty;
        let side = |ok: bool, nx: i32, ny: i32, next: &mut (i32, i32), open: &mut i32, penalty: &mut i32| {
            if !ok {
                return;
            }
            let v = val(&de, nx, ny);
            if !truthy(v) {
                *next = (nx, ny);
                *open += 1;
            } else {
                *penalty = 1.max((v.unwrap() - 1).min(*penalty));
            }
        };
        side(maze.v_open(cx, cy) && cx > 0, cx - 1, cy, &mut next, &mut open, &mut penalty);
        side(maze.v_open(cx + 1, cy) && cx < w - 1, cx + 1, cy, &mut next, &mut open, &mut penalty);
        side(maze.h_open(cx, cy - 1) && cy > 0, cx, cy - 1, &mut next, &mut open, &mut penalty);
        side(maze.h_open(cx, cy) && cy < h - 1, cx, cy + 1, &mut next, &mut open, &mut penalty);
        if open == 1 {
            de[idx(cx, cy)] = Some(penalty);
            stack.push(next);
        }
        if open == 0 {
            de[idx(cx, cy)] = Some(penalty);
        }
    }
    let mut out = Grid::new(maze.w, maze.h, f64::NAN);
    for (i, v) in de.iter().enumerate() {
        if let Some(n) = v {
            out.v[i] = *n as f64;
        }
    }
    out
}

/// The four diagonal moves, each with the openings it needs, in the
/// source's check order: down-left, down-right, up-left, up-right.
fn diagonals(maze: &Maze, cx: i32, cy: i32) -> [(bool, i32, i32); 4] {
    let (w, h) = (maze.w as i32, maze.h as i32);
    [
        (maze.h_open(cx, cy) && maze.v_open(cx, cy) && maze.h_open(cx - 1, cy)
            && maze.v_open(cx, cy + 1) && cx > 0 && cy < h - 1, cx - 1, cy + 1),
        (maze.h_open(cx, cy) && maze.v_open(cx + 1, cy) && maze.h_open(cx + 1, cy)
            && maze.v_open(cx + 1, cy + 1) && cx < w - 1 && cy < h - 1, cx + 1, cy + 1),
        (maze.v_open(cx, cy) && maze.h_open(cx, cy - 1) && maze.v_open(cx, cy - 1)
            && maze.h_open(cx - 1, cy - 1) && cx > 0 && cy > 0, cx - 1, cy - 1),
        (maze.v_open(cx + 1, cy) && maze.h_open(cx, cy - 1) && maze.h_open(cx + 1, cy - 1)
            && maze.v_open(cx + 1, cy - 1) && cx < w - 1 && cy > 0, cx + 1, cy - 1),
    ]
}

/// The four orthogonal moves in the source's order: left, right, up, down.
fn orthogonals(maze: &Maze, cx: i32, cy: i32) -> [(bool, i32, i32); 4] {
    let (w, h) = (maze.w as i32, maze.h as i32);
    [
        (maze.v_open(cx, cy) && cx > 0, cx - 1, cy),
        (maze.v_open(cx + 1, cy) && cx < w - 1, cx + 1, cy),
        (maze.h_open(cx, cy - 1) && cy > 0, cx, cy - 1),
        (maze.h_open(cx, cy) && cy < h - 1, cx, cy + 1),
    ]
}

/// frame_53:171-264. First-come FIFO flood: orthogonal steps cost 1,
/// diagonals sqrt(2), never relaxed — so neighbour order is part of the result.
pub fn calc_distances(maze: &Maze, sx: i32, sy: i32) -> Grid {
    let h = maze.h as i32;
    let mut dist = Grid::new(maze.w, maze.h, f64::NAN);
    let mut visited = vec![false; maze.w * maze.h];
    let idx = |x: i32, y: i32| (x * h + y) as usize;
    let mut queue = vec![(sx, sy)];
    let mut head = 0;
    dist.set(sx as usize, sy as usize, 0.0);
    while head < queue.len() {
        let (cx, cy) = queue[head];
        head += 1;
        visited[idx(cx, cy)] = true;
        let here = dist.at(cx, cy);
        let steps = orthogonals(maze, cx, cy).map(|(ok, x, y)| (ok, x, y, 1.0));
        let diag = diagonals(maze, cx, cy).map(|(ok, x, y)| (ok, x, y, SQRT2));
        for (ok, nx, ny, cost) in steps.into_iter().chain(diag) {
            if ok && !visited[idx(nx, ny)] {
                visited[idx(nx, ny)] = true;
                dist.set(nx as usize, ny as usize, here + cost);
                queue.push((nx, ny));
            }
        }
    }
    dist
}

/// frame_53:270-338. Walk downhill from the end cell back to the start;
/// start-adjacent first, end last, start excluded. Diagonals are checked
/// before orthogonals. A do-while in the source, so start == end still
/// yields one element; a step cap stands in for the source's reliance on
/// connectivity.
pub fn shortest_path(maze: &Maze, distances: &Grid, sx: i32, sy: i32, ex: i32, ey: i32) -> Vec<Cell> {
    let mut path = Vec::new();
    let (mut cx, mut cy) = (ex, ey);
    let mut best = distances.at(cx, cy);
    let (mut nx, mut ny) = (ex, ey);
    let mut safety = (maze.w * maze.h * 4 + 8) as i64;
    loop {
        path.push(Cell { x: cx, y: cy });
        for (ok, x, y) in diagonals(maze, cx, cy).into_iter().chain(orthogonals(maze, cx, cy)) {
            if ok && distances.at(x, y) < best {
                best = distances.at(x, y);
                nx = x;
                ny = y;
            }
        }
        if (nx == cx && ny == cy) || safety <= 0 {
            break;
        }
        cx = nx;
        cy = ny;
        safety -= 1;
        if cx == sx && cy == sy {
            break;
        }
    }
    path.reverse();
    path
}

/// frame_53:506-586: climb distance minus dead-end depth (escape). A
/// do-while, so it always yields a cell; NaN never compares greater.
pub fn escape_path(maze: &Maze, distances: &Grid, dead_ends: &Grid, sx: i32, sy: i32, mut max_len: i32) -> Vec<Cell> {
    let value = |x: i32, y: i32| distances.at(x, y) - dead_ends.at(x, y);
    let mut path = Vec::new();
    let (mut cx, mut cy) = (sx, sy);
    let mut best = value(cx, cy);
    loop {
        let mut found = false;
        let (mut nx, mut ny) = (cx, cy);
        for (ok, x, y) in diagonals(maze, cx, cy).into_iter().chain(orthogonals(maze, cx, cy)) {
            if ok && value(x, y) > best {
                best = value(x, y);
                nx = x;
                ny = y;
                found = true;
            }
        }
        cx = nx;
        cy = ny;
        path.push(Cell { x: cx, y: cy });
        max_len -= 1;
        if !(found && max_len > 0) {
            break;
        }
    }
    path
}

/// Wall segments per drawMaze (frame_53:587-692): floor-rounded grid lines,
/// bottom and left walls per cell, and the four outer edges.
pub fn wall_segments(maze: &Maze, scale: f64) -> Vec<[f64; 4]> {
    let f = |v: f64| v.floor();
    let mut segs = Vec::new();
    for x in 0..maze.w {
        for y in 0..maze.h {
            let (xf, yf) = (x as f64, y as f64);
            if maze.bottom_wall(x, y) {
                segs.push([f(xf * scale), f((yf + 1.0) * scale), f((xf + 1.0) * scale), f((yf + 1.0) * scale)]);
            }
            if maze.left_wall(x, y) {
                segs.push([f(xf * scale), f(yf * scale), f(xf * scale), f((yf + 1.0) * scale)]);
            }
        }
    }
    let (wf, hf) = (maze.w as f64, maze.h as f64);
    for x in 0..maze.w {
        let xf = x as f64;
        segs.push([f(xf * scale), 0.0, f((xf + 1.0) * scale), 0.0]);
        segs.push([f(xf * scale), f(hf * scale), f((xf + 1.0) * scale), f(hf * scale)]);
    }
    for y in 0..maze.h {
        let yf = y as f64;
        segs.push([0.0, f((yf + 1.0) * scale), 0.0, f(yf * scale)]);
        segs.push([f(wf * scale), f((yf + 1.0) * scale), f(wf * scale), f(yf * scale)]);
    }
    segs
}

/// Point-vs-wall test (mazemc.hitTest with shapeFlag). An axis-aligned stroke
/// of half-thickness t with square caps is its segment's box grown by t, so
/// the test is point-in-rectangle, bucketed on a dense grid.
#[derive(Clone, Debug)]
pub struct WallGrid {
    /// 1 / bucket size: insertion and lookup bucket by the same monotone map.
    inv: f64,
    bx0: i64,
    by0: i64,
    nx: i64,
    ny: i64,
    /// Bucket k's rectangles are rects[start[k]..start[k + 1]].
    start: Vec<u32>,
    rects: Vec<[f64; 4]>,
}

impl WallGrid {
    pub fn new(walls: &[[f64; 4]], half_t: f64, bucket: f64) -> WallGrid {
        let rects: Vec<[f64; 4]> = walls
            .iter()
            .map(|[x1, y1, x2, y2]| {
                [x1.min(*x2) - half_t, y1.min(*y2) - half_t, x1.max(*x2) + half_t, y1.max(*y2) + half_t]
            })
            .collect();
        let inv = 1.0 / bucket;
        let b = |v: f64| (v * inv).floor() as i64;
        let (mut bx0, mut bx1, mut by0, mut by1) = (i64::MAX, i64::MIN, i64::MAX, i64::MIN);
        for r in &rects {
            bx0 = bx0.min(b(r[0]));
            bx1 = bx1.max(b(r[2]));
            by0 = by0.min(b(r[1]));
            by1 = by1.max(b(r[3]));
        }
        let (nx, ny) = (bx1 - bx0 + 1, by1 - by0 + 1);
        let mut buckets = vec![Vec::new(); (nx * ny) as usize];
        for r in &rects {
            for bx in b(r[0])..=b(r[2]) {
                for by in b(r[1])..=b(r[3]) {
                    buckets[((bx - bx0) * ny + (by - by0)) as usize].push(*r);
                }
            }
        }
        let mut start = vec![0u32];
        for bucket in &buckets {
            start.push(start.last().unwrap() + bucket.len() as u32);
        }
        WallGrid { inv, bx0, by0, nx, ny, start, rects: buckets.concat() }
    }

    /// Does any wall meet the box [x0, x1] x [y0, y1]?
    pub fn box_touches(&self, x0: f64, y0: f64, x1: f64, y1: f64) -> bool {
        let b = |v: f64, lo: i64, n: i64| ((v * self.inv).floor() as i64 - lo).clamp(0, n - 1);
        let (bx0, bx1) = (b(x0, self.bx0, self.nx), b(x1, self.bx0, self.nx));
        let (by0, by1) = (b(y0, self.by0, self.ny), b(y1, self.by0, self.ny));
        (bx0..=bx1).any(|bx| (by0..=by1).any(|by| {
            let k = (bx * self.ny + by) as usize;
            self.rects[self.start[k] as usize..self.start[k + 1] as usize]
                .iter()
                .any(|r| r[0] <= x1 && x0 <= r[2] && r[1] <= y1 && y0 <= r[3])
        }))
    }

    #[inline]
    pub fn hit(&self, px: f64, py: f64) -> bool {
        let bx = (px * self.inv).floor() as i64 - self.bx0;
        let by = (py * self.inv).floor() as i64 - self.by0;
        if bx < 0 || bx >= self.nx || by < 0 || by >= self.ny {
            return false;
        }
        let k = (bx * self.ny + by) as usize;
        self.rects[self.start[k] as usize..self.start[k + 1] as usize]
            .iter()
            .any(|r| r[0] <= px && px <= r[2] && r[1] <= py && py <= r[3])
    }
}
