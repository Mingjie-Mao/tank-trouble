//! The observation: 230 values designed from the game's mechanics, all facts a
//! player could read off the screen, every one with a plain meaning.
//!
//! A local 5x5 wall map instead of the whole maze; ballistics flown with the
//! engine's own substeps; what Laika is about to do (its own aiming model, its
//! shooting radius, its rhythm); a nine-move survival lookahead that includes
//! the shot Laika is about to take; and a tactics group: the game's own route
//! to the opponent, how near each barrel's shot passes and whether turning
//! helps, the bullets' summed threat, and how long the current move has run.

use crate::constants as c;
use crate::game::{norm_rot, Game, Tank};
use crate::aim::{laika_aim_hits, turns_to_hit};
use crate::sandbox::{sandbox, Opponent};

pub const SELF_DIM: usize = 8;
const MAP_RADIUS: i32 = 2;
const MAP_SPAN: i32 = 2 * MAP_RADIUS + 1;
/// Horizontal edges (MAP_SPAN + 1 lines x MAP_SPAN), vertical edges likewise, cell offset.
pub const LOCAL_DIM: usize = (2 * (MAP_SPAN + 1) * MAP_SPAN + 2) as usize;
pub const OPPONENT_DIM: usize = 13;
pub const BARREL_DIM: usize = 6;
pub const INTENT_DIM: usize = 7;
pub const BULLET_SLOTS: usize = 10;
pub const BULLET_DIM: usize = 10;
pub const DODGE_DIM: usize = 9;
pub const TIME_DIM: usize = 6;
pub const TACTIC_DIM: usize = 19;

pub const SELF_OFFSET: usize = 0;
pub const LOCAL_OFFSET: usize = SELF_OFFSET + SELF_DIM;
pub const OPPONENT_OFFSET: usize = LOCAL_OFFSET + LOCAL_DIM;
pub const BARREL_OFFSET: usize = OPPONENT_OFFSET + OPPONENT_DIM;
pub const INTENT_OFFSET: usize = BARREL_OFFSET + BARREL_DIM;
pub const BULLET_OFFSET: usize = INTENT_OFFSET + INTENT_DIM;
pub const DODGE_OFFSET: usize = BULLET_OFFSET + BULLET_SLOTS * BULLET_DIM;
pub const TIME_OFFSET: usize = DODGE_OFFSET + DODGE_DIM;
pub const TACTIC_OFFSET: usize = TIME_OFFSET + TIME_DIM;
pub const OBS_DIM: usize = TACTIC_OFFSET + TACTIC_DIM;
/// (map_start, map_len, bullet_start, slots, bullet_dim, dodge_start, self_hit)
pub const LAYOUT: [usize; 7] = [0, 0, BULLET_OFFSET, BULLET_SLOTS, BULLET_DIM, DODGE_OFFSET, SELF_HIT_INDEX];
/// A round both tanks survive this long is a draw.
pub const DUEL_FRAMES: u32 = 750;
/// "My current shot would hit me": the shield drops fire while it is set.
pub const SELF_HIT_INDEX: usize = BARREL_OFFSET + 1;

const SHOT_FRAMES: usize = (c::BULLETLIFETIME / 3) as usize;
const FORECAST_FRAMES: usize = 75;
const DODGE_HORIZON: usize = 24;
const MAX_PATH_CELLS: f64 = 60.0;
const MAX_SPEED_CELLS: f64 = 0.5;
const BULLET_RANGE_CELLS: f64 = 8.0;
const LAIKA_SHOOT_PATH: f64 = 7.0;
const LAIKA_ALWAYS_SHOOT_PATH: f64 = 2.0;
const SHOT_CLOCK: f64 = 50.0;

/// Discrete(18): throttle {back, none, forward} x turn {left, none, right} x fire.
pub fn candidate(action: usize) -> (usize, usize, bool) {
    (action / 6, (action / 2) % 3, action % 2 == 1)
}

/// What a seat has been doing lately.
#[derive(Clone, Debug, Default)]
pub struct SeatHistory {
    pub last: Option<usize>,
    /// Consecutive frames the current action has been held.
    pub held: u32,
    /// Consecutive frames without moving or turning.
    pub idle_streak: u32,
}

impl SeatHistory {
    pub fn record(&mut self, action: usize) {
        self.held = if self.last == Some(action) { self.held + 1 } else { 1 };
        self.last = Some(action);
        let (throttle, turn, _) = candidate(action);
        self.idle_streak = if throttle == 1 && turn == 1 { self.idle_streak + 1 } else { 0 };
    }
}

fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

/// A world vector in a tank's frame: +x ahead, +y to its left.
fn own_frame(rotation: f64, dx: f64, dy: f64) -> (f64, f64) {
    let (sin, cos) = ((rotation - 90.0) * c::DEG).sin_cos();
    (dx * cos + dy * sin, -dx * sin + dy * cos)
}

/// One bullet substep with the engine's ricochet rule.
#[inline]
fn substep(g: &Game, p: &mut [f64; 4]) {
    let (px, py) = (p[0], p[1]);
    p[0] += p[2];
    p[1] += p[3];
    if g.wall_hit(p[0], p[1]) {
        let x_inv = g.wall_hit(px - p[2], py + p[3]);
        let y_inv = g.wall_hit(px + p[2], py - p[3]);
        if x_inv && !y_inv {
            p[3] = -p[3];
        } else if y_inv && !x_inv {
            p[2] = -p[2];
        } else {
            p[2] = -p[2];
            p[3] = -p[3];
        }
        p[0] = px + p[2];
        p[1] = py + p[3];
    }
}

/// A shot fired now from `shooter`'s barrel, flown with the engine's
/// substeps: Some((victim, frames)) for the first tank it would hit.
fn engine_shot(g: &Game, shooter: usize) -> Option<(usize, usize)> {
    let s = &g.tanks[shooter];
    if !s.alive {
        return None;
    }
    let scale = g.scale();
    let rad = (s.rotation - 90.0) * c::DEG;
    let speed = c::BULLETSPEED / c::BULLETHITCHECKINTERVALS as f64 * (scale / 50.0);
    let mut p = [s.x + rad.cos() * scale * 4.5 / 16.0, s.y + rad.sin() * scale * 4.5 / 16.0, rad.cos() * speed, rad.sin() * speed];
    for frame in 1..=SHOT_FRAMES {
        for _ in 0..c::BULLETHITCHECKINTERVALS {
            substep(g, &mut p);
        }
        for (i, t) in g.tanks.iter().enumerate() {
            if t.alive && t.point_in_shape(p[0], p[1]) {
                return Some((i, frame));
            }
        }
    }
    None
}

/// A live bullet flown forward against a still tank: (hits, frame of hit or
/// closest approach, closest distance).
fn bullet_forecast(g: &Game, b: &crate::game::Bullet, target: &Tank) -> (bool, usize, f64) {
    let mut p = [b.x, b.y, b.x_speed, b.y_speed];
    let (mut best, mut best_frame) = (f64::INFINITY, 0);
    for frame in 1..=FORECAST_FRAMES.min(b.lifetime.max(0) as usize) {
        for _ in 0..c::BULLETHITCHECKINTERVALS {
            substep(g, &mut p);
        }
        if target.alive && target.point_in_shape(p[0], p[1]) {
            return (true, frame, 0.0);
        }
        let d = (p[0] - target.x).hypot(p[1] - target.y);
        if d < best {
            best = d;
            best_frame = frame;
        }
    }
    (false, best_frame, best)
}

/// Is the straight segment between two points free of walls?
fn line_of_sight(g: &Game, a: (f64, f64), b: (f64, f64)) -> bool {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let d = dx.hypot(dy);
    let steps = (d / (g.scale() / 8.0)).ceil().max(1.0) as i32;
    (1..steps).all(|k| {
        let t = k as f64 / steps as f64;
        !g.wall_hit(a.0 + dx * t, a.1 + dy * t)
    })
}

/// Will Laika fire within a few frames? Its own aiming model says its barrel
/// holds a hit on me, and it has a bullet and a released trigger.
pub fn laika_about_to_fire(g: &Game, laika: usize, me: usize) -> bool {
    let t = &g.tanks[laika];
    t.alive && g.weapon_ready(laika) && t.trigger_released && laika_aim_hits(g, laika, me, t.rotation)
}

/// Per-move survival for the nine no-fire moves over 24 frames, like the
/// baseline's, but the opponent also fires the shot its aim says it is about
/// to take. Positive: survives, scaled by clearance; negative: dies, less
/// negative the longer it lasted.
fn survival(g: &Game, me: usize, predicted_shot: bool, out: &mut [f32]) {
    let other = 1 - me;
    if g.bullets.is_empty() && !g.tanks[other].fire && !predicted_shot {
        out[..DODGE_DIM].fill(1.0);
        return;
    }
    if !g.tanks[me].alive {
        out[..DODGE_DIM].fill(-1.0);
        return;
    }
    let mut world = sandbox(g, me, Opponent::Hold, 0);
    let real = world.tanks[0].clone();
    world.tanks[0].alive = false;
    if predicted_shot {
        world.tanks[1].fire = true;
    }
    let mut frames: Vec<(bool, Vec<(f64, f64)>, Vec<(f64, f64)>, bool)> = Vec::with_capacity(DODGE_HORIZON);
    for _ in 0..DODGE_HORIZON {
        let frozen = world.frozen;
        world.step();
        let ended = world.frozen || world.events.iter().any(|e| matches!(e, crate::game::Event::RoundEnd(_)));
        let hits = if frozen { Vec::new() } else { world.hit_tests.clone() };
        frames.push((frozen, hits, world.bullets.iter().map(|b| (b.x, b.y)).collect(), ended));
        if ended {
            break;
        }
    }
    let grid = &world.round.grid;
    let inv = 1.0 / world.round.scale;
    for (mv, slot) in out.iter_mut().enumerate().take(DODGE_DIM) {
        let (throttle, turn) = (mv / 3, mv % 3);
        let mut hull = real.clone();
        hull.set_buttons(throttle == 2, throttle == 0, turn == 0, turn == 2, false);
        let mut clearance: f64 = 8.0;
        let mut value = None;
        for (f, (frozen, hits, clear, ended)) in frames.iter().enumerate() {
            if hits.iter().any(|&(x, y)| hull.point_in_shape(x, y)) {
                value = Some(-1.0 + 0.5 * (f as f64 / DODGE_HORIZON as f64));
                break;
            }
            if !frozen {
                hull.drive(grid);
            }
            for &(x, y) in clear {
                clearance = clearance.min((x - hull.x).hypot(y - hull.y) * inv);
            }
            if *ended {
                break;
            }
        }
        *slot = value.unwrap_or(clamp(clearance / 8.0, 0.0, 1.0)) as f32;
    }
}

/// A shot fired now from `shooter` with its barrel at `rotation`, flown with
/// the engine's substeps: (hits target, frame of hit or closest approach,
/// closest distance to the target's centre). Stops at the first tank it hits.
fn shot_pass(g: &Game, shooter: usize, target: usize, rotation: f64) -> (bool, usize, f64) {
    let (s, t) = (&g.tanks[shooter], &g.tanks[target]);
    let scale = g.scale();
    let rad = (rotation - 90.0) * c::DEG;
    let speed = c::BULLETSPEED / c::BULLETHITCHECKINTERVALS as f64 * (scale / 50.0);
    let mut p = [s.x + rad.cos() * scale * 4.5 / 16.0, s.y + rad.sin() * scale * 4.5 / 16.0, rad.cos() * speed, rad.sin() * speed];
    let (mut best, mut best_frame) = (f64::INFINITY, SHOT_FRAMES);
    for frame in 1..=SHOT_FRAMES {
        for _ in 0..c::BULLETHITCHECKINTERVALS {
            substep(g, &mut p);
        }
        if t.alive && t.point_in_shape(p[0], p[1]) {
            return (true, frame, 0.0);
        }
        if s.point_in_shape(p[0], p[1]) {
            break;
        }
        let d = (p[0] - t.x).hypot(p[1] - t.y);
        if d < best {
            best = d;
            best_frame = frame;
        }
    }
    (false, best_frame, best)
}

/// Open sides of a cell, of four.
fn exits(g: &Game, x: i32, y: i32) -> f32 {
    let m = &g.round.maze;
    let h = m.h as i32;
    let open = [m.h_open(x, y - 1), y < h - 1 && m.h_open(x, y), m.v_open(x, y), m.v_open(x + 1, y)];
    open.iter().filter(|&&o| o).count() as f32 / 4.0
}

/// Frames of turning to aim a shot are worth this many degrees each.
const AIM_PROBE_TURNS: f64 = 3.0;
const HOLD_CAP: f64 = 25.0;

/// What a player reads off the screen to move, aim and stay alive: where the game's own route goes next, how near each barrel's
/// shot passes (and whether turning helps), and the bullets' summed threat.
fn tactics(g: &Game, me: usize, ctx: &Context, out: &mut [f32]) {
    out[..TACTIC_DIM].fill(0.0);
    let other = 1 - me;
    let (my, them) = (&g.tanks[me], &g.tanks[other]);
    let scale = g.scale();
    let near = 2.0 * scale;
    let (cx, cy) = ((my.x / scale).floor() as i32, (my.y / scale).floor() as i32);
    let (tx, ty) = ((them.x / scale).floor() as i32, (them.y / scale).floor() as i32);

    // --- route: the next cell on the game's own shortest path (diagonals allowed)
    let next = g.round.dist_map(tx, ty).and_then(|dm| {
        let path = crate::maze::shortest_path(&g.round.maze, dm, tx, ty, cx, cy);
        path.get(1).map(|cell| ((cell.x as f64 + 0.5) * scale, (cell.y as f64 + 0.5) * scale))
    });
    let (wx, wy) = next.unwrap_or((them.x, them.y));
    let (wa, wl) = own_frame(my.rotation, wx - my.x, wy - my.y);
    let d = wa.hypot(wl);
    if d > 1e-9 {
        out[0] = (wa / d) as f32;
        out[1] = (wl / d) as f32;
        out[2] = clamp(wl.atan2(wa).abs() / c::DEG / c::TANK_TURN_SPEED / 18.0, 0.0, 1.0) as f32;
    }
    out[3] = exits(g, cx, cy);
    out[4] = exits(g, tx, ty);
    let (ra, rl) = own_frame(my.rotation, them.x - my.x, them.y - my.y);
    let straight = ra.hypot(rl);
    if straight > 1e-9 {
        out[5] = (ra / straight) as f32;
        out[6] = (rl / straight) as f32;
    }

    // --- aim: how near each barrel's shot passes, and mine a few turns either way
    let pass = |shooter: usize, target: usize, rotation: f64| -> (f32, f32) {
        if !g.tanks[shooter].alive || !g.tanks[target].alive {
            return (1.0, 1.0);
        }
        let (hit, frame, closest) = shot_pass(g, shooter, target, rotation);
        let miss = if hit { 0.0 } else { clamp(closest / near, 0.0, 1.0) as f32 };
        (miss, (frame as f64 / SHOT_FRAMES as f64) as f32)
    };
    (out[7], out[8]) = pass(me, other, my.rotation);
    (out[9], out[10]) = pass(other, me, them.rotation);
    let probe = AIM_PROBE_TURNS * c::TANK_TURN_SPEED;
    out[11] = pass(me, other, norm_rot(my.rotation - probe)).0;
    out[12] = pass(me, other, norm_rot(my.rotation + probe)).0;

    // --- threat: every live bullet flown at me, and at them
    let slots = g.settings_max_bullets.max(1) as f64;
    let (mut first, mut on_me, mut closest, mut on_them) = (FORECAST_FRAMES, 0.0, f64::INFINITY, 0.0);
    for b in g.bullets.iter().filter(|b| !b.removed) {
        let (hits, frame, d) = bullet_forecast(g, b, my);
        if hits {
            first = first.min(frame);
            on_me += 1.0;
        } else {
            closest = closest.min(d);
        }
        if them.alive && bullet_forecast(g, b, them).0 {
            on_them += 1.0;
        }
    }
    out[13] = (first as f64 / FORECAST_FRAMES as f64) as f32;
    out[14] = clamp(on_me / slots, 0.0, 1.0) as f32;
    out[15] = clamp(closest / near, 0.0, 1.0) as f32;
    out[16] = clamp(on_them / slots, 0.0, 1.0) as f32;

    // --- rhythm
    out[17] = (ctx.history.held as f64).min(HOLD_CAP) as f32 / HOLD_CAP as f32;
    out[18] = (ctx.history.idle_streak as f64).min(HOLD_CAP) as f32 / HOLD_CAP as f32;
}

/// The polyline of a shot fired now from `shooter`'s barrel (a point at each
/// ricochet) and the tank it would hit first.
pub fn shot_path(g: &Game, shooter: usize) -> (Vec<(f64, f64)>, Option<usize>) {
    let s = &g.tanks[shooter];
    let scale = g.scale();
    let rad = (s.rotation - 90.0) * c::DEG;
    let speed = c::BULLETSPEED / c::BULLETHITCHECKINTERVALS as f64 * (scale / 50.0);
    let mut p = [s.x + rad.cos() * scale * 4.5 / 16.0, s.y + rad.sin() * scale * 4.5 / 16.0, rad.cos() * speed, rad.sin() * speed];
    let mut points = vec![(p[0], p[1])];
    for _ in 0..SHOT_FRAMES {
        for _ in 0..c::BULLETHITCHECKINTERVALS {
            let (vx, vy) = (p[2], p[3]);
            substep(g, &mut p);
            if (p[2], p[3]) != (vx, vy) {
                points.push((p[0], p[1]));
            }
        }
        for (i, t) in g.tanks.iter().enumerate() {
            if t.alive && t.point_in_shape(p[0], p[1]) {
                points.push((p[0], p[1]));
                return (points, Some(i));
            }
        }
    }
    points.push((p[0], p[1]));
    (points, None)
}

/// Ids of live bullets forecast to hit tank `me` if it stays put.
pub fn threat_ids(g: &Game, me: usize) -> Vec<u32> {
    let my = &g.tanks[me];
    g.bullets.iter().filter(|b| !b.removed && bullet_forecast(g, b, my).0).map(|b| b.id).collect()
}

/// What the observation needs that the game itself does not keep.
pub struct Context<'a> {
    pub prev_pose: &'a [[f64; 3]; 2],
    pub history: &'a SeatHistory,
    pub frames: u32,
    /// Frame of each seat's most recent shot this round.
    pub last_fire: [Option<u32>; 2],
}

/// Encode seat `me`'s observation into `out` (OBS_DIM) and its bullet mask.
pub fn encode(g: &Game, me: usize, ctx: &Context, out: &mut [f32], mask: &mut [u8]) {
    out[..OBS_DIM].fill(0.0);
    mask[..BULLET_SLOTS].fill(0);
    let other = 1 - me;
    let (my, them) = (&g.tanks[me], &g.tanks[other]);
    let scale = g.scale();
    let maze = &g.round.maze;
    let (w, h) = (maze.w as i32, maze.h as i32);
    let speed_scale = MAX_SPEED_CELLS * scale;
    let slots = g.settings_max_bullets as f64;

    // --- self
    let prev = ctx.prev_pose[me];
    let (va, vl) = own_frame(my.rotation, my.x - prev[0], my.y - prev[1]);
    let heading = (my.rotation - 90.0) * c::DEG;
    let s = &mut out[SELF_OFFSET..SELF_OFFSET + SELF_DIM];
    s[0] = ((slots - my.bullets_fired as f64).max(0.0) / slots) as f32;
    s[1] = (my.alive && g.weapon_ready(me) && my.trigger_released) as u8 as f32;
    s[2] = heading.cos() as f32;
    s[3] = heading.sin() as f32;
    s[4] = clamp(va / speed_scale, -1.0, 1.0) as f32;
    s[5] = clamp(vl / speed_scale, -1.0, 1.0) as f32;
    s[6] = clamp(norm_rot(my.rotation - prev[2]) / c::TANK_TURN_SPEED, -1.0, 1.0) as f32;
    s[7] = my.hit_something as u8 as f32;

    // --- local walls: the 5x5 cells around mine, maze-aligned; the arena edge is a wall
    let (cx, cy) = ((my.x / scale).floor() as i32, (my.y / scale).floor() as i32);
    let edge_below = |x: i32, y: i32| x >= 0 && x < w && y >= 0 && y < h - 1 && maze.h_open(x, y);
    let edge_left = |x: i32, y: i32| x > 0 && x < w && y >= 0 && y < h && maze.v_open(x, y);
    let mut k = LOCAL_OFFSET;
    for line in 0..=MAP_SPAN {
        for col in 0..MAP_SPAN {
            // the line above row `line`: below cell (x, y - 1)
            let (x, y) = (cx - MAP_RADIUS + col, cy - MAP_RADIUS + line);
            out[k] = if edge_below(x, y - 1) { 0.0 } else { 1.0 };
            k += 1;
        }
    }
    for line in 0..=MAP_SPAN {
        for row in 0..MAP_SPAN {
            let (x, y) = (cx - MAP_RADIUS + line, cy - MAP_RADIUS + row);
            out[k] = if edge_left(x, y) { 0.0 } else { 1.0 };
            k += 1;
        }
    }
    out[k] = clamp(my.x / scale - cx as f64, 0.0, 1.0) as f32;
    out[k + 1] = clamp(my.y / scale - cy as f64, 0.0, 1.0) as f32;

    // --- opponent
    let span = (w + h) as f64 * scale;
    let (ra, rl) = own_frame(my.rotation, them.x - my.x, them.y - my.y);
    let straight = ra.hypot(rl);
    let (tx, ty) = ((them.x / scale).floor() as i32, (them.y / scale).floor() as i32);
    let path = g.round.cell_dist(cx, cy, tx, ty);
    let o = &mut out[OPPONENT_OFFSET..OPPONENT_OFFSET + OPPONENT_DIM];
    o[0] = clamp(ra / span, -1.0, 1.0) as f32;
    o[1] = clamp(rl / span, -1.0, 1.0) as f32;
    o[2] = clamp(straight / span, 0.0, 1.0) as f32;
    o[3] = if path.is_finite() { clamp(path / MAX_PATH_CELLS, 0.0, 1.0) as f32 } else { 1.0 };
    if path.is_finite() {
        // The neighbouring cell that is closest to the opponent, as a direction in my frame.
        let mut best: Option<(f64, usize)> = None;
        for (dx, dy, open) in [(0, -1, edge_below(cx, cy - 1)), (1, 0, edge_left(cx + 1, cy)),
                               (0, 1, edge_below(cx, cy)), (-1, 0, edge_left(cx, cy))] {
            let d = g.round.cell_dist(cx + dx, cy + dy, tx, ty);
            if open && d < path && best.is_none_or(|b| d < b.0) {
                let (a, l) = own_frame(my.rotation, dx as f64, dy as f64);
                let q = if a.abs() >= l.abs() { if a > 0.0 { 0 } else { 2 } } else if l > 0.0 { 1 } else { 3 };
                best = Some((d, q));
            }
        }
        if let Some((_, q)) = best {
            o[4 + q] = 1.0;
        }
    }
    o[8] = (them.alive && line_of_sight(g, (my.x, my.y), (them.x, them.y))) as u8 as f32;
    // Their barrel against the bearing from them to me.
    let bearing = (my.y - them.y).atan2(my.x - them.x);
    let barrel = (them.rotation - 90.0) * c::DEG;
    o[9] = (barrel - bearing).cos() as f32;
    o[10] = (barrel - bearing).sin() as f32;
    let tp = ctx.prev_pose[other];
    let radial = ((them.x - tp[0]) * (my.x - them.x) + (them.y - tp[1]) * (my.y - them.y)) / straight.max(1e-9);
    o[11] = clamp(radial / speed_scale, -1.0, 1.0) as f32;
    o[12] = them.alive as u8 as f32;

    // --- barrels, with the engine's own ballistics
    let b = &mut out[BARREL_OFFSET..BARREL_OFFSET + BARREL_DIM];
    let shot_time = |f: usize| (f as f64 / SHOT_FRAMES as f64) as f32;
    match engine_shot(g, me) {
        Some((v, f)) if v == other => {
            b[0] = 1.0;
            b[2] = shot_time(f);
        }
        Some((_, f)) => {
            b[1] = 1.0;
            b[2] = shot_time(f);
        }
        None => b[2] = 1.0,
    }
    match engine_shot(g, other) {
        Some((v, f)) if v == me => {
            b[3] = 1.0;
            b[5] = shot_time(f);
        }
        Some((_, f)) => {
            b[4] = 1.0;
            b[5] = shot_time(f);
        }
        None => b[5] = 1.0,
    }

    // --- opponent intent
    let about_to_fire = laika_about_to_fire(g, other, me);
    let turns = |t: i32| if t < 0 { 1.0 } else { t as f32 / 19.0 };
    let i = &mut out[INTENT_OFFSET..INTENT_OFFSET + INTENT_DIM];
    i[0] = about_to_fire as u8 as f32;
    i[1] = turns(turns_to_hit(g, me, other));
    i[2] = turns(turns_to_hit(g, other, me));
    // Laika hunts when its shortest path to me is under 7 cells and always
    // shoots within 2 — counted the way it counts them.
    let cells = match g.round.dist_map(tx, ty) {
        Some(dm) if path.is_finite() => crate::maze::shortest_path(maze, dm, tx, ty, cx, cy).len() as f64,
        _ => f64::INFINITY,
    };
    i[3] = (cells < LAIKA_SHOOT_PATH) as u8 as f32;
    i[4] = (cells <= LAIKA_ALWAYS_SHOOT_PATH) as u8 as f32;
    i[5] = match ctx.last_fire[other] {
        Some(f) => clamp((ctx.frames - f) as f64 / SHOT_CLOCK, 0.0, 1.0) as f32,
        None => 1.0,
    };
    i[6] = ((slots - them.bullets_fired as f64).max(0.0) / slots) as f32;

    // --- bullets, the most urgent first
    let range = BULLET_RANGE_CELLS * scale;
    let mut rows: Vec<(f64, [f32; BULLET_DIM])> = g
        .bullets
        .iter()
        .filter(|b| !b.removed)
        .map(|b| {
            let (a, l) = own_frame(my.rotation, b.x - my.x, b.y - my.y);
            let speed = b.x_speed.hypot(b.y_speed).max(1e-12);
            let (vx, vy) = own_frame(my.rotation, b.x_speed / speed, b.y_speed / speed);
            let (hits, frame, closest) = bullet_forecast(g, b, my);
            let urgency = if hits { frame as f64 } else { FORECAST_FRAMES as f64 + closest / scale };
            (urgency, [
                clamp(a / range, -1.0, 1.0) as f32,
                clamp(l / range, -1.0, 1.0) as f32,
                vx as f32,
                vy as f32,
                (b.owner == me) as u8 as f32,
                b.has_bounced as u8 as f32,
                clamp(b.lifetime as f64 / c::BULLETLIFETIME as f64, 0.0, 1.0) as f32,
                hits as u8 as f32,
                (frame as f64 / FORECAST_FRAMES as f64) as f32,
                if hits { 0.0 } else { clamp(closest / (2.0 * scale), 0.0, 1.0) as f32 },
            ])
        })
        .collect();
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (slot, (_, row)) in rows.iter().take(BULLET_SLOTS).enumerate() {
        out[BULLET_OFFSET + slot * BULLET_DIM..BULLET_OFFSET + (slot + 1) * BULLET_DIM].copy_from_slice(row);
        mask[slot] = 1;
    }

    // --- survival, including the shot Laika is about to take
    survival(g, me, about_to_fire, &mut out[DODGE_OFFSET..DODGE_OFFSET + DODGE_DIM]);

    // --- time and what I have been doing
    let t = &mut out[TIME_OFFSET..TIME_OFFSET + TIME_DIM];
    t[0] = clamp(ctx.frames as f64 / DUEL_FRAMES as f64, 0.0, 1.0) as f32;
    t[1] = (g.alive_count <= 1) as u8 as f32;
    if let Some(a) = ctx.history.last {
        let (throttle, turn, fire) = candidate(a);
        t[2] = throttle as f32 / 2.0;
        t[3] = turn as f32 / 2.0;
        t[4] = fire as u8 as f32;
    }
    t[5] = match ctx.last_fire[me] {
        Some(f) => clamp((ctx.frames - f) as f64 / SHOT_CLOCK, 0.0, 1.0) as f32,
        None => 1.0,
    };

    tactics(g, me, ctx, &mut out[TACTIC_OFFSET..OBS_DIM]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::{Episode, OpponentKind, Reward};

    /// Every value is finite and in [-1, 1], and every local wall edge agrees
    /// with the engine's own wall test at the edge's midpoint.
    #[test]
    fn values_are_bounded_and_the_local_map_matches_the_walls() {
        let mut out = vec![0f32; OBS_DIM];
        let mut mask = vec![0u8; BULLET_SLOTS];
        let (mut checked, mut s) = (0, 7u32);
        for seed in [3u64, 970_001, 5_000_123] {
            let mut ep = Episode::new(seed, OpponentKind::Laika, Reward::default());
            while ep.outcome == crate::env::Outcome::Running && ep.frames < 300 {
                for seat in [0, 1] {
                    ep.observe(seat, &mut out, &mut mask);
                    assert!(out.iter().all(|v| v.is_finite() && (-1.0..=1.0).contains(v)), "seed {seed} frame {}", ep.frames);
                }
                ep.observe(0, &mut out, &mut mask);
                let g = &ep.game;
                let scale = g.scale();
                let (cx, cy) = ((g.tanks[0].x / scale).floor() as i32, (g.tanks[0].y / scale).floor() as i32);
                let (w, h) = (g.round.maze.w as i32, g.round.maze.h as i32);
                let mut k = LOCAL_OFFSET;
                for line in 0..=MAP_SPAN {
                    for col in 0..MAP_SPAN {
                        let (x, y) = (cx - MAP_RADIUS + col, cy - MAP_RADIUS + line);
                        if x >= 0 && x < w && y > 0 && y < h {
                            let mid = ((x as f64 + 0.5) * scale, y as f64 * scale);
                            assert_eq!(out[k] == 1.0, g.wall_hit(mid.0, mid.1), "horizontal edge ({x},{y}) seed {seed}");
                            checked += 1;
                        }
                        k += 1;
                    }
                }
                for line in 0..=MAP_SPAN {
                    for row in 0..MAP_SPAN {
                        let (x, y) = (cx - MAP_RADIUS + line, cy - MAP_RADIUS + row);
                        if x > 0 && x < w && y >= 0 && y < h {
                            let mid = (x as f64 * scale, (y as f64 + 0.5) * scale);
                            assert_eq!(out[k] == 1.0, g.wall_hit(mid.0, mid.1), "vertical edge ({x},{y}) seed {seed}");
                            checked += 1;
                        }
                        k += 1;
                    }
                }
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                ep.step((s % 18) as usize, None);
            }
        }
        assert!(checked > 1000, "only {checked} edges checked");
    }
}
