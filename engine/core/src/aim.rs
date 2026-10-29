//! Laika's own aiming model (checkBulletPath's flight), shared by the
//! observation (is a barrel holding a hit, how many frames of turning until it
//! does) and by the search's notion of danger.

use crate::constants as c;
use crate::game::{Game, Tank};

/// Turning in place further than this many frames is not counted.
const MAX_TURNS: i32 = 18;

/// A still tank's hit shape with its rotation's sin/cos taken once.
#[derive(Clone, Copy)]
struct Shape {
    x: f64,
    y: f64,
    c: f64,
    s: f64,
    k: f64,
    r2: f64,
}

fn shape_of(t: &Tank, rotation: f64) -> Shape {
    let th = rotation * c::DEG;
    let reach = t.display_scale * (c::TANK_BASE_WIDTH / 2.0).hypot(-c::TANK_SHAPE_BARREL_TIP_Y);
    Shape { x: t.x, y: t.y, c: th.cos(), s: th.sin(), k: 1.0 / t.display_scale, r2: reach * reach }
}

fn in_shape(f: &Shape, px: f64, py: f64) -> bool {
    let (dx, dy) = (px - f.x, py - f.y);
    if dx * dx + dy * dy > f.r2 {
        return false;
    }
    let lx = (dx * f.c + dy * f.s) * f.k;
    let ly = (-dx * f.s + dy * f.c) * f.k;
    if lx.abs() <= c::TANK_BASE_WIDTH / 2.0 && ly.abs() <= c::TANK_BASE_HEIGHT / 2.0 {
        return true;
    }
    lx.abs() <= c::TANK_SHAPE_BARREL_HALF_WIDTH && c::TANK_SHAPE_BARREL_TIP_Y <= ly && ly <= 0.0
}

/// Does a shot fired at `angle` hit `target` before `shooter` (Laika's flight)?
fn shot_hits(g: &Game, shooter: &Tank, from: &Shape, target: &Shape, angle: f64) -> bool {
    let scale = g.scale();
    let rad = ((angle - 90.0) * std::f64::consts::PI) / 180.0;
    let mut x = shooter.x + rad.cos() * scale * 4.5 / 16.0;
    let mut y = shooter.y + rad.sin() * scale * 4.5 / 16.0;
    let mut xs = rad.cos() * c::BULLETSPEED * (scale / 50.0);
    let mut ys = rad.sin() * c::BULLETSPEED * (scale / 50.0);
    let mut life = c::BULLETLIFETIME as f64 / 3.0;
    while life > 0.0 {
        let (px, py) = (x, y);
        x += xs;
        y += ys;
        if g.wall_hit(x, y) {
            let x_inv = g.wall_hit(px - xs, py + ys);
            let y_inv = g.wall_hit(px + xs, py - ys);
            if x_inv && !y_inv {
                ys = -ys;
            } else if y_inv && !x_inv {
                xs = -xs;
            } else {
                xs = -xs;
                ys = -ys;
            }
            x = px + xs;
            y = py + ys;
        }
        if in_shape(target, x, y) {
            return true;
        }
        if in_shape(from, x, y) {
            return false;
        }
        life -= 1.0;
    }
    false
}

/// Laika's own aiming model (checkBulletPath's flight) for a barrel at
/// `angle`: does the shot reach `target` before its shooter?
pub fn laika_aim_hits(g: &Game, shooter: usize, target: usize, angle: f64) -> bool {
    let (s, t) = (&g.tanks[shooter], &g.tanks[target]);
    s.alive && t.alive && shot_hits(g, s, &shape_of(s, angle), &shape_of(t, t.rotation), angle)
}

/// Fewest frames `shooter` must turn in place before its barrel holds a hit
/// on `target`, or -1 if none within half a turn.
pub fn turns_to_hit(g: &Game, shooter: usize, target: usize) -> i32 {
    let (s, t) = (&g.tanks[shooter], &g.tanks[target]);
    if !s.alive || !t.alive {
        return -1;
    }
    let goal = shape_of(t, t.rotation);
    let at = |rotation: f64| shot_hits(g, s, &shape_of(s, rotation), &goal, rotation);
    let r = s.rotation;
    if at(r) {
        return 0;
    }
    for k in 1..=MAX_TURNS {
        let turn = k as f64 * c::TANK_TURN_SPEED;
        if at(r - turn) {
            return k;
        }
        if k < MAX_TURNS && at(r + turn) {
            return k;
        }
    }
    -1
}
