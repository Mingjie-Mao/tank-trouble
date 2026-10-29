//! The simulation, from `tank_trouble_original/game.py`.
//!
//! Fixed 25 FPS; one `step()` is one AVM1 frame. AVM1 runs the most recently
//! created clip first: bullets (newest first), then tank 1 (Laika decides
//! inside its own update), then tank 0, and last the root timeline's round
//! logic. Bullets therefore move and hit-test before either tank moves, and a
//! bullet fired this frame first moves on the next.

use std::fmt;
use std::sync::Arc;

use crate::constants as c;
use crate::laika::Laika;
use crate::maze::{calc_distances, calc_reachable, find_dead_ends, wall_segments, Cell, Grid, Maze, WallGrid};
use crate::rng::Rng;

/// Flash `_rotation` semantics: normalise to (-180, 180] on every write.
pub fn norm_rot(deg: f64) -> f64 {
    let d = deg % 360.0;
    if d > 180.0 {
        d - 360.0
    } else if d <= -180.0 {
        d + 360.0
    } else {
        d
    }
}

const BW: f64 = c::TANK_BASE_WIDTH;
const BH: f64 = c::TANK_BASE_HEIGHT;
const TW: f64 = c::TANK_TURRET_WIDTH;
const TH: f64 = c::TANK_TURRET_HEIGHT;
/// Wall probes in local sprite units (tank sprite :58-82). The front row has
/// no centre point and the rear row does; that asymmetry is original.
/// [x0, y0, x1, y1] around every side's probes.
fn probe_bounds() -> [f64; 4] {
    static BOUNDS: std::sync::OnceLock<[f64; 4]> = std::sync::OnceLock::new();
    *BOUNDS.get_or_init(|| {
        let all = PROBES_FRONT.iter().chain(&PROBES_REAR).chain(&PROBES_LEFT).chain(&PROBES_RIGHT);
        all.fold([f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY], |b, p| {
            [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])]
        })
    })
}

pub const PROBES_FRONT: [[f64; 2]; 6] = [
    [-BW / 2.0, -BH / 2.0], [-BW / 4.0, -BH / 2.0], [BW / 4.0, -BH / 2.0], [BW / 2.0, -BH / 2.0],
    [-TW / 6.0, (-TH / 16.0) * 11.0], [TW / 6.0, (-TH / 16.0) * 11.0],
];
pub const PROBES_REAR: [[f64; 2]; 5] = [
    [-BW / 2.0, BH / 2.0], [-BW / 4.0, BH / 2.0], [0.0, BH / 2.0], [BW / 4.0, BH / 2.0], [BW / 2.0, BH / 2.0],
];
pub const PROBES_RIGHT: [[f64; 2]; 5] = [
    [BW / 2.0, (-BH / 6.0) * 2.0], [BW / 2.0, -BH / 6.0], [BW / 2.0, 0.0], [BW / 2.0, BH / 6.0], [BW / 2.0, (BH / 6.0) * 2.0],
];
pub const PROBES_LEFT: [[f64; 2]; 5] = [
    [-BW / 2.0, (-BH / 6.0) * 2.0], [-BW / 2.0, -BH / 6.0], [-BW / 2.0, 0.0], [-BW / 2.0, BH / 6.0], [-BW / 2.0, (BH / 6.0) * 2.0],
];

/// Everything fixed for one round: the maze and what is derived from it.
#[derive(Debug)]
pub struct Round {
    pub maze: Maze,
    pub scale: f64,
    pub walls: Vec<[f64; 4]>,
    pub wall_half_t: f64,
    pub grid: WallGrid,
    pub reachable: Vec<Cell>,
    /// Distance map from each reachable cell, index x * h + y.
    pub distances: Vec<Option<Grid>>,
    pub dead_ends: Grid,
}

impl Round {
    /// distancesForMaze[fx][fy]; None for an unreachable or out-of-range cell.
    pub fn dist_map(&self, fx: i32, fy: i32) -> Option<&Grid> {
        if fx >= 0 && (fx as usize) < self.maze.w && fy >= 0 && (fy as usize) < self.maze.h {
            self.distances[fx as usize * self.maze.h + fy as usize].as_ref()
        } else {
            None
        }
    }

    /// Maze distance between two cells; NaN when unknown.
    pub fn cell_dist(&self, fx: i32, fy: i32, cx: i32, cy: i32) -> f64 {
        self.dist_map(fx, fy).map_or(f64::NAN, |d| d.at(cx, cy))
    }
}

#[derive(Clone, Debug)]
pub struct Tank {
    pub number: usize,
    pub x: f64,
    pub y: f64,
    pub rotation: f64,
    pub forward_speed: f64,
    pub backup_speed: f64,
    pub turn_speed: f64,
    pub display_scale: f64,
    pub trigger_released: bool,
    pub bullets_fired: i32,
    pub alive: bool,
    pub hit_something: bool,
    pub forward: bool,
    pub backup: bool,
    pub turn_left: bool,
    pub turn_right: bool,
    pub fire: bool,
}

impl Tank {
    fn deploy(number: usize, cell: Cell, scale: f64, rng: &mut Rng) -> Tank {
        Tank {
            number,
            x: (cell.x as f64 + 0.5) * scale,
            y: (cell.y as f64 + 0.5) * scale,
            rotation: norm_rot((rng.random() * 32.0).floor() * 11.25),
            forward_speed: c::TANK_FORWARD_SPEED_BASE * (scale / 50.0),
            backup_speed: c::TANK_BACKUP_SPEED_BASE * (scale / 50.0),
            turn_speed: c::TANK_TURN_SPEED,
            display_scale: c::TANK_DISPLAY_SCALE_FACTOR * scale,
            trigger_released: true,
            bullets_fired: 0,
            alive: true,
            hit_something: false,
            forward: false,
            backup: false,
            turn_left: false,
            turn_right: false,
            fire: false,
        }
    }

    pub fn set_buttons(&mut self, forward: bool, backup: bool, left: bool, right: bool, fire: bool) {
        self.forward = forward;
        self.backup = backup;
        self.turn_left = left;
        self.turn_right = right;
        self.fire = fire;
    }

    /// tank:12-41 — does any probe point, pushed out by `factor`, touch a wall?
    pub fn hit_check(&self, grid: &WallGrid, points: &[[f64; 2]], factor: f64) -> bool {
        self.probes_hit(grid, points, factor, (self.rotation * c::DEG).sin_cos())
    }

    fn probes_hit(&self, grid: &WallGrid, points: &[[f64; 2]], factor: f64, (sn, cs): (f64, f64)) -> bool {
        let s = self.display_scale;
        points.iter().any(|p| {
            let (lx, ly) = (p[0] * factor, p[1] * factor);
            grid.hit(self.x + s * (lx * cs - ly * sn), self.y + s * (lx * sn + ly * cs))
        })
    }

    pub fn any_side_hit(&self, grid: &WallGrid) -> bool {
        let (sn, cs) = (self.rotation * c::DEG).sin_cos();
        // Every probe lies inside the turned box of the probes' bounds: when
        // no wall meets that box, none can hit.
        let s = self.display_scale;
        let [x0, y0, x1, y1] = probe_bounds();
        let (mut lo_x, mut hi_x, mut lo_y, mut hi_y) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
        for (lx, ly) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
            let gx = self.x + s * (lx * cs - ly * sn);
            let gy = self.y + s * (lx * sn + ly * cs);
            lo_x = lo_x.min(gx);
            hi_x = hi_x.max(gx);
            lo_y = lo_y.min(gy);
            hi_y = hi_y.max(gy);
        }
        const MARGIN: f64 = 1e-6;
        if !grid.box_touches(lo_x - MARGIN, lo_y - MARGIN, hi_x + MARGIN, hi_y + MARGIN) {
            return false;
        }
        [&PROBES_FRONT[..], &PROBES_REAR, &PROBES_LEFT, &PROBES_RIGHT].iter().any(|p| self.probes_hit(grid, p, 1.0, (sn, cs)))
    }

    /// hitTest(x, y, true): hull rectangle union barrel rectangle.
    pub fn point_in_shape(&self, px: f64, py: f64) -> bool {
        let s = self.display_scale;
        let th = self.rotation * c::DEG;
        let (sn, cs) = th.sin_cos();
        let (dx, dy) = (px - self.x, py - self.y);
        let lx = (dx * cs + dy * sn) / s;
        let ly = (-dx * sn + dy * cs) / s;
        if -BW / 2.0 <= lx && lx <= BW / 2.0 && -BH / 2.0 <= ly && ly <= BH / 2.0 {
            return true;
        }
        lx.abs() <= c::TANK_SHAPE_BARREL_HALF_WIDTH && c::TANK_SHAPE_BARREL_TIP_Y <= ly && ly <= 0.0
    }

    /// hitTest(x, y, false): the axis-aligned box around the rotated bounds.
    pub fn point_in_bbox(&self, px: f64, py: f64) -> bool {
        let s = self.display_scale;
        let th = self.rotation * c::DEG;
        let (sn, cs) = th.sin_cos();
        let [xmin, ymin, xmax, ymax] = c::TANK_BOUNDS_LOCAL;
        let (mut lo_x, mut hi_x, mut lo_y, mut hi_y) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
        for (lx, ly) in [(xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax)] {
            let gx = self.x + s * (lx * cs - ly * sn);
            let gy = self.y + s * (lx * sn + ly * cs);
            lo_x = lo_x.min(gx);
            hi_x = hi_x.max(gx);
            lo_y = lo_y.min(gy);
            hi_y = hi_y.max(gy);
        }
        lo_x <= px && px <= hi_x && lo_y <= py && py <= hi_y
    }

    /// The movement half of tank onEnterFrame (:322-410): five substeps, a
    /// careful redo on wall contact, then the heading snap.
    pub fn drive(&mut self, grid: &WallGrid) {
        let (old_x, old_y, old_rot) = (self.x, self.y, self.rotation);
        let steps = c::TANK_MOVE_STEPS;
        let mut move_size = 0.0;
        let mut turn_size = 0.0;
        if self.forward {
            move_size = self.forward_speed / steps as f64;
        }
        if self.backup {
            move_size -= self.backup_speed / steps as f64;
        }
        if self.turn_left {
            turn_size = -self.turn_speed / steps as f64;
        }
        if self.turn_right {
            turn_size += self.turn_speed / steps as f64;
        }

        self.hit_something = false;
        for _ in 0..steps {
            self.rotation = norm_rot(self.rotation + turn_size);
            let rad = (self.rotation - 90.0) * c::DEG;
            self.x += rad.cos() * move_size;
            self.y += rad.sin() * move_size;
        }

        if self.any_side_hit(grid) {
            self.x = old_x;
            self.y = old_y;
            self.rotation = old_rot;
            for _ in 0..steps {
                let step_old_rot = self.rotation;
                self.rotation = norm_rot(self.rotation + turn_size);
                if self.any_side_hit(grid) {
                    self.rotation = step_old_rot;
                    self.hit_something = true;
                }
                let (sx, sy) = (self.x, self.y);
                let rad = (self.rotation - 90.0) * c::DEG;
                self.x += rad.cos() * move_size;
                self.y += rad.sin() * move_size;
                if (move_size > 0.0 && self.hit_check(grid, &PROBES_FRONT, 1.0))
                    || (move_size < 0.0 && self.hit_check(grid, &PROBES_REAR, 1.0))
                {
                    self.x = sx;
                    self.y = sy;
                    self.hit_something = true;
                }
            }
        }

        let offset = (360.0 + self.rotation) % self.turn_speed;
        if !self.hit_something && turn_size != 0.0 && offset != 0.0 {
            if offset < self.turn_speed / 2.0 {
                self.rotation = norm_rot(self.rotation - offset);
            } else {
                self.rotation = norm_rot(self.rotation + (self.turn_speed - offset));
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Bullet {
    /// "bullet<n>", n counting from 1 within the round.
    pub id: u32,
    pub owner: usize,
    pub x: f64,
    pub y: f64,
    pub x_speed: f64,
    pub y_speed: f64,
    pub lifetime: i32,
    pub deadly: i32,
    pub removed: bool,
    /// A fact for observers, not a rule: there is no shooter exemption.
    pub has_bounced: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    NewRound(u32),
    Fire(usize),
    Bounce(u32),
    Hit { owner: usize, victim: usize },
    Destroy(usize),
    RoundEnd(Option<usize>),
}

impl fmt::Display for Event {
    /// The Python port's event tuples, comma-joined.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Event::NewRound(n) => write!(f, "new_round,{n}"),
            Event::Fire(t) => write!(f, "fire,{t}"),
            Event::Bounce(id) => write!(f, "bounce,bullet{id}"),
            Event::Hit { owner, victim } => write!(f, "hit,{owner},{victim}"),
            Event::Destroy(t) => write!(f, "destroy,{t}"),
            Event::RoundEnd(None) => write!(f, "round_end,null"),
            Event::RoundEnd(Some(w)) => write!(f, "round_end,{w}"),
        }
    }
}

#[derive(Clone)]
pub struct Game {
    pub rng: Rng,
    pub seed: u64,
    pub tanks_count: usize,
    /// Native Laika on tank 1, rebuilt every round.
    pub laika_on_tank1: bool,
    /// Native Laika on tank 0 as well (the web demo's Laika-vs-Laika).
    pub laika_on_tank0: bool,
    pub settings_max_bullets: i32,
    pub alive_count: i32,
    pub end_count: i32,
    pub reset_count: i32,
    pub frozen: bool,
    pub shake: f64,
    pub crate_timer: f64,
    pub scores: Vec<i32>,
    pub round_number: u32,
    pub frame: u64,
    pub events: Vec<Event>,
    pub round: Arc<Round>,
    pub tank_fields: Vec<Cell>,
    pub tanks: Vec<Tank>,
    pub ai: Vec<Option<Box<Laika>>>,
    pub bullets: Vec<Bullet>,
    pub bullet_depth: u32,
    /// Where each bullet updated this frame took its hit test, removed ones
    /// included. Lookaheads replay these against hypothetical hulls.
    pub hit_tests: Vec<(f64, f64)>,
}

impl Game {
    pub fn new(seed: u64, laika_on_tank1: bool) -> Game {
        Game::with_laika(seed, [false, laika_on_tank1])
    }

    /// A game with native Laika on the tanks marked true.
    pub fn with_laika(seed: u64, laika: [bool; 2]) -> Game {
        let mut rng = Rng::new(seed);
        // Crates never spawn against Laika, but the timer runs and draws from the RNG.
        let crate_timer = (c::CRATESPAWNTIMEBASE + rng.randrange(c::CRATESPAWNTIMERANDOM) as f64)
            * c::SETTINGS_CRATE_SPAWN_MODIFIER;
        let tanks_count = 2;
        let mut game = Game {
            rng,
            seed,
            tanks_count,
            laika_on_tank1: laika[1],
            laika_on_tank0: laika[0],
            settings_max_bullets: c::SETTINGS_MAX_BULLETS,
            alive_count: 0,
            end_count: -1,
            reset_count: -1,
            frozen: false,
            shake: 0.0,
            crate_timer,
            scores: vec![0; tanks_count],
            round_number: 0,
            frame: 0,
            events: Vec::new(),
            round: Arc::new(Round::placeholder()),
            tank_fields: Vec::new(),
            tanks: Vec::new(),
            ai: Vec::new(),
            bullets: Vec::new(),
            bullet_depth: 0,
            hit_tests: Vec::new(),
        };
        game.setup_battle();
        game
    }

    pub fn scale(&self) -> f64 {
        self.round.scale
    }

    #[inline]
    pub fn wall_hit(&self, px: f64, py: f64) -> bool {
        self.round.grid.hit(px, py)
    }

    /// frame_53 setupBattle + setupStandardMaze + deployTank.
    fn setup_battle(&mut self) {
        self.round_number += 1;
        let n = self.tanks_count;
        let rng = &mut self.rng;
        let mut spawn = vec![Cell { x: 0, y: 0 }; n];
        let (mut maze, mut scale, mut reachable);
        loop {
            let width = rng.randrange(9) as usize + 4; // 4..12
            let height = rng.randrange(7) as usize + 4; // 4..10
            scale = ((c::MOVIEHEIGHT - 80.0) / (height as f64 + 0.125)).min(c::MOVIEWIDTH / (width as f64 + 0.125));
            maze = Maze::generate(width, height, rng);
            spawn[0] = Cell { x: (rng.random() * width as f64).floor() as i32, y: (rng.random() * height as f64).floor() as i32 };
            reachable = calc_reachable(&maze, spawn[0].x, spawn[0].y);
            if reachable.len() >= 2 * n {
                break;
            }
        }
        let mut used = vec![false; reachable.len()];
        used[0] = true;
        let mut i = 1;
        while i < n {
            let k = (rng.random() * reachable.len() as f64).floor() as usize;
            if !used[k] {
                spawn[i] = reachable[k];
                used[k] = true;
                i += 1;
            }
        }

        let walls = wall_segments(&maze, scale);
        let wall_half_t = (scale / 16.0).floor();
        let grid = WallGrid::new(&walls, wall_half_t, scale);

        self.tanks = (0..n).map(|k| Tank::deploy(k, spawn[k], scale, &mut self.rng)).collect();
        self.bullets.clear();
        self.bullet_depth = 0;
        self.alive_count = n as i32;

        let mut distances = vec![None; maze.w * maze.h];
        for cell in &reachable {
            distances[cell.x as usize * maze.h + cell.y as usize] = Some(calc_distances(&maze, cell.x, cell.y));
        }
        let dead_ends = find_dead_ends(&maze, &reachable, c::MAXDEADENDPENALTY);
        self.round = Arc::new(Round { maze, scale, walls, wall_half_t, grid, reachable, distances, dead_ends });
        self.tank_fields = spawn;
        let native = |k: usize| (k == 1 && self.laika_on_tank1) || (k == 0 && self.laika_on_tank0);
        self.ai = (0..n).map(|k| if native(k) { Some(Box::new(Laika::new(self.round.scale))) } else { None }).collect();
        self.events.push(Event::NewRound(self.round_number));
    }

    pub fn weapon_ready(&self, tank: usize) -> bool {
        self.tanks[tank].bullets_fired < self.settings_max_bullets
    }

    /// frame_53:1227-1282
    fn fire_weapon(&mut self, t: usize) {
        self.bullet_depth += 1;
        let scale = self.round.scale;
        let owner = &self.tanks[t];
        let rad = (owner.rotation - 90.0) * c::DEG;
        let speed = c::BULLETSPEED / c::BULLETHITCHECKINTERVALS as f64 * (scale / 50.0);
        let b = Bullet {
            id: self.bullet_depth,
            owner: t,
            x: owner.x + rad.cos() * scale * 4.5 / 16.0,
            y: owner.y + rad.sin() * scale * 4.5 / 16.0,
            x_speed: rad.cos() * speed,
            y_speed: rad.sin() * speed,
            lifetime: c::BULLETLIFETIME,
            deadly: c::BULLETDEADLY,
            removed: false,
            has_bounced: false,
        };
        self.bullets.push(b);
        self.tanks[t].bullets_fired += 1;
        self.events.push(Event::Fire(t));
    }

    /// frame_53:900-911. A second death re-arms the window: that is a double KO.
    fn destroy_tank(&mut self, t: usize) {
        self.tanks[t].alive = false;
        self.alive_count -= 1;
        self.end_count = c::NUMBEROFFRAMESBEFOREEND;
        self.shake = c::MAXSHAKE.max(self.shake + 7.0);
        self.events.push(Event::Destroy(t));
    }

    /// bullet onEnterFrame.
    fn update_bullet(&mut self, i: usize) {
        let round = Arc::clone(&self.round);
        let mut b = self.bullets[i].clone();
        for _ in 0..c::BULLETHITCHECKINTERVALS {
            let (px, py) = (b.x, b.y);
            b.x += b.x_speed;
            b.y += b.y_speed;
            if round.grid.hit(b.x, b.y) {
                self.events.push(Event::Bounce(b.id));
                b.has_bounced = true;
                // The two inversion probes (bullet :31-58) look asymmetric
                // because they are; that is what fixes every ricochet angle.
                let x_inv = round.grid.hit(px - b.x_speed, py + b.y_speed);
                let y_inv = round.grid.hit(px + b.x_speed, py - b.y_speed);
                if x_inv && !y_inv {
                    b.y_speed = -b.y_speed;
                } else if y_inv && !x_inv {
                    b.x_speed = -b.x_speed;
                } else {
                    b.x_speed = -b.x_speed;
                    b.y_speed = -b.y_speed;
                }
                b.x = px + b.x_speed;
                b.y = py + b.y_speed;
            }
        }
        // One hit test per frame, after the substeps; no break: one bullet can
        // hit two tanks on the same frame.
        self.hit_tests.push((b.x, b.y));
        if b.deadly == 0 {
            for t in 0..self.tanks_count {
                if self.tanks[t].alive && self.tanks[t].point_in_shape(b.x, b.y) {
                    self.events.push(Event::Hit { owner: b.owner, victim: t });
                    self.tanks[b.owner].bullets_fired -= 1;
                    self.destroy_tank(t);
                    b.removed = true;
                }
            }
        }
        if b.deadly > 0 {
            b.deadly -= 1;
        }
        b.lifetime -= 1;
        if b.lifetime <= 0 && !b.removed {
            self.tanks[b.owner].bullets_fired -= 1;
            b.removed = true;
        }
        self.bullets[i] = b;
    }

    /// tank onEnterFrame :138-433.
    fn update_tank(&mut self, t: usize) {
        if self.frozen || !self.tanks[t].alive {
            return;
        }
        // The AI writes this tank's input before anything moves (tank :314-321).
        if let Some(mut ai) = self.ai[t].take() {
            if ai.make_decisions_and_update_goal(self, t) {
                ai.decide_actions_to_achieve_goal(self, t);
            }
            ai.set_input_to_do_actions(self, t);
            self.ai[t] = Some(ai);
        }
        let round = Arc::clone(&self.round);
        self.tanks[t].drive(&round.grid);
        // Edge-triggered fire (tank :411-419).
        if self.tanks[t].fire && self.tanks[t].trigger_released && self.weapon_ready(t) {
            self.tanks[t].trigger_released = false;
            self.fire_weapon(t);
        } else if !self.tanks[t].fire {
            self.tanks[t].trigger_released = true;
        }
    }

    /// frame_53:2015-2050, local branch: the survivor scores.
    fn assign_points(&mut self) {
        let mut winner = None;
        for t in 0..self.tanks_count {
            if self.tanks[t].alive {
                self.scores[t] += 1;
                winner = Some(t);
            }
        }
        self.events.push(Event::RoundEnd(winner));
    }

    /// Advance one frame (1/25 s) in AVM1 execution order; returns its events.
    pub fn step(&mut self) -> &[Event] {
        self.frame += 1;
        self.events.clear();
        self.hit_tests.clear();

        // Bullets, newest first; ones fired this frame first move next frame.
        if !self.frozen {
            for i in (0..self.bullets.len()).rev() {
                if !self.bullets[i].removed {
                    self.update_bullet(i);
                }
            }
        }
        self.bullets.retain(|b| !b.removed);

        // Tanks, most recently created first: tank 1, then tank 0.
        for t in (0..self.tanks.len()).rev() {
            self.update_tank(t);
        }

        // The root timeline's onEnterFrame (frame_53:2366-2536) runs last.
        let scale = self.round.scale;
        for t in 0..self.tanks_count {
            self.tank_fields[t] = Cell {
                x: (self.tanks[t].x / scale).floor() as i32,
                y: (self.tanks[t].y / scale).floor() as i32,
            };
        }
        if !self.frozen {
            self.crate_timer -= 1.0;
        }
        if !self.frozen && self.crate_timer <= 0.0 {
            self.crate_timer = (c::CRATESPAWNTIMEBASE
                + self.rng.randrange(c::CRATESPAWNTIMERANDOM) as f64
                + c::CRATESPAWNMAZESIZESCALE / self.round.reachable.len() as f64)
                * c::SETTINGS_CRATE_SPAWN_MODIFIER;
        }
        if self.shake >= 0.0 {
            self.shake -= 0.5;
        }
        if self.alive_count <= 1 {
            if self.end_count >= 0 {
                self.end_count -= 1;
            }
            if self.end_count == c::NUMBEROFFRAMESFROZEN {
                self.frozen = true;
                self.assign_points();
            }
            if self.end_count == 0 {
                self.bullets.clear();
                self.reset_count = c::NUMBEROFFRAMESBEFORERESET;
            }
        }
        if self.reset_count >= 0 {
            self.reset_count -= 1;
        }
        if self.reset_count == 0 {
            self.end_count = c::NUMBEROFFRAMESBEFOREEND + c::NUMBEROFFRAMESFROZEN;
            self.frozen = false;
            // New tanks join behind this frame's pass: they first run next frame.
            self.setup_battle();
        }
        &self.events
    }
}

impl Round {
    fn placeholder() -> Round {
        let maze = Maze::generate(1, 1, &mut Rng::new(0));
        Round {
            grid: WallGrid::new(&[[0.0; 4]], 0.0, 1.0),
            maze,
            scale: 50.0,
            walls: Vec::new(),
            wall_half_t: 3.0,
            reachable: Vec::new(),
            distances: Vec::new(),
            dead_ends: Grid::new(0, 0, f64::NAN),
        }
    }
}
