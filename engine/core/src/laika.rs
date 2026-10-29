//! Laika, the original game's scripted AI (DefineSprite_186_tankTroubleAI),
//! from `tank_trouble_original/laika.py`.
//!
//! Three hooks run every frame inside the tank's own update: re-score goals
//! when the current one's period runs out, turn a goal into a LIFO action
//! stack, and turn the top action into buttons. Laika is rebuilt every round.
//! Numeric literals, float artefacts included, are those of the decompiled source.

use crate::constants as c;
use crate::game::{Game, PROBES_FRONT, PROBES_LEFT, PROBES_REAR};
use crate::maze::{escape_path, shortest_path, Cell, Grid};

const PI: f64 = 3.141592653589793;

/// Python's round(): nearest integer, ties to even.
pub fn py_round(x: f64) -> f64 {
    let f = x.floor();
    let diff = x - f;
    if diff > 0.5 {
        f + 1.0
    } else if diff < 0.5 {
        f
    } else if f % 2.0 == 0.0 {
        f
    } else {
        f + 1.0
    }
}

/// Heading, in the game's convention, of the vector (dx, dy).
fn heading_of(dx: f64, dy: f64, fallback: f64) -> f64 {
    if dx != 0.0 {
        if dx > 0.0 {
            90.0 + (dy / dx).atan() * 180.0 / PI
        } else {
            -90.0 + (dy / dx).atan() * 180.0 / PI
        }
    } else if dy > 0.0 {
        180.0
    } else if dy < 0.0 {
        0.0
    } else {
        fallback
    }
}

#[derive(Clone, Debug)]
enum GoalKind {
    Idle,
    DodgeBullet { x: f64, y: f64, closest: (f64, f64), dist: f64, t: f64, dir: (f64, f64), max_time: f64, max_dist: f64 },
    ShootAfter { target: usize },
    RunAway { dist: Grid },
    BackAway,
    DriveTo { x: i32, y: i32 },
}

#[derive(Clone, Debug)]
struct Goal {
    kind: GoalKind,
    priority: f64,
    period: i32,
    id: i64,
    update_continuously: bool,
}

#[derive(Clone, Debug)]
enum Action {
    DriveToField { x: i32, y: i32 },
    TurnTo { angle: f64 },
    FireWeapon { delay: i32 },
    DriveToPos { x: f64, y: f64, can_reverse: bool },
    Forward { dist: i32 },
    Backup { dist: i32 },
    BackupAndTurn { dist: i32, left: bool },
    Idle,
}

struct Collision {
    x: f64,
    y: f64,
    x_speed: f64,
    y_speed: f64,
    t: f64,
}

enum Shot {
    Hit { time: f64 },
    Suicide,
    Nothing { closest: f64 },
}

#[derive(Clone, Debug)]
pub struct Laika {
    aggresiveness: f64,
    cowardness: f64,
    longest_path_to_shoot: f64,
    longest_path_to_not_hesitate: f64,
    longest_path_to_run: f64,
    max_stuck_time: f64,
    stuck_time: f64,
    current_aggresiveness: f64,
    idle_drive_toward_enemy_priority: f64,
    max_closest_cell_distance: f64,
    max_closest_distance: f64,
    max_time_to_dodge: f64,
    max_dist_to_dodge: f64,
    max_cell_dist_to_dodge: f64,
    goal_id: i64,
    goal: Goal,
    actions: Vec<Action>,
}

impl Laika {
    /// AI :1355-1385; several constants depend on this round's cell size.
    pub fn new(scale: f64) -> Laika {
        let aggresiveness = 0.5;
        Laika {
            aggresiveness,
            cowardness: 0.7000000000000001,
            longest_path_to_shoot: 7.0,
            longest_path_to_not_hesitate: 2.0,
            longest_path_to_run: 10.0,
            max_stuck_time: 1.0,
            stuck_time: 0.0,
            current_aggresiveness: aggresiveness,
            idle_drive_toward_enemy_priority: 0.1,
            max_closest_cell_distance: 2.0,
            max_closest_distance: scale * 2.0,
            max_time_to_dodge: 75.0,
            max_dist_to_dodge: 4.0 * scale,
            max_cell_dist_to_dodge: (75.0 * c::BULLETSPEED) / 50.0,
            goal_id: 1,
            goal: Goal { kind: GoalKind::Idle, priority: 0.0, period: 15, id: 0, update_continuously: true },
            actions: Vec::new(),
        }
    }

    /// This Laika with the two tank indices swapped, for a mirrored game.
    pub fn mirrored(&self) -> Laika {
        let mut l = self.clone();
        if let GoalKind::ShootAfter { target } = &mut l.goal.kind {
            *target = 1 - *target;
        }
        l
    }

    fn update_goal(&mut self, temp: Goal) {
        if self.goal.priority < temp.priority {
            self.goal = temp;
        }
    }

    fn next_id(&mut self) -> i64 {
        let id = self.goal_id;
        self.goal_id += 1;
        id
    }

    /// AI :165-229. Walk a straight line; on the first wall, bounce and report.
    #[allow(clippy::too_many_arguments)]
    fn check_path_for_collision(g: &Game, mut x: f64, mut y: f64, mut xs: f64, mut ys: f64,
                                interval: i32, maxtime: f64, lifetime: f64) -> Option<Collision> {
        let mut lifetime = maxtime.min(lifetime);
        let mut t = 0.0;
        while lifetime > 0.0 {
            for _ in 0..interval {
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
                    return Some(Collision { x: px + xs, y: py + ys, x_speed: xs, y_speed: ys, t });
                }
            }
            lifetime -= 1.0;
            t += 1.0;
        }
        None
    }

    /// Straight unit-step wall check from (x, y) along (dx, dy) for its length.
    fn blocked(g: &Game, x: f64, y: f64, dx: f64, dy: f64) -> Option<bool> {
        let d = (dx * dx + dy * dy).sqrt();
        if d > 0.0 {
            Some(Self::check_path_for_collision(g, x, y, dx / d, dy / d, 1, d.ceil(), d.ceil()).is_some())
        } else {
            None
        }
    }

    /// AI :230-336. Fly one bullet from the muzzle at `angle` with Laika's
    /// deliberately coarse ballistics: one substep per frame, a third of a life.
    fn check_bullet_path(&self, g: &Game, me: usize, angle: f64) -> Shot {
        let scale = g.scale();
        let my = &g.tanks[me];
        let rad = ((angle - 90.0) * PI) / 180.0;
        let mut x = my.x + rad.cos() * scale * 4.5 / 16.0;
        let mut y = my.y + rad.sin() * scale * 4.5 / 16.0;
        let mut xs = rad.cos() * c::BULLETSPEED * (scale / 50.0);
        let mut ys = rad.sin() * c::BULLETSPEED * (scale / 50.0);
        let full = c::BULLETLIFETIME as f64 / 3.0; // fractional, as in the source
        let mut life = full;
        let mut deadly = c::BULLETDEADLY;
        let mut closest = c::MOVIEWIDTH + c::MOVIEHEIGHT;
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
            if deadly == 0 {
                for i in 0..g.tanks_count {
                    let tank = &g.tanks[i];
                    if tank.alive && tank.point_in_bbox(x, y) {
                        if tank.point_in_shape(x, y) {
                            return if i == me { Shot::Suicide } else { Shot::Hit { time: full - life } };
                        }
                    } else if tank.alive && i != me {
                        // Manhattan, and only near that tank in maze terms.
                        let d = (tank.x - x).abs() + (tank.y - y).abs();
                        if d < self.max_closest_distance {
                            let tf = g.tank_fields[i];
                            let cd = g.round.cell_dist(tf.x, tf.y, (x / scale).floor() as i32, (y / scale).floor() as i32);
                            if cd <= self.max_closest_cell_distance && d < closest {
                                closest = d;
                            }
                        }
                    }
                }
            }
            if deadly > 0 {
                deadly -= 1;
            }
            life -= 1.0;
        }
        Shot::Nothing { closest }
    }

    #[allow(clippy::too_many_arguments)]
    fn dodge_goal(&mut self, bx: f64, by: f64, cx: f64, cy: f64, dist: f64, t: f64, dir: (f64, f64)) -> Goal {
        let id = self.next_id();
        Goal {
            kind: GoalKind::DodgeBullet {
                x: bx, y: by, closest: (cx, cy), dist, t, dir,
                max_time: self.max_time_to_dodge, max_dist: self.max_dist_to_dodge,
            },
            priority: 1.0,
            period: 10,
            id,
            update_continuously: false,
        }
    }

    /// AI :8-88. For each bullet near me in maze terms, find its closest
    /// approach — directly and after its next bounce — and raise a dodge goal
    /// if that approach is near and unobstructed. Laika dodges its own shots too.
    fn dodge_trajectories(&mut self, g: &Game, me: usize, fx: i32, fy: i32) -> Option<Goal> {
        let scale = g.scale();
        let interval = c::BULLETHITCHECKINTERVALS;
        let max_time = self.max_time_to_dodge;
        let mut best_dist = self.max_dist_to_dodge;
        let mut result = None;
        let (tx, ty) = (g.tanks[me].x, g.tanks[me].y);
        for b in &g.bullets {
            let (bx, by) = (b.x, b.y);
            if !(g.round.cell_dist(fx, fy, (bx / scale).floor() as i32, (by / scale).floor() as i32) <= self.max_cell_dist_to_dodge) {
                continue;
            }
            let mut x2 = b.x + b.x_speed * interval as f64;
            let mut y2 = b.y + b.y_speed * interval as f64;
            let mut seg_sq = (x2 - bx) * (x2 - bx) + (y2 - by) * (y2 - by);
            let mut t = if seg_sq != 0.0 { ((tx - bx) * (x2 - bx) + (ty - by) * (y2 - by)) / seg_sq } else { 0.0 };
            if -1.0 < t && t < max_time {
                let (cx, cy) = (bx + t * (x2 - bx), by + t * (y2 - by));
                let dist = ((tx - cx) * (tx - cx) + (ty - cy) * (ty - cy)).sqrt();
                let col = Self::blocked(g, cx, cy, tx - cx, ty - cy).unwrap_or(false);
                if !col && dist < best_dist {
                    let col2 = Self::blocked(g, cx, cy, x2 - cx, y2 - cy).unwrap_or(false);
                    if !col2 {
                        best_dist = best_dist.min(dist);
                        result = Some(self.dodge_goal(b.x, b.y, cx, cy, dist, t, (x2 - bx, y2 - by)));
                    }
                }
            }
            // The threat after the bullet's next bounce (AI :50-83).
            if best_dist > scale / 4.0 {
                if let Some(c5) = Self::check_path_for_collision(g, bx, by, b.x_speed, b.y_speed, interval, 12.0, b.lifetime as f64) {
                    let (bx2, by2) = (c5.x, c5.y);
                    x2 = c5.x + c5.x_speed * interval as f64;
                    y2 = c5.y + c5.y_speed * interval as f64;
                    seg_sq = (x2 - bx2) * (x2 - bx2) + (y2 - by2) * (y2 - by2);
                    t = if seg_sq != 0.0 { ((tx - bx2) * (x2 - bx2) + (ty - by2) * (y2 - by2)) / seg_sq } else { 0.0 };
                    if 0.0 < t && t < max_time - c5.t {
                        let (cx, cy) = (bx2 + t * (x2 - bx2), by2 + t * (y2 - by2));
                        let dist = ((tx - cx) * (tx - cx) + (ty - cy) * (ty - cy)).sqrt();
                        let col = Self::blocked(g, cx, cy, tx - cx, ty - cy).unwrap_or(false);
                        if !col && dist < best_dist {
                            let col2 = Self::blocked(g, bx2, by2, cx - bx2, cy - by2).unwrap_or(false);
                            if !col2 {
                                best_dist = best_dist.min(dist);
                                result = Some(self.dodge_goal(b.x, b.y, cx, cy, dist, t + c5.t, (x2 - bx2, y2 - by2)));
                            }
                        }
                    }
                }
            }
        }
        result
    }

    /// AI :89-164, bullet branch: fire back while dodging if the barrel is on.
    fn try_to_retaliate(&mut self, g: &Game, me: usize) {
        if self.current_aggresiveness < self.aggresiveness / 2.0 {
            return;
        }
        if g.tanks[me].bullets_fired >= g.settings_max_bullets {
            return;
        }
        let closest_start = c::MOVIEWIDTH + c::MOVIEHEIGHT;
        let (found, closest) = match self.check_bullet_path(g, me, g.tanks[me].rotation) {
            Shot::Hit { .. } => (true, closest_start),
            Shot::Nothing { closest } if closest < closest_start => (false, closest),
            _ => (false, closest_start),
        };
        if found || closest < self.max_closest_distance / 2.0 {
            self.actions.push(Action::FireWeapon { delay: 1 });
            self.current_aggresiveness = (self.current_aggresiveness - 0.2).max(0.0);
        }
    }

    /// AI :337-375. LIFO: driveToField from the far end down to path[1], then
    /// driveToPos(path[0]) last so it runs first.
    fn push_actions_to_follow_path(&mut self, scale: f64, path: &[Cell]) {
        for cell in path.iter().skip(1).rev() {
            self.actions.push(Action::DriveToField { x: cell.x, y: cell.y });
        }
        if let Some(first) = path.first() {
            self.actions.push(Action::DriveToPos {
                x: (first.x as f64 + 0.5) * scale,
                y: (first.y as f64 + 0.5) * scale,
                can_reverse: path.len() <= 2,
            });
        }
    }

    /// AI :376-721. True when the action stack must be rebuilt.
    pub fn make_decisions_and_update_goal(&mut self, g: &mut Game, me: usize) -> bool {
        let scale = g.scale();
        if self.goal.period > 0 {
            self.goal.period -= 1;
            return self.goal.update_continuously;
        }
        self.goal.priority *= 0.9000000000000002;
        let old_id = self.goal.id;
        let fx = (g.tanks[me].x / scale).floor() as i32;
        let fy = (g.tanks[me].y / scale).floor() as i32;

        // goForCrate (AI :387-414): no crates in the duel; its zero-priority goal never wins.

        // dodgeBullet (AI :415-424)
        if let Some(goal) = self.dodge_trajectories(g, me, fx, fy) {
            self.update_goal(goal);
        }

        // shootAfter (AI :526-616)
        if g.tanks[me].bullets_fired < g.settings_max_bullets {
            for i in 0..g.tanks_count {
                if !(g.tanks[i].alive && i != me) {
                    continue;
                }
                let Some(dm) = g.round.dist_map(fx, fy) else { continue };
                let path = shortest_path(&g.round.maze, dm, fx, fy, g.tank_fields[i].x, g.tank_fields[i].y);
                let len = path.len() as f64;
                if len < self.longest_path_to_shoot {
                    let priority = if len <= self.longest_path_to_not_hesitate {
                        1.0
                    } else {
                        ((self.longest_path_to_shoot - len) / self.longest_path_to_shoot) * self.current_aggresiveness
                    };
                    let id = self.next_id();
                    self.update_goal(Goal { kind: GoalKind::ShootAfter { target: i }, priority, period: 10, id, update_continuously: false });
                }
            }
        }

        // runAway (AI :617-662), once the magazine is empty
        if g.alive_count > 1 && g.tanks[me].bullets_fired == g.settings_max_bullets {
            let (w, h) = (g.round.maze.w, g.round.maze.h);
            // The source's off-by-one: (W-1) x (H-1); the missing row and column read NaN.
            let mut summed = Grid::new(w - 1, h - 1, 0.0);
            for i in 0..g.tanks_count {
                let t = &g.tanks[i];
                if t.alive && i != me && t.bullets_fired != g.settings_max_bullets {
                    let dm = g.round.dist_map(g.tank_fields[i].x, g.tank_fields[i].y);
                    for xx in 0..w - 1 {
                        for yy in 0..h - 1 {
                            let add = dm.map_or(f64::NAN, |d| d.at(xx as i32, yy as i32));
                            summed.v[xx * (h - 1) + yy] += add;
                        }
                    }
                }
            }
            let here = summed.at(fx, fy);
            if here < self.longest_path_to_run {
                let priority = ((self.longest_path_to_run - here) / self.longest_path_to_run) * self.cowardness
                    * (g.tanks[me].bullets_fired as f64 / g.settings_max_bullets as f64);
                let id = self.next_id();
                self.update_goal(Goal { kind: GoalKind::RunAway { dist: summed }, priority, period: 10, id, update_continuously: false });
            }
        }

        // backAway (AI :663-672)
        self.stuck_time = if g.tanks[me].hit_something { (self.stuck_time + 1.0).min(self.max_stuck_time) } else { 0.0 };
        let id = self.next_id();
        let priority = self.stuck_time / (self.max_stuck_time - 0.1);
        self.update_goal(Goal { kind: GoalKind::BackAway, priority, period: 5, id, update_continuously: false });

        // Idle pursuit (AI :673-685)
        if g.alive_count > 1 {
            let n = g.tanks_count as f64;
            let mut k = g.rng.random_int(n) as usize;
            let mut guard = 0;
            while (k == me || !g.tanks[k].alive) && guard < 1000 {
                k = g.rng.random_int(n) as usize;
                guard += 1;
            }
            if k != me {
                let id = self.next_id();
                let field = g.tank_fields[k];
                self.update_goal(Goal {
                    kind: GoalKind::DriveTo { x: field.x, y: field.y },
                    priority: self.idle_drive_toward_enemy_priority,
                    period: 10,
                    id,
                    update_continuously: false,
                });
            }
        }

        // Goal switch (AI :686-720)
        if old_id != self.goal.id {
            if matches!(self.goal.kind, GoalKind::ShootAfter { .. }) {
                self.current_aggresiveness = (self.current_aggresiveness - 0.2).max(0.0);
            }
            return true;
        }
        self.current_aggresiveness = self.aggresiveness.min(self.current_aggresiveness + self.aggresiveness / 50.0);
        self.goal.update_continuously
    }

    /// AI :722-1051
    pub fn decide_actions_to_achieve_goal(&mut self, g: &mut Game, me: usize) {
        let scale = g.scale();
        self.actions.clear();
        let my = g.tanks[me].clone();
        let fx = (my.x / scale).floor() as i32;
        let fy = (my.y / scale).floor() as i32;
        let goal = self.goal.kind.clone();
        match goal {
            GoalKind::ShootAfter { target } => {
                let mut best_angle = my.rotation;
                let mut found = false;
                let mut best_time = c::BULLETLIFETIME as f64;
                let mut closest = c::MOVIEWIDTH + c::MOVIEHEIGHT;
                let mut angle = my.rotation;
                // Direct line of sight (AI :735-767)
                let (dx, dy) = (g.tanks[target].x - my.x, g.tanks[target].y - my.y);
                if !Self::blocked(g, my.x, my.y, dx, dy).unwrap_or(false) {
                    found = true;
                    closest = 0.0;
                    best_angle = heading_of(dx, dy, angle);
                }
                if !found {
                    // Probe +- turnSpeed * k^2 (AI :768-810)
                    for k in 1..4 {
                        match self.check_bullet_path(g, me, angle) {
                            Shot::Hit { time } => {
                                found = true;
                                if time < best_time {
                                    best_time = time;
                                    closest = 0.0;
                                    best_angle = angle;
                                }
                            }
                            Shot::Nothing { closest: cl } if !found && cl < closest => {
                                closest = cl;
                                best_angle = angle;
                            }
                            _ => {}
                        }
                        let step = my.turn_speed * (k * k) as f64;
                        if g.rng.random() < 0.5 {
                            angle += step;
                        } else {
                            angle -= step;
                        }
                        if angle < -180.0 {
                            angle += 360.0;
                        }
                        if angle > 180.0 {
                            angle -= 360.0;
                        }
                    }
                }
                if found || closest < self.max_closest_distance {
                    self.actions.push(Action::FireWeapon { delay: 5 });
                    self.actions.push(Action::TurnTo { angle: best_angle });
                } else if best_angle != my.rotation {
                    self.actions.push(Action::TurnTo { angle: best_angle });
                } else {
                    let mut a = my.rotation + 180.0;
                    if a > 180.0 {
                        a -= 360.0;
                    }
                    self.actions.push(Action::TurnTo { angle: a });
                }
            }
            GoalKind::DriveTo { x, y } => {
                if let Some(dm) = g.round.dist_map(fx, fy) {
                    let path = shortest_path(&g.round.maze, dm, fx, fy, x, y);
                    self.push_actions_to_follow_path(scale, &path);
                }
            }
            GoalKind::RunAway { dist } => {
                let path = escape_path(&g.round.maze, &dist, &g.round.dead_ends, fx, fy, 5);
                self.push_actions_to_follow_path(scale, &path);
            }
            GoalKind::BackAway => {
                // AI :909-951
                self.actions.push(Action::DriveToPos { x: (fx as f64 + 0.5) * scale, y: (fy as f64 + 0.5) * scale, can_reverse: false });
                let grid = &g.round.grid;
                let front = my.hit_check(grid, &PROBES_FRONT, 1.1);
                let rear = my.hit_check(grid, &PROBES_REAR, 1.1);
                if front {
                    if rear {
                        let left = my.hit_check(grid, &PROBES_LEFT, 1.3000000000000005);
                        self.actions.push(Action::BackupAndTurn { dist: 5, left });
                    } else {
                        self.actions.push(Action::Backup { dist: 3 });
                    }
                } else if rear {
                    // `front` is already known false, so the source's nested
                    // re-check of the front probes always takes its else branch.
                    self.actions.push(Action::Forward { dist: 3 });
                } else {
                    self.actions.push(Action::Backup { dist: 3 });
                }
            }
            GoalKind::DodgeBullet { x, y, closest, dist, t, dir, max_time, max_dist } => {
                // AI :952-1019
                let path = match g.round.dist_map((x / scale).floor() as i32, (y / scale).floor() as i32) {
                    Some(dm) => escape_path(&g.round.maze, dm, &g.round.dead_ends, fx, fy, 5),
                    None => Vec::new(),
                };
                let close_call = t < max_time / 3.0 && dist < max_dist / 5.0;
                if close_call || path.len() <= 1 {
                    // Cornered or out of time: turn parallel to the trajectory (AI :962-995).
                    let cur = my.rotation;
                    let mut a = heading_of(dir.0, dir.1, cur);
                    if (a - cur).abs() > 90.0 && (a - cur).abs() < 270.0 {
                        a += 180.0;
                        if a > 180.0 {
                            a -= 360.0;
                        }
                    }
                    a = py_round(a / my.turn_speed) * my.turn_speed;
                    self.actions.push(Action::TurnTo { angle: a });
                    if dist < scale / 4.0 {
                        // Sidestep perpendicular to the trajectory (AI :996-1012).
                        let dl = (dir.0 * dir.0 + dir.1 * dir.1).sqrt();
                        if dl > 0.0 {
                            let (px, py) = (-dir.1 / dl, dir.0 / dl);
                            let p1 = (closest.0 + (px * scale) / 2.0, closest.1 + (py * scale) / 2.0);
                            let p2 = (closest.0 - (px * scale) / 2.0, closest.1 - (py * scale) / 2.0);
                            let d1 = (my.x - p1.0).hypot(my.y - p1.1);
                            let d2 = (my.x - p2.0).hypot(my.y - p2.1);
                            let p = if d1 < d2 { p1 } else { p2 };
                            self.actions.push(Action::DriveToPos { x: p.0, y: p.1, can_reverse: true });
                        }
                    }
                } else {
                    self.push_actions_to_follow_path(scale, &path);
                }
                self.try_to_retaliate(g, me);
            }
            GoalKind::Idle => self.actions.push(Action::Idle),
        }
    }

    /// AI :1052-1354 — pop and re-push unfinished actions, then drive by the top one.
    pub fn set_input_to_do_actions(&mut self, g: &mut Game, me: usize) {
        let scale = g.scale();
        let (mx, my_y, rot, turn_speed) = {
            let t = &g.tanks[me];
            (t.x, t.y, t.rotation, t.turn_speed)
        };
        let fx = (mx / scale).floor() as i32;
        let fy = (my_y / scale).floor() as i32;

        // Part one (AI :1056-1113)
        if let Some(action) = self.actions.pop() {
            let keep = match action {
                Action::DriveToField { x, y } => {
                    (mx - (x as f64 + 0.5) * scale).abs() > scale / 3.0 || (my_y - (y as f64 + 0.5) * scale).abs() > scale / 3.0
                }
                Action::TurnTo { angle } => (rot - angle).abs() >= turn_speed,
                Action::FireWeapon { delay } => delay != 0,
                Action::DriveToPos { x, y, .. } => (mx - x).abs() > scale / 4.0 || (my_y - y).abs() > scale / 4.0,
                Action::Forward { dist } | Action::Backup { dist } | Action::BackupAndTurn { dist, .. } => dist != 0,
                Action::Idle => true,
            };
            if keep {
                let action = match action {
                    Action::FireWeapon { delay } => Action::FireWeapon { delay: delay - 1 },
                    Action::Forward { dist } => Action::Forward { dist: dist - 1 },
                    Action::Backup { dist } => Action::Backup { dist: dist - 1 },
                    Action::BackupAndTurn { dist, left } => Action::BackupAndTurn { dist: dist - 1, left },
                    other => other,
                };
                self.actions.push(action);
            }
        }

        // Part two (AI :1114-1353)
        let Some(action) = self.actions.last().cloned() else {
            g.tanks[me].set_buttons(false, false, false, false, false);
            self.goal.period = 0;
            return;
        };
        let tank = &mut g.tanks[me];
        match action {
            Action::DriveToField { x, y } => {
                let cur = rot;
                let target = if fx > x {
                    -90.0
                } else if fx < x {
                    90.0
                } else if fy > y {
                    0.0
                } else if fy < y {
                    180.0
                } else {
                    cur
                };
                turn_toward(tank, target, cur);
                tank.forward = !((target - cur).abs() > 90.0 && (target - cur).abs() < 270.0);
                tank.backup = false;
                tank.fire = false;
            }
            Action::TurnTo { angle } => {
                turn_toward(tank, angle, rot);
                tank.forward = false;
                tank.backup = false;
                tank.fire = false;
            }
            Action::FireWeapon { .. } => tank.set_buttons(false, false, false, false, true),
            Action::DriveToPos { x, y, can_reverse } => {
                let cur = rot;
                let mut reverse = false;
                let mut target = heading_of(x - mx, y - my_y, cur);
                target = turn_speed * py_round(target / turn_speed);
                if can_reverse && (target - cur).abs() > 90.0 && (target - cur).abs() < 270.0 {
                    reverse = true;
                    target += 180.0;
                    if target > 180.0 {
                        target -= 360.0;
                    }
                }
                // Turning with a dead band (AI :1268-1298)
                let gap = (target - cur).abs();
                if target > cur {
                    if gap > 180.0 {
                        tank.turn_left = gap < 360.0 - turn_speed;
                        tank.turn_right = false;
                    } else {
                        tank.turn_left = false;
                        tank.turn_right = gap > turn_speed;
                    }
                } else if target < cur {
                    if gap > 180.0 {
                        tank.turn_left = false;
                        tank.turn_right = gap < 360.0 - turn_speed;
                    } else {
                        tank.turn_left = gap > turn_speed;
                        tank.turn_right = false;
                    }
                } else {
                    tank.turn_left = false;
                    tank.turn_right = false;
                }
                if gap > 45.0 && gap < 315.0 {
                    tank.forward = false;
                    tank.backup = false;
                } else {
                    tank.forward = !reverse;
                    tank.backup = reverse;
                }
                tank.fire = false;
            }
            Action::Forward { .. } => tank.set_buttons(true, false, false, false, false),
            Action::Backup { .. } => tank.set_buttons(false, true, false, false, false),
            Action::BackupAndTurn { left, .. } => tank.set_buttons(false, true, left, !left, false),
            Action::Idle => tank.set_buttons(false, false, false, false, false),
        }
    }
}

/// AI :1140-1216 — turn the nearer way, no dead band.
fn turn_toward(tank: &mut crate::game::Tank, target: f64, cur: f64) {
    if target > cur {
        tank.turn_left = (target - cur).abs() > 180.0;
        tank.turn_right = !tank.turn_left;
    } else if target < cur {
        tank.turn_right = (target - cur).abs() > 180.0;
        tank.turn_left = !tank.turn_right;
    } else {
        tank.turn_left = false;
        tank.turn_right = false;
    }
}
