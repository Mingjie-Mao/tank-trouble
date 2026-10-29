//! The web demo's arena in WebAssembly: the Rust engine, native Laika and the
//! deployed controller (`tank_core::agent`), stepped one frame per call. The
//! page's worker drives it and reads back JSON snapshots in the shape the
//! renderer draws.

use std::collections::VecDeque;
use std::sync::Arc;

use serde_json::{json, Value};
use tank_core::agent::{press, Agent, Plan, RandomDriver};
use tank_core::constants as c;
use tank_core::game::{Event, Game};
use tank_core::laika::Laika;
use tank_core::obs::{self, candidate, Context, SeatHistory};
use tank_core::policy::Policy;
use tank_core::teacher::TeacherConfig;
use wasm_bindgen::prelude::*;

static CHAMPION: &[u8] = include_bytes!("../../models/champion.policy");

const AI: &str = "ppo-league";
const LAIKA: &str = "laika-js";
/// Copies of Laika the search keeps in step with the real one.
const PARTICLES: usize = 16;
const RANDOM: &str = "random-js";
const HUMAN: &str = "human";
const BULLET_VISUAL_RADIUS: f64 = 3.5;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = performance, js_name = now)]
    fn performance_now() -> f64;
}

/// Milliseconds on a monotonic clock.
fn now_ms() -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        performance_now()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::OnceLock;
        use std::time::Instant;
        static START: OnceLock<Instant> = OnceLock::new();
        START.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0
    }
}

fn champion() -> Arc<Policy> {
    Arc::new(Policy::from_bytes(CHAMPION).expect("embedded policy"))
}

enum Driver {
    Ai(Box<Agent>),
    Laika,
    Random(RandomDriver),
    Human,
}

fn label(name: &str) -> &'static str {
    match name {
        HUMAN => "你",
        AI => "Our AI",
        LAIKA => "Laika",
        RANDOM => "Random",
        _ => "?",
    }
}

fn controls(v: &Value) -> [bool; 5] {
    ["forward", "backup", "turn_left", "turn_right", "fire"].map(|k| v.get(k).and_then(Value::as_bool).unwrap_or(false))
}

fn percentile(values: &VecDeque<f64>, q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v: Vec<f64> = values.iter().copied().collect();
    v.sort_by(|a, b| a.total_cmp(b));
    v[((q * (v.len() - 1) as f64).floor() as usize).min(v.len() - 1)]
}

/// Our AI's search: the page widens it when other workers share the rollouts.
#[derive(Clone, Copy)]
struct Search {
    candidates: usize,
    rollouts: usize,
    budget_ms: f64,
}

impl Search {
    fn apply(self, agent: &mut Agent) {
        agent.search = Some(TeacherConfig { candidates: self.candidates, rollouts: self.rollouts, ..TeacherConfig::default() });
        agent.budget_ms = self.budget_ms;
    }
}

/// The page's whole match: two drivers, scores, streaks and telemetry.
#[wasm_bindgen]
pub struct Arena {
    game: Game,
    policy: Arc<Policy>,
    mode: String,
    names: [String; 2],
    drivers: [Driver; 2],
    human: [bool; 5],
    paused: bool,
    vision: bool,
    streak: u32,
    best_streak: u32,
    decision_ms: [f64; 2],
    samples: VecDeque<f64>,
    frames: u64,
    searches: u64,
    changes: u64,
    /// Searches the budget cut short.
    cut: u64,
    last_reason: &'static str,
    search: Search,
    /// This frame's moves so far, and the searches still out (`plan`).
    planned: [Option<usize>; 2],
    pending: Vec<(usize, Plan)>,
    /// What the last frame pressed, for copies kept in step (`frame`).
    pressed: [Option<usize>; 2],
    pressed_human: [bool; 5],
}

#[wasm_bindgen]
impl Arena {
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u32) -> Arena {
        let policy = champion();
        let mut arena = Arena {
            game: Game::new(seed as u64, true),
            policy,
            mode: "watch".into(),
            names: [AI.into(), LAIKA.into()],
            drivers: [Driver::Human, Driver::Laika],
            human: [false; 5],
            paused: false,
            vision: false,
            streak: 0,
            best_streak: 0,
            decision_ms: [0.0; 2],
            samples: VecDeque::new(),
            frames: 0,
            searches: 0,
            changes: 0,
            cut: 0,
            last_reason: "initialising",
            search: Search { candidates: 2, rollouts: 2, budget_ms: 30.0 },
            planned: [None; 2],
            pending: Vec::new(),
            pressed: [None; 2],
            pressed_human: [false; 5],
        };
        arena.game.events.clear();
        arena.configure();
        arena
    }

    /// Advance one frame and return the snapshot as JSON.
    pub fn step(&mut self) -> String {
        if !self.paused {
            self.advance();
        }
        self.state()
    }

    /// Apply a UI command (the BrowserArena protocol) and return the snapshot.
    pub fn command(&mut self, payload: &str) -> Result<String, JsError> {
        let p: Value = serde_json::from_str(payload).map_err(|e| JsError::new(&e.to_string()))?;
        let s = |k: &str, d: &str| p.get(k).and_then(Value::as_str).unwrap_or(d).to_string();
        match p.get("action").and_then(Value::as_str).unwrap_or("") {
            "input" => self.human = controls(p.get("controls").unwrap_or(&Value::Null)),
            "mode" => {
                let mode = s("mode", &self.mode.clone());
                let pick = |k: &str, d: &str| {
                    let v = s(k, d);
                    if [AI, LAIKA, RANDOM].contains(&v.as_str()) { Ok(v) } else { Err(JsError::new(&format!("unknown policy {v}"))) }
                };
                self.names = match mode.as_str() {
                    "watch" => [pick("left_policy", AI)?, LAIKA.into()],
                    "play" => [HUMAN.into(), pick("right_policy", AI)?],
                    "selfplay" => [pick("left_policy", AI)?, pick("right_policy", AI)?],
                    _ => return Err(JsError::new("invalid mode")),
                };
                self.mode = mode;
                self.configure();
            }
            "new_maze" => {
                let seed = match p.get("seed") {
                    Some(Value::Number(n)) => n.as_f64().unwrap_or(0.0) as u64,
                    Some(Value::String(v)) if !v.is_empty() && v != "random" => {
                        v.parse::<f64>().map_err(|_| JsError::new("invalid seed"))? as u64
                    }
                    _ => (js_random() * 2147483647.0) as u64,
                };
                let scores = self.game.scores.clone();
                self.game = Game::new(seed & 0xffff_ffff, false);
                self.game.events.clear();
                self.game.scores = scores;
                self.configure();
            }
            "reset_score" => {
                self.game.scores = vec![0, 0];
                self.streak = 0;
                self.best_streak = 0;
            }
            "search" => {
                let n = |k: &str, d: f64| p.get(k).and_then(Value::as_f64).unwrap_or(d);
                self.search = Search {
                    candidates: n("candidates", 2.0) as usize,
                    rollouts: n("rollouts", 2.0) as usize,
                    budget_ms: n("budget_ms", 30.0),
                };
                for driver in &mut self.drivers {
                    if let Driver::Ai(agent) = driver {
                        self.search.apply(agent);
                    }
                }
            }
            "pause" => self.paused = p.get("paused").and_then(Value::as_bool).unwrap_or(!self.paused),
            "vision" => self.vision = p.get("enabled").and_then(Value::as_bool).unwrap_or(false),
            other => return Err(JsError::new(&format!("unknown action: {other}"))),
        }
        Ok(self.state())
    }

    /// The snapshot as JSON.
    pub fn state(&self) -> String {
        self.snapshot().to_string()
    }

    /// `step` in three parts, so searches can run on copies in other
    /// workers: decide every seat and return the searches still to play
    /// out, as JSON [{seat, sandbox_seed, candidates, seeds}]; play their
    /// rollouts (`rollouts`, here or on a copy); then `finish`.
    pub fn plan(&mut self) -> String {
        self.planned = [None; 2];
        self.pending.clear();
        if self.paused {
            return "[]".into();
        }
        self.decision_ms = [0.0; 2];
        for side in 0..2 {
            let t0 = now_ms();
            self.planned[side] = match &mut self.drivers[side] {
                Driver::Ai(agent) => match agent.plan(&self.game) {
                    Some(plan) => {
                        self.pending.push((side, plan));
                        None
                    }
                    None => Some(agent.last.action),
                },
                Driver::Random(r) => Some(r.decide()),
                Driver::Human | Driver::Laika => None,
            };
            self.decision_ms[side] = now_ms() - t0;
        }
        Value::from(self.pending.iter().map(|(seat, p)| json!({
            "seat": seat, "sandbox_seed": p.sandbox_seed, "candidates": p.candidates, "seeds": p.seeds,
        })).collect::<Vec<_>>()).to_string()
    }

    /// Play a planned search's `jobs` ([candidate, seed index] pairs) within
    /// `budget_ms`; the values in job order, null where unplayed.
    pub fn rollouts(&self, request: &str) -> Result<String, JsError> {
        let r: Value = serde_json::from_str(request).map_err(|e| JsError::new(&e.to_string()))?;
        let seat = r["seat"].as_u64().unwrap_or(0) as usize;
        let Driver::Ai(agent) = &self.drivers[seat.min(1)] else { return Err(JsError::new("no AI in that seat")) };
        let plan = plan_from(&r);
        let jobs: Vec<(usize, usize)> = r["jobs"].as_array().into_iter().flatten()
            .map(|j| (j[0].as_u64().unwrap_or(0) as usize, j[1].as_u64().unwrap_or(0) as usize))
            .filter(|&(i, j)| i < plan.candidates.len() && j < plan.seeds.len()).collect();
        let values = agent.rollouts(&self.game, &plan, &jobs, r["budget_ms"].as_f64().unwrap_or(self.search.budget_ms), &mut now_ms);
        Ok(json!(values).to_string())
    }

    /// Conclude `plan`'s searches from their values (one candidate-major
    /// array per search, null where unplayed), step, and return the snapshot.
    /// `ms` is how long the searches took.
    pub fn finish(&mut self, values: &str, ms: f64) -> Result<String, JsError> {
        let v: Value = serde_json::from_str(values).map_err(|e| JsError::new(&e.to_string()))?;
        if self.paused {
            return Ok(self.state());
        }
        for (k, (side, plan)) in std::mem::take(&mut self.pending).into_iter().enumerate() {
            let values: Vec<f32> = (0..plan.candidates.len() * plan.seeds.len())
                .map(|i| v[k][i].as_f64().map_or(f32::NAN, |x| x as f32)).collect();
            if let Driver::Ai(agent) = &mut self.drivers[side] {
                self.planned[side] = Some(agent.conclude(&plan, &values));
            }
            self.decision_ms[side] += ms;
        }
        self.apply(self.planned);
        Ok(self.state())
    }

    /// The last frame's moves, for copies kept in step (`follow`).
    pub fn frame(&self) -> String {
        json!({ "actions": self.pressed, "human": self.pressed_human, "check": self.check() }).to_string()
    }

    /// As a copy: take the frame the real arena just stepped; false if the
    /// two have drifted apart.
    pub fn follow(&mut self, frame: &str) -> Result<bool, JsError> {
        let f: Value = serde_json::from_str(frame).map_err(|e| JsError::new(&e.to_string()))?;
        let actions = [0, 1].map(|i| f["actions"][i].as_u64().map(|a| a as usize));
        self.human = controls_array(&f["human"]);
        for side in 0..2 {
            if let (Driver::Ai(agent), Some(a)) = (&mut self.drivers[side], actions[side]) {
                agent.follow(a);
            }
        }
        self.apply(actions);
        Ok(f["check"].as_str() == Some(self.check().as_str()))
    }
}

fn plan_from(r: &Value) -> Plan {
    let list = |k: &str| r[k].as_array().into_iter().flatten().filter_map(Value::as_u64).collect::<Vec<_>>();
    Plan {
        sandbox_seed: r["sandbox_seed"].as_u64().unwrap_or(0),
        candidates: list("candidates").into_iter().map(|a| a as usize).collect(),
        seeds: list("seeds"),
    }
}

fn controls_array(v: &Value) -> [bool; 5] {
    [0, 1, 2, 3, 4].map(|i| v[i].as_bool().unwrap_or(false))
}

#[cfg(target_arch = "wasm32")]
fn js_random() -> f64 {
    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = Math, js_name = random)]
        fn math_random() -> f64;
    }
    math_random()
}

#[cfg(not(target_arch = "wasm32"))]
fn js_random() -> f64 {
    0.5
}

impl Arena {
    /// Rebuild both drivers for the current names; Laika runs natively.
    fn configure(&mut self) {
        let seed = self.game.seed;
        for side in 0..2 {
            let name = self.names[side].clone();
            let laika = name == LAIKA;
            if side == 0 {
                self.game.laika_on_tank0 = laika;
            } else {
                self.game.laika_on_tank1 = laika;
            }
            self.game.ai[side] = laika.then(|| Box::new(Laika::new(self.game.scale())));
            self.drivers[side] = match name.as_str() {
                AI => {
                    let mut agent = Agent::new(self.policy.clone(), side, &self.game, seed + 101 + side as u64 * 7818);
                    self.search.apply(&mut agent);
                    if self.names[1 - side] == LAIKA {
                        agent.track_laika(&self.game, PARTICLES);
                    }
                    Driver::Ai(Box::new(agent))
                }
                LAIKA => Driver::Laika,
                RANDOM => Driver::Random(RandomDriver::new(seed + 7919 + side as u64)),
                _ => Driver::Human,
            };
        }
        self.decision_ms = [0.0; 2];
    }

    fn advance(&mut self) {
        self.decision_ms = [0.0; 2];
        let mut actions = [None; 2];
        for side in 0..2 {
            let t0 = now_ms();
            actions[side] = match &mut self.drivers[side] {
                Driver::Ai(agent) => Some(agent.decide(&self.game, &mut now_ms)),
                Driver::Random(r) => Some(r.decide()),
                Driver::Human | Driver::Laika => None,
            };
            self.decision_ms[side] = now_ms() - t0;
        }
        self.apply(actions);
    }

    /// Press this frame's moves (None: the human's buttons) and step.
    fn apply(&mut self, actions: [Option<usize>; 2]) {
        let frozen = self.game.frozen;
        for side in 0..2 {
            if matches!(self.drivers[side], Driver::Laika) {
                continue;
            }
            match actions[side] {
                _ if frozen => self.game.tanks[side].set_buttons(false, false, false, false, false),
                Some(a) => press(&mut self.game, side, a),
                None => {
                    let [f, b, l, r, fire] = self.human;
                    self.game.tanks[side].set_buttons(f, b, l, r, fire);
                }
            }
            if side == 0 {
                self.record(side);
            }
        }
        self.pressed = actions;
        self.pressed_human = self.human;
        for driver in &mut self.drivers {
            if let Driver::Ai(agent) = driver {
                agent.before_step(&self.game);
            }
        }
        let events = self.game.step().to_vec();
        for driver in &mut self.drivers {
            if let Driver::Ai(agent) = driver {
                agent.after_step(&self.game, &events);
            }
        }
        for e in &events {
            if let Event::RoundEnd(w) = e {
                if *w == Some(0) {
                    self.streak += 1;
                    self.best_streak = self.best_streak.max(self.streak);
                } else {
                    self.streak = 0;
                }
            }
        }
    }

    /// A fingerprint of the game, to catch copies drifting apart (as bits:
    /// JSON numbers need not round-trip).
    fn check(&self) -> String {
        let g = &self.game;
        let sum = g.tanks.iter().map(|t| t.x + 3.0 * t.y + 7.0 * t.rotation).sum::<f64>() + g.frame as f64 + 0.5 * g.bullets.len() as f64;
        format!("{:016x}", sum.to_bits())
    }

    fn record(&mut self, side: usize) {
        self.frames += 1;
        self.samples.push_back(self.decision_ms[side]);
        if self.samples.len() > 600 {
            self.samples.pop_front();
        }
        self.last_reason = match &self.drivers[side] {
            Driver::Ai(agent) if agent.last.searched => {
                self.searches += 1;
                self.changes += agent.last.changed as u64;
                self.cut += (agent.last.scored < agent.last.planned) as u64;
                "search"
            }
            Driver::Ai(_) => "network",
            Driver::Random(_) => "random",
            Driver::Laika => "laika",
            Driver::Human => "human input",
        };
    }

    fn snapshot(&self) -> Value {
        let g = &self.game;
        let scale = g.scale();
        let left = if self.mode == "play" { HUMAN } else { self.names[0].as_str() };
        json!({
            "connected": true,
            "mode": self.mode,
            "paused": self.paused,
            "seed": g.seed,
            "frame": g.frame,
            "round": g.round_number,
            "fps": c::FPS,
            "frozen": g.frozen,
            "world_width": g.round.maze.w as f64 * scale,
            "world_height": g.round.maze.h as f64 * scale,
            "wall_width": g.round.wall_half_t * 2.0,
            "bullet_radius": BULLET_VISUAL_RADIUS,
            "walls": g.round.walls,
            "scores": g.scores,
            "streak": self.streak,
            "best_streak": self.best_streak,
            "left_policy": left,
            "right_policy": self.names[1],
            "left_label": label(left),
            "right_label": label(&self.names[1]),
            "tanks": g.tanks.iter().enumerate().map(|(i, t)| json!({
                "index": i, "x": t.x, "y": t.y, "rotation": t.rotation, "display_scale": t.display_scale,
                "alive": t.alive, "bullets_fired": t.bullets_fired,
            })).collect::<Vec<_>>(),
            "bullets": g.bullets.iter().filter(|b| !b.removed).map(|b| json!({
                "id": format!("bullet{}", b.id), "x": b.x, "y": b.y, "owner": b.owner, "lifetime": b.lifetime,
            })).collect::<Vec<_>>(),
            "runtime": { "left_decision_ms": self.decision_ms[0], "right_decision_ms": self.decision_ms[1] },
            "telemetry": {
                "frames": self.frames,
                "searches": self.searches,
                "search_rate": self.searches as f64 / self.frames.max(1) as f64,
                "search_changes": self.changes,
                "change_rate": self.changes as f64 / self.searches.max(1) as f64,
                "cut_searches": self.cut,
                "last_decision_ms": self.decision_ms[0],
                "p50_ms": percentile(&self.samples, 0.5),
                "p95_ms": percentile(&self.samples, 0.95),
                "last_reason": if self.mode == "play" { "human input" } else { self.last_reason },
                "search": format!("{}×{}", self.search.candidates, self.search.rollouts),
            },
            "available_policies": [
                { "value": AI, "label": "我们的 AI（网络 + 实时搜索）" },
                { "value": LAIKA, "label": "Laika（原版 AI）" },
                { "value": RANDOM, "label": "Random（随机）" },
            ],
            "vision": if self.vision && self.mode != "play" { self.vision_json() } else { Value::Null },
        })
    }

    /// Tank 0's view: which moves survive, where a shot now would fly, the
    /// bullets bound for it, and (for our AI) its top actions.
    fn vision_json(&self) -> Value {
        let g = &self.game;
        if !g.tanks[0].alive || g.frozen {
            return Value::Null;
        }
        let dodge = match &self.drivers[0] {
            Driver::Ai(agent) => agent.last.dodge,
            _ => {
                let (mut out, mut mask) = (vec![0.0; obs::OBS_DIM], vec![0u8; obs::BULLET_SLOTS]);
                let history = SeatHistory::default();
                let prev = [0, 1].map(|i| [g.tanks[i].x, g.tanks[i].y, g.tanks[i].rotation]);
                let ctx = Context { prev_pose: &prev, history: &history, frames: 0, last_fire: [None, None] };
                obs::encode(g, 0, &ctx, &mut out, &mut mask);
                let mut d = [0.0; 9];
                d.copy_from_slice(&out[obs::DODGE_OFFSET..obs::DODGE_OFFSET + 9]);
                d
            }
        };
        let (points, hit) = obs::shot_path(g, 0);
        let mut v = json!({
            "dodge": dodge.iter().map(|&d| d >= 0.0).collect::<Vec<_>>(),
            "shot": {
                "points": points.iter().map(|&(x, y)| [x, y]).collect::<Vec<_>>(),
                "hit": match hit { Some(0) => "self", Some(_) => "opponent", None => "none" },
            },
            "threats": obs::threat_ids(g, 0).iter().map(|id| format!("bullet{id}")).collect::<Vec<_>>(),
        });
        if let Driver::Ai(agent) = &self.drivers[0] {
            let p = agent.last.probs;
            let mut order: Vec<usize> = (0..18).collect();
            order.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
            v["actions"] = order.iter().take(3).filter(|&&a| p[a] > 0.005).map(|&a| {
                let (throttle, turn, fire) = candidate(a);
                json!([throttle, turn, fire as u8, p[a]])
            }).collect();
            v["shieldVetoed"] = json!(agent.last.vetoed);
            v["searched"] = json!(agent.last.searched);
        }
        v
    }
}

/// A lockstep copy for matches against controllers that live outside Rust:
/// our controller drives `seat`, the caller presses the other tank's buttons
/// and checks that both engines still agree.
#[wasm_bindgen]
pub struct Duel {
    game: Game,
    agent: Agent,
}

#[wasm_bindgen]
impl Duel {
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u32, seat: usize, search: bool) -> Duel {
        let mut game = Game::new(seed as u64, false);
        game.events.clear();
        let mut agent = Agent::new(champion(), seat, &game, seed as u64 + 101);
        if !search {
            agent.search = None;
        }
        Duel { game, agent }
    }

    /// Our action for this frame, as [throttle, turn, fire].
    pub fn decide(&mut self) -> Vec<u8> {
        let a = self.agent.decide(&self.game, &mut now_ms);
        let (throttle, turn, fire) = candidate(a);
        press(&mut self.game, self.agent.seat, a);
        vec![throttle as u8, turn as u8, fire as u8]
    }

    /// Step with the other tank's buttons; returns the events, comma-joined per line.
    pub fn step(&mut self, forward: bool, backup: bool, left: bool, right: bool, fire: bool) -> String {
        let other = 1 - self.agent.seat;
        self.game.tanks[other].set_buttons(forward, backup, left, right, fire);
        self.agent.before_step(&self.game);
        let events = self.game.step().to_vec();
        self.agent.after_step(&self.game, &events);
        events.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("\n")
    }

    /// [x, y, rotation] of both tanks, for drift checks.
    pub fn poses(&self) -> Vec<f64> {
        self.game.tanks.iter().flat_map(|t| [t.x, t.y, t.rotation]).collect()
    }

    pub fn searched(&self) -> bool {
        self.agent.last.searched
    }
}
