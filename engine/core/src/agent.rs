//! The deployed controller for one seat: the network behind its safety
//! shield, and in danger the teacher's short search (by default two
//! candidates, two rollouts) inside a time budget; when the budget runs out
//! the network's own sampled action stands. The search can also be planned
//! here and played out on copies kept in step elsewhere (`Plan`). Against Laika the search plays copies of it
//! kept in step with what it visibly does (`Belief`).

use std::sync::Arc;

use crate::belief::Belief;
use crate::env::{Episode, OpponentKind, Outcome, Reward};
use crate::game::{Event, Game};
use crate::obs::{self, candidate, Context, SeatHistory};
use crate::policy::Policy;
use crate::rng::Rng;
use crate::sandbox::{sandbox, Opponent};
use crate::teacher::{self, TeacherConfig};

/// Throttle none, turn none, no fire.
pub const IDLE: usize = 8;

#[derive(Clone, Debug, Default)]
pub struct Decision {
    pub action: usize,
    /// After the shield.
    pub probs: [f32; 18],
    /// The network's favourite move was dropped by the shield.
    pub vetoed: bool,
    pub searched: bool,
    /// The search replaced the network's sampled action.
    pub changed: bool,
    /// Candidates the search finished within the budget, of `planned`.
    pub scored: usize,
    pub planned: usize,
    /// Survival lookahead of the nine moves (>= 0 survives).
    pub dodge: [f32; 9],
}

/// A danger frame's search, playable on any copy of the agent kept in step:
/// the episode's sandbox seed, the candidates (the sampled move first) and
/// the seeds of the futures they all face.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub sandbox_seed: u64,
    pub candidates: Vec<usize>,
    pub seeds: Vec<u64>,
}

pub struct Agent {
    pub seat: usize,
    policy: Arc<Policy>,
    /// None: the network alone.
    pub search: Option<TeacherConfig>,
    pub budget_ms: f64,
    /// Against Laika: its inferred hidden state.
    pub belief: Option<Belief>,
    before: Option<Game>,
    history: SeatHistory,
    prev_pose: [[f64; 3]; 2],
    last_fire: [Option<u32>; 2],
    frames: u32,
    rng: Rng,
    obs: Vec<f32>,
    mask: Vec<u8>,
    pub last: Decision,
}

fn poses(g: &Game) -> [[f64; 3]; 2] {
    [0, 1].map(|i| [g.tanks[i].x, g.tanks[i].y, g.tanks[i].rotation])
}

impl Agent {
    pub fn new(policy: Arc<Policy>, seat: usize, g: &Game, seed: u64) -> Agent {
        assert!(policy.obs_dim == obs::OBS_DIM && policy.layout == obs::LAYOUT, "the policy reads another observation");
        let search = Some(TeacherConfig { candidates: 2, rollouts: 2, ..TeacherConfig::default() });
        Agent {
            seat,
            obs: vec![0.0; policy.obs_dim],
            mask: vec![0; policy.layout[3]],
            policy,
            search,
            budget_ms: 30.0,
            belief: None,
            before: None,
            history: SeatHistory::default(),
            prev_pose: poses(g),
            last_fire: [None, None],
            frames: 0,
            rng: Rng::new(seed),
            last: Decision::default(),
        }
    }

    /// The opponent is Laika: track its hidden state with `particles` copies.
    pub fn track_laika(&mut self, g: &Game, particles: usize) {
        self.belief = Some(Belief::new(g, 1 - self.seat, particles, self.rng.randrange(1 << 30) as u64));
    }

    /// Choose this frame's action; `now` is a millisecond clock.
    pub fn decide(&mut self, g: &Game, now: &mut dyn FnMut() -> f64) -> usize {
        let Some(plan) = self.plan(g) else { return self.last.action };
        let jobs: Vec<(usize, usize)> = (0..plan.candidates.len()).flat_map(|i| (0..plan.seeds.len()).map(move |j| (i, j))).collect();
        let budget = self.budget_ms;
        let values = self.rollouts(g, &plan, &jobs, budget, now);
        self.conclude(&plan, &values)
    }

    /// The network's move; in danger with search on, the search still to
    /// play out (`rollouts`, then `conclude`), else the move is final.
    pub fn plan(&mut self, g: &Game) -> Option<Plan> {
        let seat = self.seat;
        let ctx = Context { prev_pose: &self.prev_pose, history: &self.history, frames: self.frames, last_fire: self.last_fire };
        obs::encode(g, seat, &ctx, &mut self.obs, &mut self.mask);
        let logits = self.policy.logits(&self.obs, &self.mask);
        let probs = self.policy.shield(&logits, &self.obs);
        let favourite = (0..18).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap();
        let mut d = Decision { probs, vetoed: probs[favourite] == 0.0, ..Decision::default() };
        d.dodge.copy_from_slice(&self.obs[self.policy.layout[5]..self.policy.layout[5] + 9]);
        d.action = teacher::sample(&probs, &mut self.rng);
        if !g.tanks[seat].alive || g.frozen {
            d.action = IDLE;
        } else if let Some(cfg) = self.search.filter(|_| teacher::danger_for(g, seat, &self.policy, &self.obs)) {
            let sandbox_seed = self.rng.randrange(1 << 30) as u64;
            let (candidates, seeds) = teacher::plan(&probs, d.action, &cfg, &mut self.rng);
            self.last = d;
            return Some(Plan { sandbox_seed, candidates, seeds });
        }
        self.adopt(d);
        None
    }

    /// Play `jobs` of `plan` (candidate, seed index) in order on `g`, the game
    /// it was planned on, while within `budget_ms` of `now`; NaN for the
    /// rest. The first job always runs.
    pub fn rollouts(&self, g: &Game, plan: &Plan, jobs: &[(usize, usize)], budget_ms: f64, now: &mut dyn FnMut() -> f64) -> Vec<f32> {
        let cfg = self.search.unwrap_or_default();
        let ep = self.episode(g, plan.sandbox_seed);
        let copies = self.belief.as_ref().map_or(Vec::new(), |b| b.seen_from(self.seat));
        let start = now();
        let (mut last, mut longest) = (start, 0.0f64);
        let mut out = vec![f32::NAN; jobs.len()];
        for (k, &(i, j)) in jobs.iter().enumerate() {
            if k > 0 {
                let t = now();
                longest = longest.max(t - last);
                last = t;
                if t - start + longest > budget_ms {
                    break;
                }
            }
            out[k] = teacher::rollout_against(&ep, &self.policy, plan.candidates[i], plan.seeds[j], &cfg, &copies);
        }
        out
    }

    /// The planned search's move from its rollouts, candidate-major (NaN
    /// where unplayed): a candidate counts once all its rollouts are in, and
    /// replaces the sampled move only if strictly better.
    pub fn conclude(&mut self, plan: &Plan, values: &[f32]) -> usize {
        let mut d = std::mem::take(&mut self.last);
        let n = plan.seeds.len();
        let means: Vec<f32> = values.chunks(n).map(|c| c.iter().sum::<f32>() / n as f32).collect();
        d.searched = true;
        d.scored = means.iter().filter(|v| !v.is_nan()).count();
        d.planned = means.len();
        let best = (0..means.len()).filter(|&i| !means[i].is_nan()).max_by(|&a, &b| means[a].total_cmp(&means[b]));
        if let Some(best) = best.filter(|&b| !means[0].is_nan() && means[b] > means[0]) {
            d.action = plan.candidates[best];
            d.changed = true;
        }
        self.adopt(d);
        self.last.action
    }

    /// A copy kept in step elsewhere: take the action the real agent chose.
    pub fn follow(&mut self, action: usize) {
        self.adopt(Decision { action, ..Decision::default() });
    }

    /// Take `d` as this frame's decision.
    fn adopt(&mut self, d: Decision) {
        self.history.record(d.action);
        self.last = d;
    }

    /// Call after the action is written and before the game steps.
    pub fn before_step(&mut self, g: &Game) {
        self.prev_pose = poses(g);
        if self.belief.is_some() {
            self.before = Some(g.clone());
        }
    }

    /// Call with the events of the step just taken.
    pub fn after_step(&mut self, g: &Game, events: &[Event]) {
        self.frames += 1;
        if let (Some(b), Some(before)) = (self.belief.as_mut(), self.before.take()) {
            b.update(&before, self.last.action, g, events);
        }
        for e in events {
            match e {
                Event::Fire(t) => self.last_fire[*t] = Some(self.frames),
                Event::NewRound(_) => {
                    self.history = SeatHistory::default();
                    self.last_fire = [None, None];
                    self.frames = 0;
                    self.prev_pose = poses(g);
                }
                _ => {}
            }
        }
    }

    /// The round as an episode seen from this seat (tank 0 = me).
    fn episode(&self, g: &Game, sandbox_seed: u64) -> Episode {
        let game = sandbox(g, self.seat, Opponent::Hold, sandbox_seed);
        let mut prev_pose = self.prev_pose;
        let mut last_fire = self.last_fire;
        if self.seat == 1 {
            prev_pose.swap(0, 1);
            last_fire.swap(0, 1);
        }
        Episode {
            game,
            opponent: OpponentKind::Laika,
            reward: Reward::default(),
            frames: self.frames,
            outcome: Outcome::Running,
            prev_pose,
            history: [self.history.clone(), SeatHistory::default()],
            kills: Vec::new(),
            last_fire,
        }
    }
}

/// Write an action's buttons to a tank.
pub fn press(g: &mut Game, tank: usize, action: usize) {
    let (throttle, turn, fire) = candidate(action);
    g.tanks[tank].set_buttons(throttle == 2, throttle == 0, turn == 0, turn == 2, fire);
}

/// The demo's random driver: a random move held 3-10 frames, firing 8% of picks.
pub struct RandomDriver {
    rng: Rng,
    remaining: u32,
    action: usize,
}

impl RandomDriver {
    pub fn new(seed: u64) -> RandomDriver {
        RandomDriver { rng: Rng::new(seed), remaining: 0, action: IDLE }
    }

    pub fn decide(&mut self) -> usize {
        if self.remaining == 0 {
            let (throttle, turn) = (self.rng.randrange(3) as usize, self.rng.randrange(3) as usize);
            let fire = self.rng.random() < 0.08;
            self.action = throttle * 6 + turn * 2 + fire as usize;
            self.remaining = 3 + self.rng.randrange(8);
        }
        self.remaining -= 1;
        self.action
    }
}
