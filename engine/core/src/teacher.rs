//! The MPC teacher: in danger, each of the policy's most likely actions is
//! played out by the shielded policy itself in a fair sandbox (an independent
//! random stream; Laika rebuilt from scratch, or one of the copies `Belief`
//! keeps in step with what the real one visibly does), and the outcomes are
//! scored. It labels states for distillation and searches in real time.

use crate::env::{Episode, Outcome};
use crate::game::Game;
use crate::laika::Laika;
use crate::aim::laika_aim_hits;
use crate::policy::Policy;
use crate::rng::Rng;
use crate::sandbox::{sandbox, Opponent};

#[derive(Clone, Copy, Debug)]
pub struct TeacherConfig {
    pub candidates: usize,
    pub rollouts: usize,
    pub horizon: usize,
    pub double_ko: f32,
}

impl Default for TeacherConfig {
    fn default() -> Self {
        TeacherConfig { candidates: 4, rollouts: 4, horizon: 24, double_ko: -0.5 }
    }
}

/// An exchange is on: either barrel holds a hit, or some move is forecast to die.
pub fn danger(g: &Game, policy: &Policy, obs: &[f32]) -> bool {
    danger_for(g, 0, policy, obs)
}

/// `danger` for seat `seat`, from its own observation.
pub fn danger_for(g: &Game, seat: usize, policy: &Policy, obs: &[f32]) -> bool {
    let other = 1 - seat;
    let (me, them) = (&g.tanks[seat], &g.tanks[other]);
    if !me.alive || !them.alive || g.frozen {
        return false;
    }
    let dodge = &obs[policy.layout[5]..policy.layout[5] + 9];
    dodge.iter().any(|&d| d < 0.0)
        || laika_aim_hits(g, seat, other, me.rotation)
        || laika_aim_hits(g, other, seat, them.rotation)
}

pub fn sample(p: &[f32; 18], rng: &mut Rng) -> usize {
    let mut u = rng.random() as f32;
    let mut action = 0;
    for (a, &q) in p.iter().enumerate() {
        if q <= 0.0 {
            continue;
        }
        action = a;
        u -= q;
        if u < 0.0 {
            break;
        }
    }
    action
}

/// Play `first`, then the shielded policy, for the horizon in a fair sandbox
/// against `laika` (a fresh one if None).
fn rollout(ep: &Episode, policy: &Policy, first: usize, seed: u64, cfg: &TeacherConfig, laika: Option<&Laika>) -> f32 {
    let mut r = ep.clone();
    r.game = sandbox(&ep.game, 0, Opponent::FreshLaika, seed);
    if let Some(l) = laika {
        r.game.ai[1] = Some(Box::new(l.clone()));
    }
    r.outcome = Outcome::Running;
    let mut rng = Rng::new(seed ^ 0x9e37_79b9);
    let mut obs = vec![0f32; policy.obs_dim];
    let mut mask = vec![0u8; policy.layout[3]];
    let mut action = first;
    for t in 0..cfg.horizon {
        if r.game.frozen {
            break;
        }
        if t > 0 {
            if !r.game.tanks[0].alive {
                break;
            }
            r.observe(0, &mut obs, &mut mask);
            action = sample(&policy.shielded_probs(&obs, &mask), &mut rng);
        }
        r.step(action, None);
    }
    match (r.game.tanks[0].alive, r.game.tanks[1].alive) {
        (false, true) => -1.0,
        (false, false) => cfg.double_ko,
        (true, false) => 1.0,
        (true, true) => 0.0,
    }
}

/// Mean rollout outcome of the policy's top `candidates` actions (plus
/// `sampled`), all facing the same futures: each rollout's seed also
/// picks its Laika from `belief`, or a fresh Laika when `belief` is empty. NaN for actions
/// not scored.
pub fn scores(ep: &Episode, policy: &Policy, probs: &[f32; 18], sampled: usize, cfg: &TeacherConfig, rng: &mut Rng,
              belief: &[Laika]) -> [f32; 18] {
    let (ranked, seeds) = plan(probs, sampled, cfg, rng);
    let mut out = [f32::NAN; 18];
    for a in ranked {
        out[a] = seeds.iter().map(|&s| rollout_against(ep, policy, a, s, cfg, belief)).sum::<f32>() / cfg.rollouts as f32;
    }
    out
}

/// The actions a search scores (`sampled` first, then the policy's top
/// `candidates` by probability) and the seeds of the futures they all face.
pub fn plan(probs: &[f32; 18], sampled: usize, cfg: &TeacherConfig, rng: &mut Rng) -> (Vec<usize>, Vec<u64>) {
    let mut ranked: Vec<usize> = (0..18).filter(|&a| probs[a] > 0.0).collect();
    ranked.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
    ranked.truncate(cfg.candidates);
    ranked.retain(|&a| a != sampled);
    ranked.insert(0, sampled);
    (ranked, (0..cfg.rollouts).map(|_| rng.randrange(1 << 30) as u64).collect())
}

/// One rollout of `first`; the seed also picks its Laika from `belief`
/// (a fresh one when empty).
pub fn rollout_against(ep: &Episode, policy: &Policy, first: usize, seed: u64, cfg: &TeacherConfig, belief: &[Laika]) -> f32 {
    let laika = if belief.is_empty() { None } else { Some(&belief[seed as usize % belief.len()]) };
    rollout(ep, policy, first, seed, cfg, laika)
}
