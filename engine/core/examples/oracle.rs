//! Diagnostic only: how much does the real-time search lose by not knowing
//! Laika? Plays rounds exactly as `play(mpc=True)`, but the rollouts' Laika is
//! rebuilt (fair), a clone of the real one with a fresh random stream (state),
//! or a clone with the real random stream as well (full).
//! `oracle <policy> <first seed> <rounds> <fair|state|full> <candidates> <rollouts> [threads]`
//! prints the outcome counts, then one outcome code per round (1 win, 2 loss, 3 double KO, 4 draw).
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tank_core::env::{Episode, OpponentKind, Outcome, Reward};
use tank_core::laika::Laika;
use tank_core::policy::Policy;
use tank_core::rng::Rng;
use tank_core::teacher::{self, TeacherConfig};

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Fair,
    State,
    Full,
}

fn rollout(ep: &Episode, policy: &Policy, first: usize, seed: u64, cfg: &TeacherConfig, mode: Mode) -> f32 {
    let mut r = ep.clone();
    let g = &mut r.game;
    if mode != Mode::Full {
        g.rng = Rng::new(seed);
    }
    g.laika_on_tank1 = false;
    g.events.clear();
    g.hit_tests.clear();
    if mode == Mode::Fair {
        g.ai[1] = Some(Box::new(Laika::new(g.round.scale)));
    }
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
            action = teacher::sample(&policy.shielded_probs(&obs, &mask), &mut rng);
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

fn play(policy: &Policy, seed: u64, cfg: &TeacherConfig, mode: Mode) -> u8 {
    let mut ep = Episode::new(seed, OpponentKind::Laika, Reward::default());
    let mut rng = Rng::new(seed + 101);
    let (mut obs, mut mask) = (vec![0f32; policy.obs_dim], vec![0u8; policy.layout[3]]);
    while ep.outcome == Outcome::Running {
        let mut action = 0;
        if !ep.game.frozen {
            ep.observe(0, &mut obs, &mut mask);
            let probs = policy.shielded_probs(&obs, &mask);
            action = teacher::sample(&probs, &mut rng);
            if teacher::danger(&ep.game, policy, &obs) {
                let mut ranked: Vec<usize> = (0..18).filter(|&a| probs[a] > 0.0).collect();
                ranked.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
                ranked.truncate(cfg.candidates);
                ranked.retain(|&a| a != action);
                ranked.insert(0, action);
                let seeds: Vec<u64> = (0..cfg.rollouts).map(|_| rng.randrange(1 << 30) as u64).collect();
                let mut out = [f32::NAN; 18];
                for &a in &ranked {
                    out[a] = seeds.iter().map(|&s| rollout(&ep, policy, a, s, cfg, mode)).sum::<f32>() / cfg.rollouts as f32;
                }
                let best = (0..18).filter(|&a| !out[a].is_nan()).max_by(|&a, &b| out[a].total_cmp(&out[b])).unwrap();
                if out[best] > out[action] {
                    action = best;
                }
            }
        }
        ep.step(action, None);
    }
    match ep.outcome {
        Outcome::Win => 1,
        Outcome::Loss => 2,
        Outcome::DoubleKo => 3,
        _ => 4,
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let policy = Arc::new(Policy::from_bytes(&std::fs::read(&a[1]).unwrap()).unwrap());
    let first: u64 = a[2].parse().unwrap();
    let rounds: usize = a[3].parse().unwrap();
    let mode = match a[4].as_str() {
        "fair" => Mode::Fair,
        "state" => Mode::State,
        "full" => Mode::Full,
        m => panic!("unknown mode {m}"),
    };
    let cfg = TeacherConfig { candidates: a[5].parse().unwrap(), rollouts: a[6].parse().unwrap(), horizon: 24, ..TeacherConfig::default() };
    let threads: usize = a.get(7).map_or(9, |t| t.parse().unwrap());
    let next = Arc::new(AtomicUsize::new(0));
    let handles: Vec<_> = (0..threads).map(|_| {
        let (policy, next) = (policy.clone(), next.clone());
        std::thread::spawn(move || {
            let mut out = Vec::new();
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= rounds {
                    break out;
                }
                out.push((i, play(&policy, first + i as u64, &cfg, mode)));
            }
        })
    }).collect();
    let mut all: Vec<(usize, u8)> = handles.into_iter().flat_map(|h| h.join().unwrap()).collect();
    all.sort();
    let count = |c: u8| all.iter().filter(|r| r.1 == c).count();
    println!("{} win {} loss {} double {} draw {} -> {:.4}", a[4], count(1), count(2), count(3), count(4), count(1) as f64 / rounds as f64);
    println!("{}", all.iter().map(|r| r.1.to_string()).collect::<String>());
}
