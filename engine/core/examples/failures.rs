//! Replay rounds exactly as `tank_rl.play(mpc=True, particles=16)` plays them
//! and describe each one that is not a win, one JSON line per round:
//! `failures <policy> <seeds file, one per line> [threads]`.
use std::collections::HashMap;
use std::sync::Arc;

use tank_core::belief::Belief;
use tank_core::env::{Episode, OpponentKind, Outcome, Reward};
use tank_core::game::Event;
use tank_core::obs::{self, BULLET_SLOTS, DODGE_OFFSET, OBS_DIM};
use tank_core::policy::Policy;
use tank_core::rng::Rng;
use tank_core::teacher::{self, TeacherConfig};

struct Frame {
    searched: bool,
    changed: bool,
    safe_moves: usize,
    chosen_safe: bool,
    los: bool,
}

fn line_of_sight(g: &tank_core::game::Game) -> bool {
    let (a, b) = (&g.tanks[0], &g.tanks[1]);
    let (dx, dy) = (b.x - a.x, b.y - a.y);
    let steps = ((dx.hypot(dy)) / (g.scale() / 8.0)).ceil().max(1.0) as i32;
    (1..steps).all(|k| {
        let t = k as f64 / steps as f64;
        !g.wall_hit(a.x + dx * t, a.y + dy * t)
    })
}

fn replay(policy: &Policy, seed: u64) -> Option<String> {
    let cfg = TeacherConfig { candidates: 2, rollouts: 2, horizon: 24, ..TeacherConfig::default() };
    let mut ep = Episode::new(seed, OpponentKind::Laika, Reward::default());
    let mut rng = Rng::new(seed + 101);
    let mut belief = Belief::new(&ep.game, 1, 16, seed + 202);
    let (mut o, mut m) = (vec![0f32; OBS_DIM], vec![0u8; BULLET_SLOTS]);
    let mut frames: Vec<Frame> = Vec::new();
    let mut born: HashMap<u32, (u32, usize)> = HashMap::new(); // id -> (frame fired, owner)
    let mut deaths: Vec<String> = Vec::new();
    let mut shots = [0u32; 2];
    let mut my_death_frame: Option<u32> = None;
    while ep.outcome == Outcome::Running {
        let mut action = 0;
        let g = &ep.game;
        let mut f = Frame { searched: false, changed: false, safe_moves: 9, chosen_safe: true, los: g.tanks[1].alive && line_of_sight(g) };
        if !g.frozen {
            ep.observe(0, &mut o, &mut m);
            let probs = policy.shielded_probs(&o, &m);
            action = teacher::sample(&probs, &mut rng);
            if teacher::danger(&ep.game, policy, &o) {
                let s = teacher::scores(&ep, policy, &probs, action, &cfg, &mut rng, belief.particles());
                let best = (0..18).filter(|&a| !s[a].is_nan()).max_by(|&a, &b| s[a].total_cmp(&s[b])).unwrap();
                f.searched = true;
                if s[best] > s[action] {
                    action = best;
                    f.changed = true;
                }
            }
            let dodge = &o[DODGE_OFFSET..DODGE_OFFSET + 9];
            f.safe_moves = dodge.iter().filter(|&&d| d >= 0.0).count();
            f.chosen_safe = dodge[action / 2] >= 0.0;
        }
        let before: Vec<(u32, usize, f64, f64, bool)> =
            ep.game.bullets.iter().filter(|b| !b.removed).map(|b| (b.id, b.owner, b.x, b.y, b.has_bounced)).collect();
        frames.push(f);
        let before_step = ep.game.clone();
        ep.step(action, None);
        belief.update(&before_step, action, &ep.game, &ep.game.events);
        let now = ep.frames;
        for e in &ep.game.events {
            match e {
                Event::Fire(t) => shots[*t] += 1,
                Event::Hit { owner, victim } => {
                    let v = &ep.game.tanks[*victim];
                    let alive: Vec<u32> = ep.game.bullets.iter().filter(|b| !b.removed).map(|b| b.id).collect();
                    let killer = before.iter().filter(|b| b.1 == *owner && !alive.contains(&b.0))
                        .min_by(|a, b| (a.2 - v.x).hypot(a.3 - v.y).total_cmp(&(b.2 - v.x).hypot(b.3 - v.y)));
                    let (age, bounced) = match killer {
                        Some(k) => (born.get(&k.0).map_or(-1, |&(f0, _)| now as i64 - f0 as i64), k.4),
                        None => (-1, false),
                    };
                    if *victim == 0 && my_death_frame.is_none() {
                        my_death_frame = Some(now);
                    }
                    deaths.push(format!("{{\"frame\":{now},\"owner\":{owner},\"victim\":{victim},\"age\":{age},\"bounced\":{bounced}}}"));
                }
                _ => {}
            }
        }
        for b in ep.game.bullets.iter().filter(|b| !b.removed) {
            born.entry(b.id).or_insert((now, b.owner));
        }
    }
    if ep.outcome == Outcome::Win {
        return None;
    }
    // The last second before my death (or the round's end): frame k's decision is frames[k - 1].
    let my_death = deaths.iter().position(|d| d.contains("\"victim\":0"));
    let end = frames.len();
    let cut = my_death_frame.map_or(end, |f| f as usize);
    let tail = &frames[cut.saturating_sub(25)..cut];
    let long_tail = &frames[cut.saturating_sub(40)..cut];
    let outcome = format!("{:?}", ep.outcome);
    let los_share = frames.iter().filter(|f| f.los).count() as f64 / end.max(1) as f64;
    let g = &ep.game;
    let path = g.round.cell_dist((g.tanks[0].x / g.scale()) as i32, (g.tanks[0].y / g.scale()) as i32,
                                 (g.tanks[1].x / g.scale()) as i32, (g.tanks[1].y / g.scale()) as i32);
    Some(format!(
        "{{\"seed\":{seed},\"outcome\":\"{outcome}\",\"frames\":{},\"deaths\":[{}],\"my_death_first\":{},\
\"tail_searched\":{},\"tail_changed\":{},\"tail_min_safe\":{},\"tail_unsafe_choice\":{},\"safe_tail\":[{}],\"search_tail\":[{}],\"shots\":[{},{}],\"los_share\":{los_share:.3},\"end_path\":{}}}",
        ep.frames, deaths.join(","), my_death == Some(0),
        tail.iter().filter(|f| f.searched).count(), tail.iter().filter(|f| f.changed).count(),
        tail.iter().map(|f| f.safe_moves).min().unwrap_or(9), tail.iter().filter(|f| !f.chosen_safe).count(),
        long_tail.iter().map(|f| f.safe_moves.to_string()).collect::<Vec<_>>().join(","),
        long_tail.iter().map(|f| (f.searched as u8 + f.changed as u8).to_string()).collect::<Vec<_>>().join(","),
        shots[0], shots[1], if path.is_finite() { path } else { -1.0 }))
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let policy = Arc::new(Policy::from_bytes(&std::fs::read(&a[1]).unwrap()).unwrap());
    assert_eq!(policy.obs_dim, obs::OBS_DIM);
    let seeds: Vec<u64> = std::fs::read_to_string(&a[2]).unwrap().lines().filter_map(|l| l.trim().parse().ok()).collect();
    let threads: usize = a.get(3).map_or(9, |t| t.parse().unwrap());
    let chunks: Vec<Vec<u64>> = (0..threads).map(|i| seeds.iter().skip(i).step_by(threads).copied().collect()).collect();
    let handles: Vec<_> = chunks.into_iter().map(|chunk| {
        let policy = policy.clone();
        std::thread::spawn(move || chunk.iter().map(|&s| (s, replay(&policy, s))).collect::<Vec<_>>())
    }).collect();
    let mut all: Vec<(u64, Option<String>)> = handles.into_iter().flat_map(|h| h.join().unwrap()).collect();
    all.sort_by_key(|r| r.0);
    for (seed, line) in all {
        println!("{}", line.unwrap_or_else(|| format!("{{\"seed\":{seed},\"outcome\":\"Win\"}}")));
    }
}
