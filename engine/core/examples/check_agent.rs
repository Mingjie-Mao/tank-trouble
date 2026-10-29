//! Grade the deployed controller against native Laika from either seat:
//! `check_agent <policy> <first seed> <rounds> <seat> [search 0/1] [Laika copies tracked]`.
use std::sync::Arc;
use std::time::Instant;
use tank_core::agent::{press, Agent};
use tank_core::game::{Event, Game};
use tank_core::policy::Policy;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let policy = Arc::new(Policy::from_bytes(&std::fs::read(&a[1]).unwrap()).unwrap());
    let (first, rounds, seat): (u64, u64, usize) = (a[2].parse().unwrap(), a[3].parse().unwrap(), a[4].parse().unwrap());
    let search = a.get(5).is_none_or(|v| v == "1");
    let particles: usize = a.get(6).map_or(0, |v| v.parse().unwrap());
    let clock = Instant::now();
    let mut now = || clock.elapsed().as_secs_f64() * 1000.0;
    let (mut wins, mut searched, mut changed, mut frames) = (0, 0, 0, 0);
    let mut worst: f64 = 0.0;
    for seed in first..first + rounds {
        let mut g = Game::with_laika(seed, [seat == 1, seat == 0]);
        g.events.clear();
        let mut agent = Agent::new(policy.clone(), seat, &g, seed + 101);
        agent.budget_ms = f64::INFINITY;
        if !search {
            agent.search = None;
        }
        if particles > 0 {
            agent.track_laika(&g, particles);
        }
        let (mut ended, mut winner) = (false, None);
        let mut t = 0;
        while !ended && t < 750 + 125 {
            let t0 = now();
            let action = agent.decide(&g, &mut now);
            worst = worst.max(now() - t0);
            press(&mut g, seat, action);
            agent.before_step(&g);
            let events = g.step().to_vec();
            agent.after_step(&g, &events);
            frames += 1;
            searched += agent.last.searched as u32;
            changed += agent.last.changed as u32;
            for e in &events {
                if let Event::RoundEnd(w) = e {
                    ended = true;
                    winner = *w;
                }
            }
            t += 1;
            if t >= 750 && g.end_count < 0 {
                break;
            }
        }
        wins += (winner == Some(seat)) as u32;
    }
    println!("seat {seat} search {search} copies {particles}: won {wins}/{rounds} = {:.4}; searched {:.1}% of frames, changed {changed}; worst decision {worst:.1} ms",
             wins as f64 / rounds as f64, 100.0 * searched as f64 / frames as f64);
}
