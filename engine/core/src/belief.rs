//! Laika's hidden state — its goal, queued actions and timers — inferred from
//! what its tank visibly does. A particle filter of Laika copies: they start
//! the round as the real one does (they share its program, not its memory),
//! replay every step on a copy of the game with a random stream of their own,
//! and the ones whose tank moved and fired closest to the real one survive.

use crate::agent::press;
use crate::game::{Event, Game};
use crate::laika::Laika;
use crate::rng::Rng;

pub struct Belief {
    /// Laika's tank in the real game.
    pub laika: usize,
    particles: Vec<Laika>,
    rng: Rng,
}

fn fired(events: &[Event], tank: usize) -> bool {
    events.iter().any(|e| matches!(e, Event::Fire(t) if *t == tank))
}

impl Belief {
    pub fn new(g: &Game, laika: usize, particles: usize, seed: u64) -> Belief {
        Belief { laika, particles: vec![Laika::new(g.scale()); particles], rng: Rng::new(seed) }
    }

    /// The surviving copies, in the real game's seats.
    pub fn particles(&self) -> &[Laika] {
        &self.particles
    }

    /// The copies for a game seen from `seat` (tank 0 = `seat`).
    pub fn seen_from(&self, seat: usize) -> Vec<Laika> {
        self.particles.iter().map(|l| if seat == 1 { l.mirrored() } else { l.clone() }).collect()
    }

    /// One step: `before` is the game it started from, `action` what the
    /// other tank pressed, `after` and `events` what came of it.
    pub fn update(&mut self, before: &Game, action: usize, after: &Game, events: &[Event]) {
        let l = self.laika;
        if events.iter().any(|e| matches!(e, Event::NewRound(_))) {
            self.particles = vec![Laika::new(after.scale()); self.particles.len()];
            return;
        }
        let real = &after.tanks[l];
        let real_fired = fired(events, l);
        let replayed: Vec<(f64, Laika)> = self.particles.iter().map(|p| {
            let mut g = before.clone();
            g.rng = Rng::new(self.rng.randrange(1 << 30) as u64);
            g.laika_on_tank0 = false;
            g.laika_on_tank1 = false;
            g.ai = vec![None, None];
            g.ai[l] = Some(Box::new(p.clone()));
            press(&mut g, 1 - l, action);
            g.step();
            let t = &g.tanks[l];
            let mut err = (t.x - real.x).abs() + (t.y - real.y).abs() + (t.rotation - real.rotation).abs();
            if fired(&g.events, l) != real_fired {
                err += 1e6;
            }
            (err, *g.ai[l].take().unwrap())
        }).collect();
        let best = replayed.iter().map(|r| r.0).fold(f64::INFINITY, f64::min);
        let survivors: Vec<Laika> = replayed.into_iter().filter(|r| r.0 <= best).map(|r| r.1).collect();
        self.particles = (0..self.particles.len()).map(|i| survivors[i % survivors.len()].clone()).collect();
    }
}
