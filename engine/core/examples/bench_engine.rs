//! Raw engine throughput: Laika on tank 1, a fixed pseudo-random input stream on tank 0.
use std::time::Instant;
use tank_core::game::Game;

fn main() {
    let frames = 200_000u64;
    let mut g = Game::new(1, true);
    let mut s: u32 = 12345;
    let start = Instant::now();
    for f in 0..frames {
        if f % 7 == 0 {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
        }
        if !g.frozen {
            g.tanks[0].set_buttons(s % 3 == 2, s % 3 == 0, (s / 3) % 3 == 0, (s / 3) % 3 == 2, (s / 9) % 6 == 0);
        }
        g.step();
    }
    let secs = start.elapsed().as_secs_f64();
    println!("rust: {:.0} frames/s ({} rounds)", frames as f64 / secs, g.round_number);
}
