//! Replays the Python port's recorded trajectories (tools/record_engine_parity.py)
//! and requires the same discrete state on every frame and every float within
//! 1e-6 (Python's libm and Rust's may differ in the last bit of a sine).

use std::io::Read;

use tank_core::game::Game;

/// The recorder's xorshift32 input stream for tank 0.
struct Driver {
    s: u32,
    hold: i32,
    buttons: [bool; 5],
}

impl Driver {
    fn new(seed: u64) -> Driver {
        let s = ((seed as u128 * 2_654_435_761u128 + 1) & 0xffff_ffff) as u32;
        Driver { s: if s == 0 { 1 } else { s }, hold: 0, buttons: [false; 5] }
    }

    fn next32(&mut self) -> u32 {
        let mut s = self.s;
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        self.s = s;
        s
    }

    fn step(&mut self) -> [bool; 5] {
        if self.hold <= 0 {
            let r = self.next32();
            let throttle = r % 3;
            let turn = (r / 3) % 3;
            self.buttons = [throttle == 2, throttle == 0, turn == 0, turn == 2, (r / 9) % 6 == 0];
            self.hold = 3 + ((r / 54) % 8) as i32;
        }
        self.hold -= 1;
        self.buttons
    }
}

fn bit(v: bool) -> &'static str {
    if v { "1" } else { "0" }
}

fn snapshot(g: &Game) -> (String, Vec<f64>) {
    let scores: Vec<String> = g.scores.iter().map(|s| s.to_string()).collect();
    let mut parts = vec![
        g.frame.to_string(),
        g.round_number.to_string(),
        g.alive_count.to_string(),
        g.end_count.to_string(),
        g.reset_count.to_string(),
        bit(g.frozen).to_string(),
        format!("{:.9}", g.crate_timer),
        scores.join(","),
    ];
    let mut floats = Vec::new();
    for t in &g.tanks {
        parts.push(format!(
            "T{},{},{},{},{},{},{},{},{}",
            bit(t.alive), t.bullets_fired, bit(t.trigger_released), bit(t.hit_something),
            bit(t.forward), bit(t.backup), bit(t.turn_left), bit(t.turn_right), bit(t.fire)
        ));
        floats.extend([t.x, t.y, t.rotation]);
    }
    for b in &g.bullets {
        parts.push(format!("Bbullet{},{},{},{}", b.id, b.owner, b.lifetime, bit(b.has_bounced)));
        floats.extend([b.x, b.y, b.x_speed, b.y_speed]);
    }
    for e in &g.events {
        parts.push(format!("E{e}"));
    }
    let mut h = sha1_smol::Sha1::new();
    h.update(parts.join("|").as_bytes());
    (h.digest().to_string()[..16].to_string(), floats)
}

#[test]
fn matches_the_python_port_frame_by_frame() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/engine-parity.json.gz");
    let mut text = String::new();
    flate2::read::GzDecoder::new(std::fs::File::open(path).unwrap()).read_to_string(&mut text).unwrap();
    let fixture: serde_json::Value = serde_json::from_str(&text).unwrap();
    let frames = fixture["frames"].as_u64().unwrap();
    let mut checked = 0;
    for case in fixture["cases"].as_array().unwrap() {
        let seed = case["seed"].as_u64().unwrap();
        let expected = case["frames"].as_array().unwrap();
        let mut g = Game::new(seed, true);
        g.events.clear();
        let mut drive = Driver::new(seed);
        for frame in 0..=frames as usize {
            if frame > 0 {
                if !g.frozen {
                    let [f, b, l, r, x] = drive.step();
                    g.tanks[0].set_buttons(f, b, l, r, x);
                }
                g.step();
            }
            let (digest, floats) = snapshot(&g);
            let want = &expected[frame];
            assert_eq!(digest, want[0].as_str().unwrap(), "seed {seed}: discrete state differs at frame {frame}");
            let want_floats = want[1].as_array().unwrap();
            assert_eq!(floats.len(), want_floats.len(), "seed {seed} frame {frame}");
            for (i, (got, w)) in floats.iter().zip(want_floats).enumerate() {
                let w = w.as_f64().unwrap();
                assert!((got - w).abs() <= 1e-6, "seed {seed} frame {frame} value {i}: {got} vs {w}");
            }
            checked += 1;
        }
        assert_eq!(g.round_number as u64, case["rounds"].as_u64().unwrap(), "seed {seed}");
    }
    assert_eq!(checked, 12 * (frames as usize + 1));
}
