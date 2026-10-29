//! Check an exported policy against the PyTorch logits stored in its header.
use tank_core::policy::Policy;

fn main() {
    let path = std::env::args().nth(1).expect("policy file");
    let bytes = std::fs::read(&path).unwrap();
    let policy = Policy::from_bytes(&bytes).unwrap();
    let n = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[4..4 + n]).unwrap();
    let mut worst = 0f32;
    let cases = header["parity"].as_array().unwrap();
    for case in cases {
        let obs: Vec<f32> = case["obs"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let mask: Vec<u8> = case["mask"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u8).collect();
        let want: Vec<f32> = case["logits"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let got = policy.logits(&obs, &mask);
        for (g, w) in got.iter().zip(&want) {
            worst = worst.max((g - w).abs());
        }
        if let (Some(v), Some(want)) = (policy.value(&obs, &mask), case["value"].as_f64()) {
            worst = worst.max((v - want as f32).abs());
        }
    }
    println!("{path}: obs_dim {}, {} cases, max |dlogit| {worst:.2e}", policy.obs_dim, cases.len());
    assert!(worst < 1e-3);
}
