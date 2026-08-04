//! End-to-end A/B for experimental KV cache update strategies.
//!
//! Usage:
//! ```text
//! ORT_DYLIB_PATH=... cargo run --release --example kv_cache_e2e \
//!   --features 'load-dynamic,profiling' -- \
//!   models/lightonocr examples/SROIE-receipt.jpeg 256
//! ```

use std::time::Instant;

use fast_lightonocr::{LightOnOCR, LightOnOCROptions};

fn set_strategy(strategy: &str) {
    unsafe {
        std::env::set_var("LIGHTONOCR_KV_STRATEGY", strategy);
    }
}

fn main() -> fast_lightonocr::Result<()> {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/lightonocr".to_owned());
    let image_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "examples/SROIE-receipt.jpeg".to_owned());
    let max_new_tokens = std::env::args()
        .nth(3)
        .and_then(|v| v.parse().ok())
        .unwrap_or(256usize);

    let strategies = ["full_copy", "reusable", "delta"];

    println!("KV cache E2E A/B");
    println!("  model={model_dir}");
    println!("  image={image_path}");
    println!("  max_new_tokens={max_new_tokens}");
    println!("  decoding=greedy");
    println!();

    let options = LightOnOCROptions::default().with_max_new_tokens(max_new_tokens);
    let mut model = LightOnOCR::from_pretrained(&model_dir, options)?;
    model.generation_config_mut().do_sample = false;

    // Warm-up.
    set_strategy("full_copy");
    let _ = model.process_file(&image_path, None)?;

    let mut baseline_tokens: Option<Vec<i64>> = None;
    let mut baseline_ms = 0.0;

    for strategy in strategies {
        set_strategy(strategy);
        let _ = model.process_file(&image_path, None)?;

        let mut samples = Vec::new();
        let mut tokens = Vec::new();
        for _ in 0..3 {
            let started = Instant::now();
            let result = model.process_file(&image_path, None)?;
            samples.push(started.elapsed().as_secs_f64() * 1000.0);
            tokens = result.token_ids().to_vec();
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = samples[samples.len() / 2];

        let parity = match &baseline_tokens {
            None => {
                baseline_tokens = Some(tokens.clone());
                baseline_ms = median;
                "baseline".to_owned()
            }
            Some(base) => {
                if base == &tokens {
                    "tokens_match".to_owned()
                } else {
                    let first_diff = base
                        .iter()
                        .zip(tokens.iter())
                        .position(|(a, b)| a != b)
                        .unwrap_or(base.len().min(tokens.len()));
                    format!(
                        "TOKEN_MISMATCH (base_len={}, got_len={}, first_diff={first_diff})",
                        base.len(),
                        tokens.len()
                    )
                }
            }
        };

        let speedup = if strategy == "full_copy" {
            1.0
        } else {
            baseline_ms / median
        };

        println!(
            "{strategy:<12} median={median:>10.1} ms  tokens={}  {parity}  vs_full_copy={speedup:.3}x",
            tokens.len()
        );
    }

    unsafe {
        std::env::remove_var("LIGHTONOCR_KV_STRATEGY");
    }

    Ok(())
}
