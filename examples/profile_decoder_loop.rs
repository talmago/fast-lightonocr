//! Profile the autoregressive decoder loop with reusable KV cache.
//!
//! ```text
//! ORT_DYLIB_PATH=... LIGHTONOCR_PROFILE=1 LIGHTONOCR_KV_STRATEGY=reusable \
//!   cargo run --release --example profile_decoder_loop --features 'load-dynamic,profiling' -- \
//!   models/lightonocr examples/SROIE-receipt.jpeg 256 greedy
//! ```

use fast_lightonocr::{LightOnOCR, LightOnOCROptions};

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
    let mode = std::env::args()
        .nth(4)
        .unwrap_or_else(|| "default".to_owned());

    unsafe {
        if std::env::var_os("LIGHTONOCR_KV_STRATEGY").is_none() {
            std::env::set_var("LIGHTONOCR_KV_STRATEGY", "reusable");
        }
    }

    let options = LightOnOCROptions::default().with_max_new_tokens(max_new_tokens);
    let mut model = LightOnOCR::from_pretrained(model_dir, options)?;

    match mode.as_str() {
        "greedy" => model.generation_config_mut().do_sample = false,
        "sample" | "default" => {
            // Keep generation_config.json defaults (typically do_sample=true).
        }
        other => {
            eprintln!("Unknown mode {other:?}; use greedy|sample|default");
            std::process::exit(1);
        }
    }

    println!(
        "Profiling decoder loop: max_new_tokens={} do_sample={} kv_strategy={}",
        model.generation_config().max_new_tokens,
        model.generation_config().do_sample,
        std::env::var("LIGHTONOCR_KV_STRATEGY").unwrap_or_default()
    );

    // Warm-up (not profiled as a separate run; still prints if PROFILE is on).
    let _ = model.process_file(&image_path, None)?;

    let result = model.process_file(&image_path, None)?;
    println!(
        "finish={:?} tokens={}",
        result.finish_reason(),
        result.token_ids().len()
    );
    Ok(())
}
