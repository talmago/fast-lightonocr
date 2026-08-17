//! End-to-end latency / tokens-per-second bench across model presets.
//!
//! Reports wall-clock `process_file` time (E2E-normalized tok/s, not pure decode).
//!
//! ```bash
//! cargo run --release --features load-dynamic --example inference_bench -- \
//!   models/lightonocr examples/SROIE-receipt.jpeg q4,default,fp16
//! ```
//!
//! For CUDA (device KV by default), rebuild with `--features load-dynamic,cuda`
//! and pass a CUDA provider through your usual runtime options / example args.
//! Compare against `FAST_LIGHTONOCR_CUDA_HOST_KV=1` to measure host-KV overhead.

use std::cmp::Ordering;
use std::path::Path;
use std::time::Instant;

use fast_lightonocr::{LightOnOCR, LightOnOCROptions};

const DEFAULT_MODEL_DIR: &str = "models/lightonocr";
const DEFAULT_IMAGE: &str = "examples/SROIE-receipt.jpeg";
const WARMUP_RUNS: usize = 1;
const TIMED_RUNS: usize = 3;
const TOKEN_LIMITS: [usize; 2] = [64, 256];

fn main() -> fast_lightonocr::Result<()> {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_MODEL_DIR.to_owned());
    let image_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| DEFAULT_IMAGE.to_owned());
    let presets_arg = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "q4,default,fp16".to_owned());
    let presets: Vec<&str> = presets_arg
        .split(',')
        .map(str::trim)
        .filter(|preset| !preset.is_empty())
        .collect();

    if !Path::new(&image_path).is_file() {
        eprintln!("image not found: {image_path}");
        std::process::exit(1);
    }

    let host_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    println!("model={model_dir}");
    println!("image={image_path}");
    println!("presets={presets_arg}");
    println!("host_parallelism={host_threads}");
    println!("warmup={WARMUP_RUNS} timed_runs={TIMED_RUNS}");
    println!("tok_s is E2E-normalized (tokens / process_file seconds)");
    if std::env::var_os("ORT_DYLIB_PATH").is_some() {
        println!("ORT_DYLIB_PATH is set");
    }
    if let Ok(omp) = std::env::var("OMP_NUM_THREADS") {
        println!("OMP_NUM_THREADS={omp}");
    }
    println!();
    println!(
        "{:<8} {:<8} {:>5} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "preset", "mode", "max", "load_ms", "mean_ms", "median_ms", "tokens", "tok_s"
    );

    for preset in presets {
        let Some(base) = options_for_preset(preset) else {
            println!("{preset:<8} skipped (unknown preset)");
            continue;
        };

        if !preset_assets_exist(&model_dir, &base) {
            println!("{preset:<8} skipped (onnx assets missing under {model_dir})");
            continue;
        }

        for max_new_tokens in TOKEN_LIMITS {
            for (mode, do_sample) in [("greedy", false), ("sample", true)] {
                bench_row(
                    &model_dir,
                    &image_path,
                    base.clone().with_max_new_tokens(max_new_tokens),
                    preset,
                    mode,
                    max_new_tokens,
                    do_sample,
                )?;
            }
        }
    }

    Ok(())
}

fn options_for_preset(preset: &str) -> Option<LightOnOCROptions> {
    Some(match preset {
        "q4" => LightOnOCROptions::q4(),
        "fp16" => LightOnOCROptions::fp16(),
        "default" => LightOnOCROptions::default(),
        _ => return None,
    })
}

fn preset_assets_exist(model_dir: &str, options: &LightOnOCROptions) -> bool {
    let root = Path::new(model_dir);
    root.join(&options.vision_encoder).is_file()
        && root.join(&options.embedding).is_file()
        && root.join(&options.decoder).is_file()
}

fn bench_row(
    model_dir: &str,
    image_path: &str,
    options: LightOnOCROptions,
    preset: &str,
    mode: &str,
    max_new_tokens: usize,
    do_sample: bool,
) -> fast_lightonocr::Result<()> {
    let load_started = Instant::now();
    let mut model = LightOnOCR::from_pretrained(model_dir, options)?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;

    model.generation_config_mut().do_sample = do_sample;
    model.generation_config_mut().max_new_tokens = max_new_tokens;

    for _ in 0..WARMUP_RUNS {
        let _ = model.process_file(image_path, None)?;
    }

    let mut times_ms = Vec::with_capacity(TIMED_RUNS);
    let mut tokens = 0usize;
    for _ in 0..TIMED_RUNS {
        let started = Instant::now();
        let result = model.process_file(image_path, None)?;
        times_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        tokens = result.token_ids().len();
    }

    times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let mean = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    let median = times_ms[times_ms.len() / 2];
    let tok_s = if mean > 0.0 {
        tokens as f64 / (mean / 1000.0)
    } else {
        0.0
    };

    println!(
        "{preset:<8} {mode:<8} {max_new_tokens:>5} {load_ms:>10.0} {mean:>10.0} {median:>10.0} {tokens:>8} {tok_s:>8.2}"
    );

    Ok(())
}
