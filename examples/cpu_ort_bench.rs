//! Quick CPU ORT thread-count timing for LightOnOCR.
//!
//! Compares default `RuntimeOptions` against explicit `intra_threads` settings
//! on a short generation (64 new tokens). Useful when validating OpenMP vs
//! ORT intra-op thread behavior on a given machine.
//!
//! ```bash
//! cargo run --release --features load-dynamic --example cpu_ort_bench -- models/lightonocr q4
//! ```

use std::time::Instant;

use fast_lightonocr::{LightOnOCR, LightOnOCROptions, RuntimeOptions};

const EXAMPLE_IMAGE: &str = "examples/SROIE-receipt.jpeg";
const MAX_NEW_TOKENS: usize = 64;
const WARMUP_RUNS: usize = 1;
const TIMED_RUNS: usize = 3;

fn main() -> fast_lightonocr::Result<()> {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/lightonocr".to_owned());
    let preset = std::env::args().nth(2).unwrap_or_else(|| "q4".to_owned());

    let host_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    println!("model={model_dir} preset={preset} max_new_tokens={MAX_NEW_TOKENS}");
    println!("host_parallelism={host_threads}");
    println!();

    let configs = [
        ("default", RuntimeOptions::default()),
        ("intra=1", RuntimeOptions::default().with_intra_threads(1)),
        (
            "intra=host",
            RuntimeOptions::default().with_intra_threads(host_threads),
        ),
    ];

    for (label, runtime) in configs {
        let options = options_for_preset(&preset)
            .with_max_new_tokens(MAX_NEW_TOKENS)
            .with_runtime(runtime);
        bench(&model_dir, options, label)?;
    }

    println!();
    println!(
        "Note: Microsoft prebuilt ORT often ignores intra_threads (OpenMP). \
         Try OMP_NUM_THREADS if timings look identical."
    );

    Ok(())
}

fn options_for_preset(preset: &str) -> LightOnOCROptions {
    match preset {
        "q4" => LightOnOCROptions::q4(),
        "fp16" => LightOnOCROptions::fp16(),
        "default" => LightOnOCROptions::default(),
        other => {
            eprintln!("Unknown model preset: {other}");
            std::process::exit(1);
        }
    }
}

fn bench(model_dir: &str, options: LightOnOCROptions, label: &str) -> fast_lightonocr::Result<()> {
    let load_started = Instant::now();
    let mut model = LightOnOCR::from_pretrained(model_dir, options)?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;

    for _ in 0..WARMUP_RUNS {
        let _ = model.process_file(EXAMPLE_IMAGE, None)?;
    }

    let mut times_ms = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        let started = Instant::now();
        let _ = model.process_file(EXAMPLE_IMAGE, None)?;
        times_ms.push(started.elapsed().as_secs_f64() * 1000.0);
    }

    let mean = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    println!("{label:12} load={load_ms:8.0}ms  mean={mean:8.0}ms  runs={times_ms:?}");

    Ok(())
}
