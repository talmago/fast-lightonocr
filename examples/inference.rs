use std::path::Path;

use fast_lightonocr::{ExecutionProvider, LightOnOCR, LightOnOCROptions, RuntimeOptions};

const DEFAULT_MODEL_DIR: &str = "models/LightOnOCR-2-1B-ONNX";
const EXAMPLE_IMAGE: &str = "tests/fixtures/SROIE.png";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model_dir = DEFAULT_MODEL_DIR.to_owned();
    let mut image_path = EXAMPLE_IMAGE.to_owned();
    let mut preset = "default".to_owned();
    let mut provider = "cpu".to_owned();
    let mut device_id = 0usize;
    let mut positional = Vec::new();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        let value = |args: &mut std::iter::Skip<std::env::Args>, name: &str| {
            args.next().unwrap_or_else(|| {
                eprintln!("Missing value for {name}");
                std::process::exit(2);
            })
        };
        match arg.as_str() {
            "--model-dir" => model_dir = value(&mut args, "--model-dir"),
            "--image" => image_path = value(&mut args, "--image"),
            "--preset" => preset = value(&mut args, "--preset"),
            "--provider" => provider = value(&mut args, "--provider"),
            "--device-id" => {
                let raw = value(&mut args, "--device-id");
                device_id = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("Invalid CUDA device_id: {raw}");
                    std::process::exit(2);
                });
            }
            "--help" | "-h" => {
                println!(
                    "Usage: inference [--model-dir PATH] [--image PATH] [--preset default|fp16|q4] [--provider cpu|cuda] [--device-id N]"
                );
                println!("Defaults: model-dir={DEFAULT_MODEL_DIR}, image={EXAMPLE_IMAGE}");
                return Ok(());
            }
            _ if arg.starts_with('-') => {
                eprintln!("Unknown option: {arg}");
                std::process::exit(2);
            }
            _ => positional.push(arg),
        }
    }

    // Keep the original positional form working for existing example commands.
    if let Some(value) = positional.first() {
        model_dir = value.clone();
    }
    if let Some(value) = positional.get(1) {
        image_path = value.clone();
    }
    if let Some(value) = positional.get(2) {
        preset = value.clone();
    }
    if let Some(value) = positional.get(3) {
        provider = value.clone();
    }
    if let Some(value) = positional.get(4) {
        device_id = value.parse::<usize>().unwrap_or_else(|_| {
            eprintln!("Invalid CUDA device_id: {value}");
            std::process::exit(2);
        });
    }

    if !Path::new(&image_path).is_file() {
        return Err(format!("image file does not exist: {image_path}").into());
    }

    let mut options = match preset.as_str() {
        "q4" => LightOnOCROptions::q4(),
        "fp16" => LightOnOCROptions::fp16(),
        "default" => LightOnOCROptions::default(),
        other => {
            eprintln!("Unknown model preset: {other}");
            std::process::exit(1);
        }
    };

    let execution_provider = match provider.as_str() {
        "cpu" => ExecutionProvider::Cpu,
        "cuda" => ExecutionProvider::Cuda { device_id },
        other => {
            eprintln!("Unknown execution provider: {other} (expected cpu or cuda)");
            std::process::exit(1);
        }
    };
    options =
        options.with_runtime(RuntimeOptions::default().with_execution_provider(execution_provider));

    println!("Loading model from {model_dir} (preset: {preset}, provider: {provider})...");
    let mut model = LightOnOCR::from_pretrained(model_dir, options)?;

    println!("Processing {image_path}...");
    let result = model.process_file(image_path, None)?;

    println!("\n=== OCR Result ===\n");
    println!("{}", result.text());

    Ok(())
}
