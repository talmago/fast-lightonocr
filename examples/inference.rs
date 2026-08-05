use std::{fs, path::Path};

use fast_lightonocr::{ExecutionProvider, LightOnOCR, LightOnOCROptions, RuntimeOptions};

const EXAMPLE_IMAGE: &str = "examples/SROIE-receipt.jpeg";
const EXAMPLE_IMAGE_URL: &str = "https://huggingface.co/datasets/hf-internal-testing/fixtures_ocr/resolve/main/SROIE-receipt.jpeg";

fn ensure_example_image() -> Result<(), Box<dyn std::error::Error>> {
    if Path::new(EXAMPLE_IMAGE).exists() {
        return Ok(());
    }

    println!("Downloading example image...");

    let response = reqwest::blocking::get(EXAMPLE_IMAGE_URL)?;
    let bytes = response.bytes()?;

    fs::write(EXAMPLE_IMAGE, bytes)?;

    Ok(())
}

fn main() -> fast_lightonocr::Result<()> {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/lightonocr".to_owned());

    let image_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| EXAMPLE_IMAGE.to_owned());

    let preset = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "default".to_owned());

    let provider = std::env::args().nth(4).unwrap_or_else(|| "cpu".to_owned());

    let device_id = std::env::args()
        .nth(5)
        .map(|value| {
            value.parse::<usize>().unwrap_or_else(|_| {
                eprintln!("Invalid CUDA device_id: {value}");
                std::process::exit(1);
            })
        })
        .unwrap_or(0);

    ensure_example_image().expect("failed to download example image");

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
