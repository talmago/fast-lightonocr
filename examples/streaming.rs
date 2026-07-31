use std::io::{self, Write};
use std::{fs, path::Path};

use fast_lightonocr::{LightOnOCR, LightOnOCROptions};

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

    ensure_example_image().expect("failed to download example image");

    let options = match preset.as_str() {
        "q4" => LightOnOCROptions::q4(),
        "fp16" => LightOnOCROptions::fp16(),
        "default" => LightOnOCROptions::default(),
        other => {
            eprintln!("Unknown model preset: {other}");
            std::process::exit(1);
        }
    };

    println!("Loading model from {model_dir} (preset: {preset})...");
    let mut model = LightOnOCR::from_pretrained(model_dir, options)?;

    println!("Processing {image_path}...");
    println!("\n=== OCR Result (streaming) ===\n");

    let result = model.process_file_streaming(image_path, None, |chunk| {
        print!("{chunk}");
        io::stdout().flush().expect("failed to flush stdout");
    })?;

    println!("\n\n=== Generation ===");
    println!("finish reason: {:?}", result.finish_reason());
    println!("generated tokens: {}", result.token_ids().len());

    Ok(())
}
