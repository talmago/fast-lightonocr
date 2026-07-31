use std::path::{Path, PathBuf};
use std::process::Command;

use fast_lightonocr::Error;
use fast_lightonocr::model::ImageTensor;
use fast_lightonocr::processor::{DataFormat, ImageProcessor, ProcessorConfig, ResampleFilter};
use image::{DynamicImage, ImageBuffer, Rgb, RgbImage};

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn sample_rgb_image() -> DynamicImage {
    let image: RgbImage = ImageBuffer::from_fn(2, 2, |x, y| match (x, y) {
        (0, 0) => Rgb([255, 0, 0]),
        (1, 0) => Rgb([0, 255, 0]),
        (0, 1) => Rgb([0, 0, 255]),
        (1, 1) => Rgb([255, 255, 255]),
        _ => unreachable!(),
    });
    DynamicImage::ImageRgb8(image)
}

fn image_processor_from_file(path: impl AsRef<Path>) -> fast_lightonocr::Result<ImageProcessor> {
    let config = ProcessorConfig::from_file(path)?;
    ImageProcessor::new(config.image_processor().clone())
}

#[test]
fn loads_processor_config_from_model_directory() {
    let config =
        ProcessorConfig::from_file(fixture_path("processor_small").join("processor_config.json"))
            .unwrap();

    assert_eq!(
        config.image_processor().data_format,
        DataFormat::ChannelsFirst
    );
    assert_eq!(config.image_processor().resample, ResampleFilter::Bicubic);
    assert_eq!(config.image_processor().patch_size, 2);
}

#[test]
fn converts_rgb_image_to_nchw_image_tensor() {
    let processor =
        image_processor_from_file(fixture_path("processor_small").join("processor_config.json"))
            .unwrap();
    let image_tensor = processor.process_image(&sample_rgb_image()).unwrap();

    assert_eq!(image_tensor.shape(), (1, 3, 2, 2));
    assert_eq!(image_tensor.get(0, 0, 0, 0), Some(1.0));
    assert_eq!(image_tensor.get(0, 1, 0, 0), Some(0.0));
    assert_eq!(image_tensor.get(0, 2, 0, 0), Some(0.0));
    assert_eq!(image_tensor.get(0, 0, 0, 1), Some(0.0));
    assert_eq!(image_tensor.get(0, 1, 0, 1), Some(1.0));
    assert_eq!(image_tensor.get(0, 2, 0, 1), Some(0.0));
    assert_eq!(image_tensor.get(0, 0, 1, 0), Some(0.0));
    assert_eq!(image_tensor.get(0, 1, 1, 0), Some(0.0));
    assert_eq!(image_tensor.get(0, 2, 1, 0), Some(1.0));
    assert_eq!(image_tensor.get(0, 0, 1, 1), Some(1.0));
    assert_eq!(image_tensor.get(0, 1, 1, 1), Some(1.0));
    assert_eq!(image_tensor.get(0, 2, 1, 1), Some(1.0));
}

#[test]
fn pads_batches_after_normalization() {
    let processor =
        image_processor_from_file(fixture_path("processor_small").join("processor_config.json"))
            .unwrap();
    let first = sample_rgb_image();
    let second = DynamicImage::ImageRgb8(ImageBuffer::from_fn(4, 2, |_x, _y| Rgb([255, 0, 0])));

    let image_tensor = processor.process_images(&[first, second]).unwrap();

    assert_eq!(image_tensor.shape(), (2, 3, 2, 4));
    assert_eq!(image_tensor.get(0, 0, 0, 0), Some(1.0));
    assert_eq!(image_tensor.get(0, 0, 0, 2), Some(0.0));
    assert_eq!(image_tensor.get(0, 1, 0, 2), Some(0.0));
    assert_eq!(image_tensor.get(0, 2, 0, 2), Some(0.0));
    assert_eq!(image_tensor.get(1, 0, 1, 3), Some(1.0));
    assert_eq!(image_tensor.get(1, 1, 1, 3), Some(0.0));
    assert_eq!(image_tensor.get(1, 2, 1, 3), Some(0.0));
}

#[test]
fn resizes_to_patch_aligned_longest_edge() {
    let processor =
        image_processor_from_file(fixture_path("processor_small").join("processor_config.json"))
            .unwrap();
    let image = DynamicImage::ImageRgb8(ImageBuffer::from_fn(10, 5, |_x, _y| Rgb([255, 0, 0])));

    let image_tensor = processor.process_image(&image).unwrap();

    assert_eq!(image_tensor.shape(), (1, 3, 2, 4));
}

#[test]
fn loads_image_from_disk() {
    let processor =
        image_processor_from_file(fixture_path("processor_small").join("processor_config.json"))
            .unwrap();
    let image_path = std::env::temp_dir().join(format!(
        "fast-lightonocr-processor-{}.png",
        std::process::id()
    ));
    sample_rgb_image().save(&image_path).unwrap();

    let image_tensor = processor.process_path(&image_path).unwrap();

    assert_eq!(image_tensor.shape(), (1, 3, 2, 2));
    std::fs::remove_file(image_path).ok();
}

#[test]
fn reports_missing_processor_config() {
    let error = ProcessorConfig::from_file(
        fixture_path("lightonocr_missing_config").join("processor_config.json"),
    )
    .expect_err("processor loading should fail");

    match error {
        Error::MissingProcessorAsset { path } => {
            assert_eq!(
                path.file_name().and_then(|name| name.to_str()),
                Some("processor_config.json")
            );
        }
        other => panic!("expected missing processor asset error, got {other:?}"),
    }
}

#[test]
fn rejects_invalid_image_tensor_shapes() {
    let error =
        ImageTensor::new(vec![0.0; 3], 1, 3, 2, 2).expect_err("shape validation should fail");

    match error {
        Error::ImageProcessing { reason } => {
            assert!(reason.contains("does not match shape"));
        }
        other => panic!("expected image processing error, got {other:?}"),
    }
}

#[test]
fn python_pixtral_processor_parity_when_enabled() {
    if std::env::var("LIGHTONOCR_RUN_PYTHON_PROCESSOR_PARITY").is_err() {
        return;
    }

    let processor =
        image_processor_from_file(fixture_path("processor_small").join("processor_config.json"))
            .unwrap();
    let image_path = std::env::temp_dir().join(format!(
        "fast-lightonocr-processor-python-{}.png",
        std::process::id()
    ));
    sample_rgb_image().save(&image_path).unwrap();
    let image_tensor = processor.process_path(&image_path).unwrap();

    let script = format!(
        r#"
import json
from PIL import Image
from transformers.models.pixtral.image_processing_pixtral import PixtralImageProcessor

processor = PixtralImageProcessor(
    do_resize=True,
    size={{"longest_edge": 4}},
    patch_size={{"height": 2, "width": 2}},
    do_rescale=True,
    rescale_factor=1.0 / 255.0,
    do_normalize=True,
    image_mean=[0.0, 0.0, 0.0],
    image_std=[1.0, 1.0, 1.0],
    do_convert_rgb=True,
)
image = Image.open({image_path:?})
out = processor.preprocess(image, return_tensors=None)
print(json.dumps(out["pixel_values"].tolist()))
"#,
        image_path = image_path.display().to_string()
    );

    let output = Command::new("python3")
        .arg("-c")
        .arg(script)
        .output()
        .expect("failed to run python3");
    assert!(
        output.status.success(),
        "python parity script failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let expected: Vec<Vec<Vec<Vec<f32>>>> =
        serde_json::from_slice(&output.stdout).expect("invalid python JSON output");
    let expected_flat = expected
        .into_iter()
        .flatten()
        .flatten()
        .flatten()
        .collect::<Vec<_>>();

    assert_eq!(image_tensor.as_slice(), expected_flat.as_slice());
    std::fs::remove_file(image_path).ok();
}
