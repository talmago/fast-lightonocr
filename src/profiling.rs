//! Lightweight inference profiling for KV-cache and decoder investigation.
//!
//! Compiled in fully when the `profiling` Cargo feature is enabled. Without
//! that feature, this module exposes only inlinable no-op stubs so call sites
//! remain present with zero runtime cost.

#[cfg(feature = "profiling")]
mod active {
    use std::cell::RefCell;
    use std::fmt::Write as _;
    use std::sync::OnceLock;
    use std::time::{Duration, Instant};

    const PROFILE_ENV: &str = "LIGHTONOCR_PROFILE";

    thread_local! {
        static ACTIVE_RUN: RefCell<Option<ProfileRun>> = const { RefCell::new(None) };
    }

    /// Timed inference stage.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum Stage {
        PipelineTotal,
        PrepareInputs,
        ProcessorTotal,
        ImagePreprocessing,
        TextProcessing,
        PlaceholderExpansion,
        AttentionMask,
        VisionEncoder,
        VisionOnnx,
        PromptEmbedding,
        TokenEmbedding,
        EmbeddingOnnx,
        MergeImageFeatures,
        GenerationTotal,
        PrefillDecoder,
        AutoregressiveDecoder,
        DecoderTensorPreparation,
        DecoderOnnx,
        LogitsExtraction,
        KvCacheUpdate,
        KvCacheReplace,
        Sampling,
        GreedyArgmax,
        SampleCandidateBuild,
        SampleTopK,
        SampleTopP,
        SampleDraw,
        AttentionMaskUpdate,
        StreamDecode,
        StreamCallback,
        FinalTextDecode,
    }

    impl Stage {
        const COUNT: usize = 31;

        const fn index(self) -> usize {
            match self {
                Self::PipelineTotal => 0,
                Self::PrepareInputs => 1,
                Self::ProcessorTotal => 2,
                Self::ImagePreprocessing => 3,
                Self::TextProcessing => 4,
                Self::PlaceholderExpansion => 5,
                Self::AttentionMask => 6,
                Self::VisionEncoder => 7,
                Self::VisionOnnx => 8,
                Self::PromptEmbedding => 9,
                Self::TokenEmbedding => 10,
                Self::EmbeddingOnnx => 11,
                Self::MergeImageFeatures => 12,
                Self::GenerationTotal => 13,
                Self::PrefillDecoder => 14,
                Self::AutoregressiveDecoder => 15,
                Self::DecoderTensorPreparation => 16,
                Self::DecoderOnnx => 17,
                Self::LogitsExtraction => 18,
                Self::KvCacheUpdate => 19,
                Self::KvCacheReplace => 20,
                Self::Sampling => 21,
                Self::GreedyArgmax => 22,
                Self::SampleCandidateBuild => 23,
                Self::SampleTopK => 24,
                Self::SampleTopP => 25,
                Self::SampleDraw => 26,
                Self::AttentionMaskUpdate => 27,
                Self::StreamDecode => 28,
                Self::StreamCallback => 29,
                Self::FinalTextDecode => 30,
            }
        }

        const fn label(self) -> &'static str {
            match self {
                Self::PipelineTotal => "total inference",
                Self::PrepareInputs => "input preparation",
                Self::ProcessorTotal => "processor total",
                Self::ImagePreprocessing => "image preprocessing",
                Self::TextProcessing => "text processing",
                Self::PlaceholderExpansion => "image placeholder expansion",
                Self::AttentionMask => "attention mask init",
                Self::VisionEncoder => "vision encoder total",
                Self::VisionOnnx => "vision encoder ONNX",
                Self::PromptEmbedding => "prompt embedding total",
                Self::TokenEmbedding => "generated-token embedding total",
                Self::EmbeddingOnnx => "embedding ONNX",
                Self::MergeImageFeatures => "image feature merge",
                Self::GenerationTotal => "generation total",
                Self::PrefillDecoder => "decoder prefill (inclusive)",
                Self::AutoregressiveDecoder => "decoder autoregressive (inclusive)",
                Self::DecoderTensorPreparation => "decoder tensor preparation",
                Self::DecoderOnnx => "decoder ONNX",
                Self::LogitsExtraction => "decoder logits extraction",
                Self::KvCacheUpdate => "KV cache update/extraction",
                Self::KvCacheReplace => "KV cache replace/drop",
                Self::Sampling => "token selection total",
                Self::GreedyArgmax => "greedy argmax",
                Self::SampleCandidateBuild => "sample candidate build",
                Self::SampleTopK => "sample top-k",
                Self::SampleTopP => "sample top-p",
                Self::SampleDraw => "sample draw",
                Self::AttentionMaskUpdate => "attention mask update",
                Self::StreamDecode => "stream token decode",
                Self::StreamCallback => "stream callback",
                Self::FinalTextDecode => "final text decode",
            }
        }
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct StageMeasurement {
        total: Duration,
        count: usize,
    }

    #[derive(Debug, Default)]
    struct ProfileRun {
        stages: [StageMeasurement; Stage::COUNT],
        generated_tokens: usize,
        kv_extract_bytes: u64,
        kv_extract_tensors: usize,
        logits_extract_bytes: u64,
    }

    impl ProfileRun {
        fn record(&mut self, stage: Stage, duration: Duration) {
            let measurement = &mut self.stages[stage.index()];
            measurement.total += duration;
            measurement.count += 1;
        }

        fn measurement(&self, stage: Stage) -> StageMeasurement {
            self.stages[stage.index()]
        }

        fn print(&self) {
            eprintln!("{}", self.render());
        }

        fn render(&self) -> String {
            let total = self.measurement(Stage::PipelineTotal).total;
            let onnx_total = self.measurement(Stage::VisionOnnx).total
                + self.measurement(Stage::EmbeddingOnnx).total
                + self.measurement(Stage::DecoderOnnx).total;
            let outside_onnx = total.saturating_sub(onnx_total);
            let kv_host = self.measurement(Stage::KvCacheUpdate).total
                + self.measurement(Stage::KvCacheReplace).total
                + self.measurement(Stage::DecoderTensorPreparation).total
                + self.measurement(Stage::LogitsExtraction).total;

            let mut output = String::new();
            let _ = writeln!(output);
            let _ = writeln!(output, "=== LightOnOCR Profiling ===");
            let _ = writeln!(output, "feature: profiling");
            let _ = writeln!(output, "env: {PROFILE_ENV}=1");
            append_stage(&mut output, self, Stage::PipelineTotal);
            append_stage(&mut output, self, Stage::PrepareInputs);
            append_stage(&mut output, self, Stage::ProcessorTotal);
            append_stage(&mut output, self, Stage::ImagePreprocessing);
            append_stage(&mut output, self, Stage::TextProcessing);
            append_stage(&mut output, self, Stage::PlaceholderExpansion);
            append_stage(&mut output, self, Stage::AttentionMask);
            append_stage(&mut output, self, Stage::VisionEncoder);
            append_stage(&mut output, self, Stage::VisionOnnx);
            append_stage(&mut output, self, Stage::PromptEmbedding);
            append_stage(&mut output, self, Stage::TokenEmbedding);
            append_stage(&mut output, self, Stage::EmbeddingOnnx);
            append_stage(&mut output, self, Stage::MergeImageFeatures);
            append_stage(&mut output, self, Stage::GenerationTotal);
            append_stage(&mut output, self, Stage::PrefillDecoder);
            append_stage(&mut output, self, Stage::AutoregressiveDecoder);
            append_stage(&mut output, self, Stage::DecoderOnnx);
            append_stage(&mut output, self, Stage::DecoderTensorPreparation);
            append_stage(&mut output, self, Stage::LogitsExtraction);
            append_stage(&mut output, self, Stage::KvCacheUpdate);
            append_stage(&mut output, self, Stage::KvCacheReplace);
            append_stage(&mut output, self, Stage::Sampling);
            append_stage(&mut output, self, Stage::GreedyArgmax);
            append_stage(&mut output, self, Stage::SampleCandidateBuild);
            append_stage(&mut output, self, Stage::SampleTopK);
            append_stage(&mut output, self, Stage::SampleTopP);
            append_stage(&mut output, self, Stage::SampleDraw);
            append_stage(&mut output, self, Stage::AttentionMaskUpdate);
            append_stage(&mut output, self, Stage::StreamDecode);
            append_stage(&mut output, self, Stage::StreamCallback);
            append_stage(&mut output, self, Stage::FinalTextDecode);

            let decoder_invocations = self.measurement(Stage::PrefillDecoder).count
                + self.measurement(Stage::AutoregressiveDecoder).count;
            let generation = self.measurement(Stage::GenerationTotal).total;
            let decoder_onnx = self.measurement(Stage::DecoderOnnx).total;
            let token_selection = self.measurement(Stage::Sampling).total;
            let token_embed = self.measurement(Stage::TokenEmbedding).total;
            let mask_update = self.measurement(Stage::AttentionMaskUpdate).total;
            let accounted = decoder_onnx
                + kv_host
                + token_selection
                + token_embed
                + mask_update
                + self.measurement(Stage::StreamDecode).total
                + self.measurement(Stage::StreamCallback).total;
            let unaccounted = generation.saturating_sub(accounted);

            let _ = writeln!(output);
            let _ = writeln!(output, "Summary:");
            let _ = writeln!(output, "  generated tokens: {}", self.generated_tokens);
            let _ = writeln!(output, "  decoder invocations: {decoder_invocations}");
            let _ = writeln!(
                output,
                "  prefill invocations: {}",
                self.measurement(Stage::PrefillDecoder).count
            );
            let _ = writeln!(
                output,
                "  autoregressive invocations: {}",
                self.measurement(Stage::AutoregressiveDecoder).count
            );
            let _ = writeln!(output, "  ONNX Runtime total: {}", fmt_duration(onnx_total));
            let _ = writeln!(
                output,
                "  outside ONNX Runtime: {}",
                fmt_duration(outside_onnx)
            );
            let _ = writeln!(
                output,
                "  decoder host (prep+logits+KV): {}",
                fmt_duration(kv_host)
            );
            let _ = writeln!(
                output,
                "  KV extract bytes copied: {} ({:.2} MiB)",
                self.kv_extract_bytes,
                self.kv_extract_bytes as f64 / (1024.0 * 1024.0)
            );
            let _ = writeln!(output, "  KV extract tensors: {}", self.kv_extract_tensors);
            let _ = writeln!(
                output,
                "  logits extract bytes copied: {} ({:.2} MiB)",
                self.logits_extract_bytes,
                self.logits_extract_bytes as f64 / (1024.0 * 1024.0)
            );

            if total > Duration::ZERO {
                let onnx_pct = 100.0 * onnx_total.as_secs_f64() / total.as_secs_f64();
                let kv_pct = 100.0
                    * (self.measurement(Stage::KvCacheUpdate).total
                        + self.measurement(Stage::KvCacheReplace).total)
                        .as_secs_f64()
                    / total.as_secs_f64();
                let _ = writeln!(output, "  ONNX Runtime share of total: {onnx_pct:.1}%");
                let _ = writeln!(output, "  KV update+replace share of total: {kv_pct:.1}%");
            }

            let _ = writeln!(output);
            let _ = writeln!(
                output,
                "Decoder loop attribution (exclusive, vs generation):"
            );
            append_share(&mut output, "decoder ONNX", decoder_onnx, generation);
            append_share(
                &mut output,
                "decoder host (prep+logits+KV)",
                kv_host,
                generation,
            );
            append_share(&mut output, "token selection", token_selection, generation);
            append_share(
                &mut output,
                "  greedy argmax",
                self.measurement(Stage::GreedyArgmax).total,
                generation,
            );
            append_share(
                &mut output,
                "  sample candidate build",
                self.measurement(Stage::SampleCandidateBuild).total,
                generation,
            );
            append_share(
                &mut output,
                "  sample top-k",
                self.measurement(Stage::SampleTopK).total,
                generation,
            );
            append_share(
                &mut output,
                "  sample top-p",
                self.measurement(Stage::SampleTopP).total,
                generation,
            );
            append_share(
                &mut output,
                "  sample draw",
                self.measurement(Stage::SampleDraw).total,
                generation,
            );
            append_share(
                &mut output,
                "generated-token embedding",
                token_embed,
                generation,
            );
            append_share(
                &mut output,
                "attention mask update",
                mask_update,
                generation,
            );
            append_share(
                &mut output,
                "unaccounted in generation",
                unaccounted,
                generation,
            );

            append_average_per_token(&mut output, "generation", generation, self.generated_tokens);
            append_average_per_token(
                &mut output,
                "decoder inclusive",
                self.measurement(Stage::PrefillDecoder).total
                    + self.measurement(Stage::AutoregressiveDecoder).total,
                self.generated_tokens,
            );
            append_average_per_token(
                &mut output,
                "decoder ONNX",
                decoder_onnx,
                self.generated_tokens,
            );
            append_average_per_token(
                &mut output,
                "KV update/extraction",
                self.measurement(Stage::KvCacheUpdate).total,
                self.generated_tokens,
            );
            append_average_per_token(
                &mut output,
                "token selection",
                token_selection,
                self.generated_tokens,
            );
            append_average_per_token(
                &mut output,
                "generated-token embedding",
                token_embed,
                self.generated_tokens,
            );

            output
        }
    }

    /// Runs `work` with profiling when `LIGHTONOCR_PROFILE` is truthy.
    pub(crate) fn run<R>(work: impl FnOnce() -> R) -> R {
        if !enabled() || is_active() {
            return work();
        }

        ACTIVE_RUN.with(|active| {
            *active.borrow_mut() = Some(ProfileRun::default());
        });

        let result = {
            let _timer = start(Stage::PipelineTotal);
            work()
        };

        ACTIVE_RUN.with(|active| {
            if let Some(run) = active.borrow_mut().take() {
                run.print();
            }
        });

        result
    }

    /// Starts a stage timer.
    #[must_use]
    pub(crate) fn start(stage: Stage) -> StageTimer {
        if enabled() && is_active() {
            StageTimer {
                stage,
                started_at: Some(Instant::now()),
            }
        } else {
            StageTimer {
                stage,
                started_at: None,
            }
        }
    }

    /// Records one generated token.
    pub(crate) fn record_generated_token() {
        if !enabled() {
            return;
        }

        ACTIVE_RUN.with(|active| {
            if let Some(run) = active.borrow_mut().as_mut() {
                run.generated_tokens += 1;
            }
        });
    }

    /// Records float32 bytes copied while extracting present KV tensors.
    pub(crate) fn record_kv_extract_bytes(bytes: usize) {
        if !enabled() {
            return;
        }

        ACTIVE_RUN.with(|active| {
            if let Some(run) = active.borrow_mut().as_mut() {
                run.kv_extract_bytes += bytes as u64;
                run.kv_extract_tensors += 1;
            }
        });
    }

    /// Records float32 bytes copied while extracting logits.
    pub(crate) fn record_logits_extract_bytes(bytes: usize) {
        if !enabled() {
            return;
        }

        ACTIVE_RUN.with(|active| {
            if let Some(run) = active.borrow_mut().as_mut() {
                run.logits_extract_bytes += bytes as u64;
            }
        });
    }

    /// RAII timer for a profiling stage.
    #[derive(Debug)]
    pub(crate) struct StageTimer {
        stage: Stage,
        started_at: Option<Instant>,
    }

    impl Drop for StageTimer {
        fn drop(&mut self) {
            let Some(started_at) = self.started_at else {
                return;
            };

            ACTIVE_RUN.with(|active| {
                if let Some(run) = active.borrow_mut().as_mut() {
                    run.record(self.stage, started_at.elapsed());
                }
            });
        }
    }

    fn enabled() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var(PROFILE_ENV)
                .map(|value| {
                    let value = value.trim();
                    !(value.is_empty()
                        || value == "0"
                        || value.eq_ignore_ascii_case("false")
                        || value.eq_ignore_ascii_case("off")
                        || value.eq_ignore_ascii_case("no"))
                })
                .unwrap_or(false)
        })
    }

    fn is_active() -> bool {
        ACTIVE_RUN.with(|active| active.borrow().is_some())
    }

    fn append_stage(output: &mut String, run: &ProfileRun, stage: Stage) {
        let measurement = run.measurement(stage);
        if measurement.count == 0 {
            return;
        }

        let _ = writeln!(
            output,
            "  {:34} {:>10}  calls: {:>4}  avg: {:>10}",
            stage.label(),
            fmt_duration(measurement.total),
            measurement.count,
            fmt_duration(duration_div(measurement.total, measurement.count)),
        );
    }

    fn append_average_per_token(
        output: &mut String,
        label: &str,
        duration: Duration,
        tokens: usize,
    ) {
        if tokens == 0 {
            return;
        }

        let _ = writeln!(
            output,
            "  avg {label} / generated token: {}",
            fmt_duration(duration_div(duration, tokens)),
        );
    }

    fn append_share(output: &mut String, label: &str, part: Duration, whole: Duration) {
        if part == Duration::ZERO && !label.starts_with("unaccounted") {
            return;
        }
        if whole == Duration::ZERO {
            return;
        }
        let pct = 100.0 * part.as_secs_f64() / whole.as_secs_f64();
        let _ = writeln!(
            output,
            "  {label:<34} {:>10}  ({pct:>5.1}% of generation)",
            fmt_duration(part),
        );
    }

    fn duration_div(duration: Duration, divisor: usize) -> Duration {
        if divisor == 0 {
            return Duration::ZERO;
        }

        Duration::from_secs_f64(duration.as_secs_f64() / divisor as f64)
    }

    fn fmt_duration(duration: Duration) -> String {
        format!("{:.3} ms", duration.as_secs_f64() * 1_000.0)
    }
}

#[cfg(feature = "profiling")]
pub(crate) use active::*;

#[cfg(not(feature = "profiling"))]
mod stub {
    /// Timed inference stage (no-op build).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[allow(dead_code)]
    pub(crate) enum Stage {
        PipelineTotal,
        PrepareInputs,
        ProcessorTotal,
        ImagePreprocessing,
        TextProcessing,
        PlaceholderExpansion,
        AttentionMask,
        VisionEncoder,
        VisionOnnx,
        PromptEmbedding,
        TokenEmbedding,
        EmbeddingOnnx,
        MergeImageFeatures,
        GenerationTotal,
        PrefillDecoder,
        AutoregressiveDecoder,
        DecoderTensorPreparation,
        DecoderOnnx,
        LogitsExtraction,
        KvCacheUpdate,
        KvCacheReplace,
        Sampling,
        GreedyArgmax,
        SampleCandidateBuild,
        SampleTopK,
        SampleTopP,
        SampleDraw,
        AttentionMaskUpdate,
        StreamDecode,
        StreamCallback,
        FinalTextDecode,
    }

    /// No-op timer.
    #[derive(Debug, Default)]
    pub(crate) struct StageTimer;

    impl Drop for StageTimer {
        #[inline(always)]
        fn drop(&mut self) {}
    }

    #[inline(always)]
    pub(crate) fn run<R>(work: impl FnOnce() -> R) -> R {
        work()
    }

    #[inline(always)]
    pub(crate) fn start(_stage: Stage) -> StageTimer {
        StageTimer
    }

    #[inline(always)]
    pub(crate) fn record_generated_token() {}

    #[inline(always)]
    pub(crate) fn record_kv_extract_bytes(_bytes: usize) {}

    #[inline(always)]
    pub(crate) fn record_logits_extract_bytes(_bytes: usize) {}
}

#[cfg(not(feature = "profiling"))]
pub(crate) use stub::*;
