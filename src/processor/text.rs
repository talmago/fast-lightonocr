//! Multimodal text processing.

use crate::processor::VisionGrid;
use crate::tokenizer::Tokenizer;
use crate::{Error, Result};

/// Role assigned to a chat message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageRole {
    /// System instruction message.
    System,

    /// User request message.
    User,

    /// Assistant response message.
    Assistant,
}

impl MessageRole {
    /// Returns the role string used by the chat template.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

/// A content item within a chat message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageContent {
    /// Plain text.
    Text(String),

    /// Image placeholder.
    Image,
}

/// A multimodal chat message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    /// Role assigned to this message.
    pub role: MessageRole,

    /// Ordered content items in this message.
    pub content: Vec<MessageContent>,
}

impl Message {
    /// Creates a message from a role and content items.
    #[must_use]
    pub fn new(role: MessageRole, content: Vec<MessageContent>) -> Self {
        Self { role, content }
    }

    /// Creates the default single-image user message used for OCR.
    #[must_use]
    pub fn user_image() -> Self {
        Self {
            role: MessageRole::User,
            content: vec![MessageContent::Image],
        }
    }
}

/// Multimodal text processor.
///
/// The text processor owns the model tokenizer and is responsible for
/// rendering the chat template, tokenizing the prompt, expanding image
/// placeholders into the corresponding vision-token grid, and producing
/// model-ready input IDs.
#[derive(Debug, Clone)]
pub struct TextProcessor {
    tokenizer: Tokenizer,

    image_token: String,
    image_break_token: String,
    image_end_token: String,
}

impl TextProcessor {
    /// Creates a text processor.
    pub fn new(
        tokenizer: Tokenizer,
        image_token: impl Into<String>,
        image_break_token: impl Into<String>,
        image_end_token: impl Into<String>,
    ) -> Self {
        Self {
            tokenizer,
            image_token: image_token.into(),
            image_break_token: image_break_token.into(),
            image_end_token: image_end_token.into(),
        }
    }

    /// Converts a multimodal conversation into model-ready input IDs.
    pub fn process(&self, messages: &[Message], vision_grid: VisionGrid) -> Result<Vec<i64>> {
        let prompt = self.render_template(messages)?;

        let input_ids = self.tokenizer.encode(&prompt)?;

        self.expand_image_placeholders(&input_ids, vision_grid)
    }

    /// Decodes model token IDs into text.
    ///
    /// Special tokens are preserved in the returned string.
    pub fn decode(&self, token_ids: &[i64]) -> Result<String> {
        self.tokenizer.decode(token_ids)
    }

    /// Decodes model token IDs into text while omitting special tokens.
    ///
    /// This is the preferred method for decoding OCR generation output.
    pub fn decode_skip_special_tokens(&self, token_ids: &[i64]) -> Result<String> {
        self.tokenizer.decode_skip_special_tokens(token_ids)
    }

    /// Renders the multimodal conversation into the model chat template.
    fn render_template(&self, messages: &[Message]) -> Result<String> {
        let mut prompt = String::new();

        let eos = self.tokenizer.config().eos_token.as_str();

        if !matches!(messages.first(), Some(message) if message.role == MessageRole::System) {
            prompt.push_str("<|im_start|>");
            prompt.push_str(MessageRole::System.as_str());
            prompt.push_str(eos);
            prompt.push('\n');
        }

        for message in messages {
            prompt.push_str("<|im_start|>");
            prompt.push_str(message.role.as_str());
            prompt.push('\n');

            for content in &message.content {
                match content {
                    MessageContent::Text(text) => {
                        prompt.push_str(text);
                        prompt.push('\n');
                    }
                    MessageContent::Image => {
                        prompt.push_str(&self.image_token);
                    }
                }
            }

            prompt.push_str(eos);
            prompt.push('\n');
        }

        prompt.push_str("<|im_start|>");
        prompt.push_str(MessageRole::Assistant.as_str());
        prompt.push('\n');

        Ok(prompt)
    }

    /// Expands image placeholders into the corresponding vision token grid.
    fn expand_image_placeholders(
        &self,
        input_ids: &[i64],
        vision_grid: VisionGrid,
    ) -> Result<Vec<i64>> {
        let image_token_id = self.tokenizer.token_to_id(&self.image_token)?;

        let placeholder_count = input_ids.iter().filter(|&&id| id == image_token_id).count();

        if placeholder_count != 1 {
            return Err(Error::ImageProcessing {
                reason: format!(
                    "expected exactly one image placeholder, found {placeholder_count}"
                ),
            });
        }

        let image_grid = self.image_token_grid(vision_grid)?;

        let mut expanded = Vec::with_capacity(input_ids.len() + image_grid.len().saturating_sub(1));

        for &id in input_ids {
            if id == image_token_id {
                expanded.extend_from_slice(&image_grid);
            } else {
                expanded.push(id);
            }
        }

        Ok(expanded)
    }

    /// Returns the token sequence representing a vision feature grid.
    fn image_token_grid(&self, grid: VisionGrid) -> Result<Vec<i64>> {
        let image_token_id = self.tokenizer.token_to_id(&self.image_token)?;
        let image_break_token_id = self.tokenizer.token_to_id(&self.image_break_token)?;
        let image_end_token_id = self.tokenizer.token_to_id(&self.image_end_token)?;

        let mut token_ids =
            Vec::with_capacity(grid.feature_count() + grid.height.saturating_sub(1) + 1);

        for row in 0..grid.height {
            token_ids.extend(std::iter::repeat_n(image_token_id, grid.width));

            if row + 1 < grid.height {
                token_ids.push(image_break_token_id);
            }
        }

        token_ids.push(image_end_token_id);

        Ok(token_ids)
    }
}
