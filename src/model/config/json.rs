use std::fs;
use std::path::Path;

use serde::de::DeserializeOwned;

use crate::{Error, Result};

pub(crate) fn load_json_file<T>(path: impl AsRef<Path>) -> Result<T>
where
    T: DeserializeOwned,
{
    let path = path.as_ref().to_path_buf();
    let contents = fs::read_to_string(&path).map_err(|source| {
        if source.kind() == std::io::ErrorKind::NotFound {
            Error::MissingConfigurationFile { path: path.clone() }
        } else {
            Error::ReadConfigurationFile {
                path: path.clone(),
                source,
            }
        }
    })?;

    serde_json::from_str(&contents).map_err(|source| match source.classify() {
        serde_json::error::Category::Syntax | serde_json::error::Category::Eof => {
            Error::MalformedConfigurationJson { path, source }
        }
        serde_json::error::Category::Data | serde_json::error::Category::Io => {
            Error::InvalidConfiguration { path, source }
        }
    })
}
