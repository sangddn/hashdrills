// Copyright 2026 Sang Doan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Non-secret, user-level Hashdrills defaults.

use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fmt::{self, Display, Formatter};
use std::fs::{self, File};
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;

use crate::error::ErrorReport;
use crate::model::{ReasoningEffort, SchemaMode};

pub const CONFIG_VERSION: u32 = 1;
const MAX_CONFIG_BYTES: u64 = 64 * 1024;

/// Hashdrills-owned defaults. Credentials and provider state are deliberately
/// absent: Simon Willison's `llm` CLI remains their authority.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UserConfig {
    pub config_version: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluation_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_reasoning_effort: Option<ReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluation_reasoning_effort: Option<ReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema_mode: Option<SchemaMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_timeout: Option<u64>,
}

impl Default for UserConfig {
    fn default() -> Self {
        Self {
            config_version: CONFIG_VERSION,
            model: None,
            generation_model: None,
            evaluation_model: None,
            generation_reasoning_effort: None,
            evaluation_reasoning_effort: None,
            schema_mode: None,
            llm_timeout: None,
        }
    }
}

impl UserConfig {
    pub fn load() -> Result<Self, ConfigError> {
        Self::load_from(&config_path()?)
    }

    pub fn load_from(path: &Path) -> Result<Self, ConfigError> {
        let metadata = match fs::symlink_metadata(path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Self::default()),
            Err(error) => return Err(io_context("inspect", path, error)),
        };
        if metadata.file_type().is_symlink() {
            return Err(ConfigError::new(format!(
                "configuration path '{}' may not be a symbolic link",
                path.display()
            )));
        }
        if !metadata.is_file() {
            return Err(ConfigError::new(format!(
                "configuration path '{}' is not a regular file",
                path.display()
            )));
        }

        let file = File::open(path).map_err(|error| io_context("open", path, error))?;
        let opened_metadata = file
            .metadata()
            .map_err(|error| io_context("inspect open", path, error))?;
        if !opened_metadata.is_file() {
            return Err(ConfigError::new(format!(
                "configuration path '{}' is not a regular file",
                path.display()
            )));
        }
        if opened_metadata.len() > MAX_CONFIG_BYTES {
            return Err(oversized(path));
        }

        let mut bytes = Vec::with_capacity(opened_metadata.len() as usize);
        file.take(MAX_CONFIG_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| io_context("read", path, error))?;
        if bytes.len() as u64 > MAX_CONFIG_BYTES {
            return Err(oversized(path));
        }
        let source = String::from_utf8(bytes).map_err(|_| {
            ConfigError::new(format!(
                "configuration file '{}' is not valid UTF-8",
                path.display()
            ))
        })?;
        let config: Self = toml::from_str(&source).map_err(|error| {
            ConfigError::new(format!(
                "could not parse configuration file '{}': {error}",
                path.display()
            ))
        })?;
        config.validate()?;
        Ok(config)
    }

    pub fn save(&self) -> Result<PathBuf, ConfigError> {
        let path = config_path()?;
        self.save_to(&path)?;
        Ok(path)
    }

    pub fn save_to(&self, path: &Path) -> Result<(), ConfigError> {
        self.validate()?;
        if !path.is_absolute() {
            return Err(ConfigError::new(format!(
                "configuration path '{}' must be absolute",
                path.display()
            )));
        }
        let parent = path
            .parent()
            .filter(|value| !value.as_os_str().is_empty())
            .ok_or_else(|| {
                ConfigError::new(format!(
                    "configuration path '{}' has no parent directory",
                    path.display()
                ))
            })?;
        ensure_private_directory(parent)?;
        reject_unsafe_existing_target(path)?;

        let source = toml::to_string_pretty(self).map_err(|error| {
            ConfigError::new(format!("could not serialize configuration: {error}"))
        })?;
        if source.len() as u64 > MAX_CONFIG_BYTES {
            return Err(oversized(path));
        }
        let mut temporary = NamedTempFile::new_in(parent)
            .map_err(|error| io_context("create temporary file beside", path, error))?;
        set_private_file_permissions(temporary.as_file(), path)?;
        temporary
            .write_all(source.as_bytes())
            .map_err(|error| io_context("write temporary configuration for", path, error))?;
        temporary
            .as_file()
            .sync_all()
            .map_err(|error| io_context("sync temporary configuration for", path, error))?;
        temporary
            .persist(path)
            .map_err(|error| io_context("replace", path, error.error))?;
        File::open(path)
            .and_then(|file| file.sync_all())
            .map_err(|error| io_context("sync", path, error))?;
        sync_directory(parent)?;
        Ok(())
    }

    pub fn reset() -> Result<PathBuf, ConfigError> {
        let path = config_path()?;
        Self::reset_at(&path)?;
        Ok(path)
    }

    pub fn reset_at(path: &Path) -> Result<bool, ConfigError> {
        let metadata = match fs::symlink_metadata(path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(io_context("inspect", path, error)),
        };
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(ConfigError::new(format!(
                "configuration path '{}' is not a regular non-symlink file",
                path.display()
            )));
        }
        fs::remove_file(path).map_err(|error| io_context("remove", path, error))?;
        if let Some(parent) = path.parent() {
            sync_directory(parent)?;
        }
        Ok(true)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.config_version != CONFIG_VERSION {
            return Err(ConfigError::new(format!(
                "unsupported config_version {}; expected {CONFIG_VERSION}",
                self.config_version
            )));
        }
        for (name, value) in [
            ("model", self.model.as_deref()),
            ("generation_model", self.generation_model.as_deref()),
            ("evaluation_model", self.evaluation_model.as_deref()),
        ] {
            if let Some(value) = value {
                validate_model(name, value)?;
            }
        }
        if self.llm_timeout == Some(0) {
            return Err(ConfigError::new("llm_timeout must be at least 1 second"));
        }
        Ok(())
    }
}

/// Resolve the user-level config file without creating it.
pub fn config_path() -> Result<PathBuf, ConfigError> {
    if let Some(value) = env::var_os("HASHDRILLS_CONFIG") {
        return absolute_environment_path("HASHDRILLS_CONFIG", value);
    }

    #[cfg(target_os = "windows")]
    {
        let base = required_absolute_environment_path("APPDATA")?;
        return Ok(base.join("hashdrills").join("config.toml"));
    }

    #[cfg(target_os = "macos")]
    {
        let home = required_absolute_environment_path("HOME")?;
        Ok(home
            .join("Library")
            .join("Application Support")
            .join("hashdrills")
            .join("config.toml"))
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        if let Some(value) = env::var_os("XDG_CONFIG_HOME") {
            if !value.is_empty() {
                let path = PathBuf::from(value);
                if path.is_absolute() {
                    return Ok(path.join("hashdrills").join("config.toml"));
                }
            }
        }
        let home = required_absolute_environment_path("HOME")?;
        Ok(home.join(".config").join("hashdrills").join("config.toml"))
    }
}

fn required_absolute_environment_path(name: &str) -> Result<PathBuf, ConfigError> {
    let value = env::var_os(name).ok_or_else(|| {
        ConfigError::new(format!("{name} is not set; cannot locate configuration"))
    })?;
    absolute_environment_path(name, value)
}

fn absolute_environment_path(name: &str, value: OsString) -> Result<PathBuf, ConfigError> {
    if value.is_empty() {
        return Err(ConfigError::new(format!("{name} may not be empty")));
    }
    let path = PathBuf::from(value);
    if !path.is_absolute() {
        return Err(ConfigError::new(format!("{name} must be an absolute path")));
    }
    Ok(path)
}

fn validate_model(name: &str, value: &str) -> Result<(), ConfigError> {
    if value.trim().is_empty()
        || value.trim() != value
        || value.starts_with('-')
        || value.chars().any(char::is_control)
    {
        return Err(ConfigError::new(format!(
            "{name} must be a nonblank exact model ID without surrounding whitespace, option syntax, or control characters"
        )));
    }
    Ok(())
}

fn reject_unsafe_existing_target(path: &Path) -> Result<(), ConfigError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(ConfigError::new(format!(
            "configuration path '{}' may not be a symbolic link",
            path.display()
        ))),
        Ok(metadata) if !metadata.is_file() => Err(ConfigError::new(format!(
            "configuration path '{}' is not a regular file",
            path.display()
        ))),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(io_context("inspect", path, error)),
    }
}

fn ensure_private_directory(path: &Path) -> Result<(), ConfigError> {
    if path.exists() {
        let metadata = fs::metadata(path).map_err(|error| io_context("inspect", path, error))?;
        if !metadata.is_dir() {
            return Err(ConfigError::new(format!(
                "configuration parent '{}' is not a directory",
                path.display()
            )));
        }
        return Ok(());
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        let mut builder = fs::DirBuilder::new();
        builder.recursive(true).mode(0o700);
        builder
            .create(path)
            .map_err(|error| io_context("create directory", path, error))?;
    }
    #[cfg(not(unix))]
    fs::create_dir_all(path).map_err(|error| io_context("create directory", path, error))?;
    Ok(())
}

#[cfg(unix)]
fn set_private_file_permissions(file: &File, path: &Path) -> Result<(), ConfigError> {
    use std::os::unix::fs::PermissionsExt;
    file.set_permissions(fs::Permissions::from_mode(0o600))
        .map_err(|error| io_context("set private permissions on", path, error))
}

#[cfg(not(unix))]
fn set_private_file_permissions(_file: &File, _path: &Path) -> Result<(), ConfigError> {
    Ok(())
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<(), ConfigError> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| io_context("sync directory", path, error))
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> Result<(), ConfigError> {
    Ok(())
}

fn oversized(path: &Path) -> ConfigError {
    ConfigError::new(format!(
        "configuration file '{}' exceeds the {MAX_CONFIG_BYTES}-byte limit",
        path.display()
    ))
}

fn io_context(operation: &str, path: &Path, error: std::io::Error) -> ConfigError {
    ConfigError::new(format!(
        "could not {operation} configuration path '{}': {error}",
        path.display()
    ))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConfigError {
    message: String,
}

impl ConfigError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for ConfigError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for ConfigError {}

impl From<ConfigError> for ErrorReport {
    fn from(value: ConfigError) -> Self {
        Self::new(format!("config: {value}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(directory: &Path) -> PathBuf {
        directory.join("Hashdrills Config").join("config.toml")
    }

    #[test]
    fn missing_config_loads_defaults() {
        let directory = tempfile::tempdir().unwrap();
        assert_eq!(
            UserConfig::load_from(&path(directory.path())).unwrap(),
            UserConfig::default()
        );
    }

    #[test]
    fn save_load_update_and_reset_round_trip() {
        let directory = tempfile::tempdir().unwrap();
        let path = path(directory.path());
        let mut config = UserConfig {
            model: Some("openrouter/openai/gpt-5".to_string()),
            generation_reasoning_effort: Some(ReasoningEffort::Low),
            schema_mode: Some(SchemaMode::Prompt),
            llm_timeout: Some(45),
            ..UserConfig::default()
        };
        config.save_to(&path).unwrap();
        assert_eq!(UserConfig::load_from(&path).unwrap(), config);

        config.evaluation_model = Some("local-model".to_string());
        config.save_to(&path).unwrap();
        assert_eq!(UserConfig::load_from(&path).unwrap(), config);
        assert!(UserConfig::reset_at(&path).unwrap());
        assert!(!path.exists());
        assert!(!UserConfig::reset_at(&path).unwrap());
    }

    #[test]
    fn malformed_unknown_and_invalid_values_fail_closed() {
        let directory = tempfile::tempdir().unwrap();
        let path = path(directory.path());
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        for source in [
            "not toml",
            "config_version = 1\nsecret_key = \"no\"\n",
            "config_version = 2\n",
            "config_version = 1\nmodel = \" -bad \"\n",
            "config_version = 1\nllm_timeout = 0\n",
        ] {
            fs::write(&path, source).unwrap();
            assert!(UserConfig::load_from(&path).is_err(), "{source}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn saved_config_is_private_and_symlinks_are_rejected() {
        use std::os::unix::fs::{PermissionsExt, symlink};

        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("target.toml");
        fs::write(&target, "sentinel").unwrap();
        let link = directory.path().join("config.toml");
        symlink(&target, &link).unwrap();
        assert!(UserConfig::default().save_to(&link).is_err());
        assert!(UserConfig::load_from(&link).is_err());
        assert!(UserConfig::reset_at(&link).is_err());
        assert_eq!(fs::read_to_string(&target).unwrap(), "sentinel");

        let actual = path(directory.path());
        UserConfig::default().save_to(&actual).unwrap();
        assert_eq!(
            fs::metadata(actual).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
}
