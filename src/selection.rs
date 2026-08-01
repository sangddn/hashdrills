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

//! Composable selection of authored drill specifications.
//!
//! Deck selectors compare exact logical deck names. Path selectors compare a
//! drill's source [`DrillSpec::path`], with directories selecting every source
//! file below them. Relative selectors are resolved from the collection root,
//! including when the collection was invoked through a single Markdown file.

use std::error::Error;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fs;
use std::io;
use std::path::Path;
use std::path::PathBuf;

use crate::error::ErrorReport;
use crate::spec::DrillSpec;

/// Include and exclude selectors applied to a parsed collection.
///
/// With no include selectors, selection starts with every specification.
/// Otherwise, a specification is included when any included deck or path
/// matches. Exclusions are then subtracted using the same union semantics.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Selection {
    pub include_decks: Vec<String>,
    pub include_paths: Vec<PathBuf>,
    pub exclude_decks: Vec<String>,
    pub exclude_paths: Vec<PathBuf>,
}

impl Selection {
    /// Whether this selection has no include or exclude selectors.
    pub fn is_empty(&self) -> bool {
        self.include_decks.is_empty()
            && self.include_paths.is_empty()
            && self.exclude_decks.is_empty()
            && self.exclude_paths.is_empty()
    }
}

/// Select drill specifications while retaining their original input order.
///
/// `collection_path` is the same path from which `specs` were parsed. If it is
/// a file, path selectors are resolved relative to that file's parent. Every
/// selector is validated against the full input collection, so a typo is
/// reported even when another selector in the same union matched.
pub fn select_specs<'a>(
    specs: &'a [DrillSpec],
    collection_path: &Path,
    selection: &Selection,
) -> Result<Vec<&'a DrillSpec>, SelectionError> {
    let root = canonical_collection_root(collection_path)?;
    let source_paths = canonical_source_paths(specs, &root)?;
    let include_paths = resolve_selectors(&selection.include_paths, &root, SelectorUse::Include)?;
    let exclude_paths = resolve_selectors(&selection.exclude_paths, &root, SelectorUse::Exclude)?;

    validate_deck_selectors(specs, &selection.include_decks, SelectorUse::Include)?;
    validate_deck_selectors(specs, &selection.exclude_decks, SelectorUse::Exclude)?;
    validate_path_selectors(
        &source_paths,
        &include_paths,
        &selection.include_paths,
        SelectorUse::Include,
    )?;
    validate_path_selectors(
        &source_paths,
        &exclude_paths,
        &selection.exclude_paths,
        SelectorUse::Exclude,
    )?;

    let has_includes = !selection.include_decks.is_empty() || !include_paths.is_empty();
    let mut selected = Vec::new();
    for (spec, source_path) in specs.iter().zip(&source_paths) {
        let included = !has_includes
            || deck_matches(&spec.deck_name, &selection.include_decks)
            || path_matches_any(source_path, &include_paths);
        let excluded = deck_matches(&spec.deck_name, &selection.exclude_decks)
            || path_matches_any(source_path, &exclude_paths);
        if included && !excluded {
            selected.push(spec);
        }
    }
    Ok(selected)
}

/// Whether a selector adds to the candidate set or subtracts from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectorUse {
    Include,
    Exclude,
}

impl SelectorUse {
    fn adjective(self) -> &'static str {
        match self {
            Self::Include => "included",
            Self::Exclude => "excluded",
        }
    }
}

#[derive(Debug)]
struct ResolvedPathSelector {
    path: PathBuf,
    is_directory: bool,
}

impl ResolvedPathSelector {
    fn matches(&self, source_path: &Path) -> bool {
        if self.is_directory {
            source_path.starts_with(&self.path)
        } else {
            source_path == self.path
        }
    }
}

fn canonical_collection_root(collection_path: &Path) -> Result<PathBuf, SelectionError> {
    let canonical =
        fs::canonicalize(collection_path).map_err(|source| SelectionError::CollectionPath {
            path: collection_path.to_path_buf(),
            source,
        })?;
    if canonical.is_dir() {
        return Ok(canonical);
    }
    if canonical.is_file() {
        return Ok(canonical
            .parent()
            .expect("an absolute canonical file path always has a parent")
            .to_path_buf());
    }
    Err(SelectionError::UnsupportedCollectionPath { path: canonical })
}

fn canonical_source_paths(
    specs: &[DrillSpec],
    root: &Path,
) -> Result<Vec<PathBuf>, SelectionError> {
    specs
        .iter()
        .map(|spec| {
            let canonical =
                fs::canonicalize(&spec.path).map_err(|source| SelectionError::SourcePath {
                    path: spec.path.clone(),
                    source,
                })?;
            if !canonical.starts_with(root) {
                return Err(SelectionError::SourceOutsideCollection {
                    path: canonical,
                    root: root.to_path_buf(),
                });
            }
            Ok(canonical)
        })
        .collect()
}

fn resolve_selectors(
    selectors: &[PathBuf],
    root: &Path,
    usage: SelectorUse,
) -> Result<Vec<ResolvedPathSelector>, SelectionError> {
    selectors
        .iter()
        .map(|selector| {
            let candidate = if selector.is_absolute() {
                selector.clone()
            } else {
                root.join(selector)
            };
            let canonical =
                fs::canonicalize(&candidate).map_err(|source| SelectionError::SelectorPath {
                    usage,
                    selector: selector.clone(),
                    root: root.to_path_buf(),
                    source,
                })?;
            if !canonical.starts_with(root) {
                return Err(SelectionError::SelectorOutsideCollection {
                    usage,
                    selector: selector.clone(),
                    resolved: canonical,
                    root: root.to_path_buf(),
                });
            }
            let is_directory = canonical.is_dir();
            if !is_directory && !canonical.is_file() {
                return Err(SelectionError::UnsupportedSelectorPath {
                    usage,
                    selector: selector.clone(),
                    resolved: canonical,
                });
            }
            Ok(ResolvedPathSelector {
                path: canonical,
                is_directory,
            })
        })
        .collect()
}

fn validate_deck_selectors(
    specs: &[DrillSpec],
    selectors: &[String],
    usage: SelectorUse,
) -> Result<(), SelectionError> {
    for selector in selectors {
        if !specs.iter().any(|spec| spec.deck_name == *selector) {
            return Err(SelectionError::UnmatchedDeck {
                usage,
                deck: selector.clone(),
            });
        }
    }
    Ok(())
}

fn validate_path_selectors(
    source_paths: &[PathBuf],
    resolved: &[ResolvedPathSelector],
    authored: &[PathBuf],
    usage: SelectorUse,
) -> Result<(), SelectionError> {
    for (selector, authored) in resolved.iter().zip(authored) {
        if !source_paths.iter().any(|path| selector.matches(path)) {
            return Err(SelectionError::UnmatchedPath {
                usage,
                selector: authored.clone(),
                resolved: selector.path.clone(),
            });
        }
    }
    Ok(())
}

fn deck_matches(deck_name: &str, selectors: &[String]) -> bool {
    selectors.iter().any(|selector| selector == deck_name)
}

fn path_matches_any(source_path: &Path, selectors: &[ResolvedPathSelector]) -> bool {
    selectors
        .iter()
        .any(|selector| selector.matches(source_path))
}

/// An invalid collection or selector supplied to [`select_specs`].
#[derive(Debug)]
pub enum SelectionError {
    CollectionPath {
        path: PathBuf,
        source: io::Error,
    },
    UnsupportedCollectionPath {
        path: PathBuf,
    },
    SourcePath {
        path: PathBuf,
        source: io::Error,
    },
    SourceOutsideCollection {
        path: PathBuf,
        root: PathBuf,
    },
    SelectorPath {
        usage: SelectorUse,
        selector: PathBuf,
        root: PathBuf,
        source: io::Error,
    },
    UnsupportedSelectorPath {
        usage: SelectorUse,
        selector: PathBuf,
        resolved: PathBuf,
    },
    SelectorOutsideCollection {
        usage: SelectorUse,
        selector: PathBuf,
        resolved: PathBuf,
        root: PathBuf,
    },
    UnmatchedDeck {
        usage: SelectorUse,
        deck: String,
    },
    UnmatchedPath {
        usage: SelectorUse,
        selector: PathBuf,
        resolved: PathBuf,
    },
}

impl Display for SelectionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CollectionPath { path, source } => write!(
                f,
                "could not resolve collection path '{}': {source}",
                path.display()
            ),
            Self::UnsupportedCollectionPath { path } => write!(
                f,
                "collection path '{}' is neither a file nor a directory",
                path.display()
            ),
            Self::SourcePath { path, source } => write!(
                f,
                "could not resolve drill source path '{}': {source}",
                path.display()
            ),
            Self::SourceOutsideCollection { path, root } => write!(
                f,
                "drill source path '{}' is outside collection root '{}'",
                path.display(),
                root.display()
            ),
            Self::SelectorPath {
                usage,
                selector,
                root,
                source,
            } => write!(
                f,
                "could not resolve {} path selector '{}' relative to collection root '{}': {source}",
                usage.adjective(),
                selector.display(),
                root.display()
            ),
            Self::UnsupportedSelectorPath {
                usage,
                selector,
                resolved,
            } => write!(
                f,
                "{} path selector '{}' resolves to '{}', which is neither a file nor a directory",
                usage.adjective(),
                selector.display(),
                resolved.display()
            ),
            Self::SelectorOutsideCollection {
                usage,
                selector,
                resolved,
                root,
            } => write!(
                f,
                "{} path selector '{}' resolves outside collection root '{}': '{}'",
                usage.adjective(),
                selector.display(),
                root.display(),
                resolved.display()
            ),
            Self::UnmatchedDeck { usage, deck } => write!(
                f,
                "{} deck selector '{}' matched no drill specifications",
                usage.adjective(),
                deck
            ),
            Self::UnmatchedPath {
                usage,
                selector,
                resolved,
            } => write!(
                f,
                "{} path selector '{}' matched no drill specifications (resolved to '{}')",
                usage.adjective(),
                selector.display(),
                resolved.display()
            ),
        }
    }
}

impl Error for SelectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::CollectionPath { source, .. }
            | Self::SourcePath { source, .. }
            | Self::SelectorPath { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<SelectionError> for ErrorReport {
    fn from(value: SelectionError) -> Self {
        Self::new(format!("selection: {value}"))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use super::*;
    use crate::spec::parse_path;

    type TestResult = Result<(), Box<dyn Error>>;

    fn write_collection(directory: &Path) -> io::Result<()> {
        fs::create_dir_all(directory.join("science/physics"))?;
        fs::create_dir_all(directory.join("arts"))?;
        fs::write(
            directory.join("science/Mechanics.md"),
            "+++\nname = \"Physics\"\n+++\nQ: Mechanics?\nA: Yes.\n",
        )?;
        fs::write(
            directory.join("science/physics/Waves.md"),
            "+++\nname = \"Physics\"\n+++\nQ: Waves?\nA: Yes.\n",
        )?;
        fs::write(
            directory.join("science/Chemistry.md"),
            "+++\nname = \"Chemistry\"\n+++\nQ: Chemistry?\nA: Yes.\n",
        )?;
        fs::write(
            directory.join("arts/Poetry.md"),
            "+++\nname = \"Poetry\"\n+++\nQ: Poetry?\nA: Yes.\n",
        )?;
        Ok(())
    }

    fn selected_files(specs: &[&DrillSpec]) -> Vec<String> {
        let mut files: Vec<String> = specs
            .iter()
            .map(|spec| {
                spec.path
                    .file_name()
                    .expect("test path has a file name")
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        files.sort();
        files
    }

    #[test]
    fn no_includes_starts_with_all_then_subtracts_exclusion_unions() -> TestResult {
        let directory = TempDir::new()?;
        write_collection(directory.path())?;
        let specs = parse_path(directory.path())?;
        let selection = Selection {
            exclude_decks: vec!["Chemistry".to_string()],
            exclude_paths: vec![PathBuf::from("arts")],
            ..Selection::default()
        };

        let selected = select_specs(&specs, directory.path(), &selection)?;

        assert_eq!(selected_files(&selected), ["Mechanics.md", "Waves.md"]);
        Ok(())
    }

    #[test]
    fn include_decks_and_paths_form_a_union_before_exclusions() -> TestResult {
        let directory = TempDir::new()?;
        write_collection(directory.path())?;
        let specs = parse_path(directory.path())?;
        let selection = Selection {
            include_decks: vec!["Poetry".to_string()],
            include_paths: vec![PathBuf::from("science")],
            exclude_decks: vec!["Chemistry".to_string()],
            exclude_paths: vec![PathBuf::from("science/physics")],
        };

        let selected = select_specs(&specs, directory.path(), &selection)?;

        assert_eq!(selected_files(&selected), ["Mechanics.md", "Poetry.md"]);
        Ok(())
    }

    #[test]
    fn exact_deck_union_does_not_treat_names_as_patterns() -> TestResult {
        let directory = TempDir::new()?;
        write_collection(directory.path())?;
        let specs = parse_path(directory.path())?;
        let selection = Selection {
            include_decks: vec!["Physics".to_string(), "Poetry".to_string()],
            ..Selection::default()
        };

        let selected = select_specs(&specs, directory.path(), &selection)?;

        assert_eq!(
            selected_files(&selected),
            ["Mechanics.md", "Poetry.md", "Waves.md"]
        );
        let error = select_specs(
            &specs,
            directory.path(),
            &Selection {
                include_decks: vec!["Phys*".to_string()],
                ..Selection::default()
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("included deck selector 'Phys*'"));
        assert!(
            error
                .to_string()
                .contains("matched no drill specifications")
        );
        Ok(())
    }

    #[test]
    fn file_and_directory_paths_match_drill_source_paths() -> TestResult {
        let directory = TempDir::new()?;
        write_collection(directory.path())?;
        let specs = parse_path(directory.path())?;
        let selection = Selection {
            include_paths: vec![
                PathBuf::from("science/Mechanics.md"),
                directory.path().join("arts"),
            ],
            ..Selection::default()
        };

        let selected = select_specs(&specs, directory.path(), &selection)?;

        assert_eq!(selected_files(&selected), ["Mechanics.md", "Poetry.md"]);
        Ok(())
    }

    #[test]
    fn file_collection_invocation_resolves_selectors_from_parent() -> TestResult {
        let directory = TempDir::new()?;
        write_collection(directory.path())?;
        let file = directory.path().join("science/Mechanics.md");
        let specs = parse_path(&file)?;

        let selected = select_specs(
            &specs,
            &file,
            &Selection {
                include_paths: vec![PathBuf::from("Mechanics.md")],
                ..Selection::default()
            },
        )?;

        assert_eq!(selected_files(&selected), ["Mechanics.md"]);
        Ok(())
    }

    #[test]
    fn every_selector_must_match_even_when_another_selector_does() -> TestResult {
        let directory = TempDir::new()?;
        write_collection(directory.path())?;
        let specs = parse_path(directory.path())?;

        let deck_error = select_specs(
            &specs,
            directory.path(),
            &Selection {
                include_decks: vec!["Physics".to_string(), "Typo".to_string()],
                ..Selection::default()
            },
        )
        .unwrap_err();
        assert!(
            deck_error
                .to_string()
                .contains("included deck selector 'Typo'")
        );

        fs::create_dir(directory.path().join("empty"))?;
        let path_error = select_specs(
            &specs,
            directory.path(),
            &Selection {
                exclude_paths: vec![PathBuf::from("empty")],
                ..Selection::default()
            },
        )
        .unwrap_err();
        assert!(
            path_error
                .to_string()
                .contains("excluded path selector 'empty'")
        );
        assert!(
            path_error
                .to_string()
                .contains("matched no drill specifications")
        );
        Ok(())
    }

    #[test]
    fn selector_paths_cannot_escape_collection_through_dot_dot_or_symlink() -> TestResult {
        let parent = TempDir::new()?;
        let collection = parent.path().join("collection");
        let outside = parent.path().join("outside");
        write_collection(&collection)?;
        fs::create_dir(&outside)?;
        fs::write(outside.join("Outside.md"), "Q: Outside?\nA: Yes.\n")?;
        let specs = parse_path(&collection)?;

        let dot_dot_error = select_specs(
            &specs,
            &collection,
            &Selection {
                include_paths: vec![PathBuf::from("../outside")],
                ..Selection::default()
            },
        )
        .unwrap_err();
        assert!(
            dot_dot_error
                .to_string()
                .contains("resolves outside collection root")
        );

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&outside, collection.join("linked-outside"))?;
            let symlink_error = select_specs(
                &specs,
                &collection,
                &Selection {
                    exclude_paths: vec![PathBuf::from("linked-outside")],
                    ..Selection::default()
                },
            )
            .unwrap_err();
            assert!(
                symlink_error
                    .to_string()
                    .contains("resolves outside collection root")
            );
        }
        Ok(())
    }

    #[test]
    fn missing_path_selector_reports_selector_root_and_io_failure() -> TestResult {
        let directory = TempDir::new()?;
        write_collection(directory.path())?;
        let specs = parse_path(directory.path())?;

        let error = select_specs(
            &specs,
            directory.path(),
            &Selection {
                include_paths: vec![PathBuf::from("missing")],
                ..Selection::default()
            },
        )
        .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("could not resolve included path selector 'missing'"));
        assert!(message.contains(&directory.path().canonicalize()?.display().to_string()));
        Ok(())
    }

    #[test]
    fn source_paths_are_checked_against_collection_root() -> TestResult {
        let parent = TempDir::new()?;
        let collection = parent.path().join("collection");
        let outside = parent.path().join("outside");
        write_collection(&collection)?;
        fs::create_dir(&outside)?;
        let outside_file = outside.join("Outside.md");
        fs::write(&outside_file, "Q: Outside?\nA: Yes.\n")?;
        let outside_spec = parse_path(&outside_file)?.remove(0);

        let error = select_specs(&[outside_spec], &collection, &Selection::default()).unwrap_err();

        assert!(error.to_string().contains("drill source path"));
        assert!(error.to_string().contains("outside collection root"));
        Ok(())
    }
}
