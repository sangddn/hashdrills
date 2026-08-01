// Copyright 2025–2026 Fernando Borretti
// Modifications Copyright 2026 Sang Doan
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

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;
#[cfg(unix)]
use std::path::PathBuf;

#[cfg(unix)]
use std::fs::OpenOptions;
#[cfg(unix)]
use std::fs::Permissions;
#[cfg(unix)]
use std::io::ErrorKind;
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use chrono::Datelike;
use chrono::Duration;
use rusqlite::Connection;
#[cfg(unix)]
use rusqlite::OpenFlags;
use rusqlite::OptionalExtension;
use rusqlite::Row;
use rusqlite::Transaction;
use rusqlite::config::DbConfig;
use rusqlite::params;
use serde::Serialize;

use crate::error::ErrorReport;
use crate::error::Fallible;
use crate::error::fail;
use crate::fsrs::Difficulty;
use crate::fsrs::Grade;
use crate::fsrs::Stability;
use crate::model::Evaluation;
use crate::model::GeneratedInstance;
use crate::model::Verdict;
use crate::spec::DrillSpec;
use crate::spec::SpecHash;
use crate::types::date::Date;
use crate::types::performance::Performance;
use crate::types::performance::ReviewedPerformance;
use crate::types::performance::update_performance;
use crate::types::timestamp::Timestamp;

const DATABASE_FILENAME: &str = "hashdrills.db";
const SCHEMA_VERSION: i64 = 2;

/// Schema v1 stored the model evaluation on the attempt and allowed exactly
/// one review. v2 splits those immutable facts and models undo append-only.
/// IDs and all v1 evidence are copied verbatim; unknown learner grades remain
/// NULL instead of being guessed.
const MIGRATE_V1_TO_V2: &str = r#"
drop trigger if exists attempts_are_immutable_update;
drop trigger if exists attempts_are_immutable_delete;
drop trigger if exists reviews_require_schedulable_attempt;
drop trigger if exists reviews_are_immutable_update;
drop trigger if exists reviews_are_immutable_delete;
drop index if exists attempts_answered_at_idx;
drop index if exists reviews_reviewed_at_idx;

alter table attempts rename to attempts_v1;
alter table reviews rename to reviews_v1;

create table attempts (
    attempt_id integer primary key,
    instance_id integer not null unique references instances (instance_id),
    answered_at text not null,
    user_response text not null,
    learner_grade text check (
        learner_grade is null
        or learner_grade in ('forgot', 'hard', 'good', 'easy')
    )
) strict;

create index attempts_answered_at_idx on attempts (answered_at);

create trigger attempts_are_immutable_update
before update on attempts
begin
    select raise(abort, 'attempts are immutable');
end;

create trigger attempts_are_immutable_delete
before delete on attempts
begin
    select raise(abort, 'attempts are immutable');
end;

create table evaluations (
    evaluation_id integer primary key,
    attempt_id integer not null unique references attempts (attempt_id),
    evaluated_at text not null,
    verdict text not null check (
        verdict in ('pass', 'partial', 'fail', 'uncertain', 'invalid')
    ),
    feedback text not null,
    evidence_json text not null check (json_valid(evidence_json)),
    evaluator_model text not null,
    evaluator_protocol_version integer not null check (evaluator_protocol_version > 0)
) strict;

create index evaluations_evaluated_at_idx on evaluations (evaluated_at);

create trigger evaluations_are_immutable_update
before update on evaluations
begin
    select raise(abort, 'evaluations are immutable');
end;

create trigger evaluations_are_immutable_delete
before delete on evaluations
begin
    select raise(abort, 'evaluations are immutable');
end;

create table reviews (
    review_id integer primary key,
    attempt_id integer not null references attempts (attempt_id),
    reviewed_at text not null,
    grade text not null check (grade in ('forgot', 'hard', 'good', 'easy')),
    resolution text not null check (
        resolution in (
            'learner_forgot', 'ai_confirmed', 'ai_rejected',
            'user_override', 'undo_regrade', 'legacy_v1'
        )
    ),
    stability real not null,
    difficulty real not null,
    interval_raw real not null,
    interval_days integer not null,
    due_date text not null,
    review_count integer not null check (review_count > 0)
) strict;

create index reviews_reviewed_at_idx on reviews (reviewed_at);
create index reviews_attempt_id_idx on reviews (attempt_id);

create table review_undos (
    undo_id integer primary key,
    review_id integer not null unique references reviews (review_id),
    undone_at text not null
) strict;

create index review_undos_undone_at_idx on review_undos (undone_at);

create trigger review_undos_are_immutable_update
before update on review_undos
begin
    select raise(abort, 'review undo records are immutable');
end;

create trigger review_undos_are_immutable_delete
before delete on review_undos
begin
    select raise(abort, 'review undo records are immutable');
end;

create trigger reviews_require_no_effective_review
before insert on reviews
when exists (
    select 1
    from reviews
    left join review_undos using (review_id)
    where reviews.attempt_id = new.attempt_id
      and review_undos.undo_id is null
)
begin
    select raise(abort, 'attempt already has an effective review');
end;

create trigger reviews_are_immutable_update
before update on reviews
begin
    select raise(abort, 'reviews are immutable');
end;

create trigger reviews_are_immutable_delete
before delete on reviews
begin
    select raise(abort, 'reviews are immutable');
end;

insert into attempts (
    attempt_id, instance_id, answered_at, user_response, learner_grade
)
select
    attempts_v1.attempt_id,
    attempts_v1.instance_id,
    attempts_v1.answered_at,
    attempts_v1.user_response,
    reviews_v1.grade
from attempts_v1
left join reviews_v1 using (attempt_id);

insert into evaluations (
    evaluation_id, attempt_id, evaluated_at, verdict, feedback, evidence_json,
    evaluator_model, evaluator_protocol_version
)
select
    attempt_id, attempt_id, answered_at, verdict, feedback, evidence_json,
    evaluator_model, evaluator_protocol_version
from attempts_v1;

insert into reviews (
    review_id, attempt_id, reviewed_at, grade, resolution, stability,
    difficulty, interval_raw, interval_days, due_date, review_count
)
select
    review_id, attempt_id, reviewed_at, grade, 'legacy_v1', stability,
    difficulty, interval_raw, interval_days, due_date, review_count
from reviews_v1;

-- Added after the legacy rows are copied: NULL remains an explicit marker for
-- v1 attempts that never reached the old grading step.
create trigger attempts_require_learner_grade
before insert on attempts
when new.learner_grade is null
begin
    select raise(abort, 'new attempts require a learner grade');
end;

-- v1 evaluated every answer before asking for a grade, including answers the
-- learner later marked Forgot. Preserve those rows, then enforce the v2 rule
-- for every future insert.
create trigger evaluations_reject_forgotten_attempts
before insert on evaluations
when (
    select learner_grade from attempts where attempt_id = new.attempt_id
) = 'forgot'
begin
    select raise(abort, 'forgotten attempts must not have an AI evaluation');
end;

drop table reviews_v1;
drop table attempts_v1;

pragma user_version = 2;
"#;

/// Hashdrills' durable, local state.
///
/// Scheduler state belongs to a [`SpecHash`]. Generated instances and their
/// first attempts are append-only evidence beneath that stable identity.
pub struct Storage {
    conn: Connection,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrozenInstanceRow {
    pub instance_id: i64,
    pub session_id: i64,
    pub spec_hash: SpecHash,
    pub displayed_question: String,
    pub target: String,
    pub rubric: String,
    pub model: String,
    pub protocol_version: u32,
    pub frozen_at: Timestamp,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecordedAttempt {
    pub attempt_id: i64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecordedReview {
    pub review_id: i64,
    pub performance: ReviewedPerformance,
}

/// Why the final FSRS grade was accepted.
///
/// This is deliberately separate from both the learner's pre-feedback grade
/// and the model's rubric verdict. `LegacyV1` is only emitted while reading
/// rows migrated from the original schema, which did not record resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewResolution {
    LearnerForgot,
    AiConfirmed,
    AiRejected,
    UserOverride,
    UndoRegrade,
    LegacyV1,
}

impl ReviewResolution {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LearnerForgot => "learner_forgot",
            Self::AiConfirmed => "ai_confirmed",
            Self::AiRejected => "ai_rejected",
            Self::UserOverride => "user_override",
            Self::UndoRegrade => "undo_regrade",
            Self::LegacyV1 => "legacy_v1",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AttemptRow {
    pub attempt_id: i64,
    pub instance_id: i64,
    pub answered_at: Timestamp,
    pub user_response: String,
    /// `None` occurs only for an ungraded attempt migrated from schema v1.
    pub learner_grade: Option<Grade>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluationRow {
    pub evaluation_id: i64,
    pub attempt_id: i64,
    pub evaluated_at: Timestamp,
    pub verdict: Verdict,
    pub feedback: String,
    pub evidence: Vec<String>,
    pub evaluator_model: String,
    pub evaluator_protocol_version: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReviewRow {
    pub review_id: i64,
    pub attempt_id: i64,
    pub reviewed_at: Timestamp,
    pub grade: Grade,
    pub resolution: ReviewResolution,
    pub effective: bool,
    pub performance: ReviewedPerformance,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct StorageStats {
    pub total_specs: usize,
    /// Specs never reviewed. This is authoring inventory, not due debt.
    pub unseen_specs: usize,
    /// Reviewed specs due on or before the requested date.
    pub due_specs: usize,
    pub frozen_instances: usize,
    pub attempts: usize,
    pub reviews: usize,
    pub completed_sessions: usize,
}

/// One local-calendar day of effective reviews for the active collection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReviewActivityDay {
    pub date: Date,
    pub reviews: usize,
}

/// Accepted scheduler grades over a bounded history window.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GradeCounts {
    pub forgot: usize,
    pub hard: usize,
    pub good: usize,
    pub easy: usize,
}

impl GradeCounts {
    fn add(&mut self, grade: Grade, count: usize) {
        let target = match grade {
            Grade::Forgot => &mut self.forgot,
            Grade::Hard => &mut self.hard,
            Grade::Good => &mut self.good,
            Grade::Easy => &mut self.easy,
        };
        *target += count;
    }
}

/// Current FSRS interval distribution for reviewed active drills.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SchedulingHorizons {
    pub under_7_days: usize,
    pub days_7_to_29: usize,
    pub days_30_to_89: usize,
    pub days_90_plus: usize,
}

/// Completion-page analytics, filtered to specifications in the current parse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletionHistory {
    pub stats: StorageStats,
    pub today: Date,
    pub activity_start: Date,
    pub activity: Vec<ReviewActivityDay>,
    pub recent_grades: GradeCounts,
    pub horizons: SchedulingHorizons,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DueSpecRow {
    pub spec_hash: SpecHash,
    pub deck_name: String,
    pub performance: Performance,
}

impl Storage {
    /// Open (or create) `hashdrills.db` beneath a collection root.
    pub fn open(root: impl AsRef<Path>) -> Fallible<Self> {
        let root: &Path = root.as_ref();
        if !root.is_dir() {
            return fail(format!(
                "Hashdrills collection root is not a directory: {}",
                root.display()
            ));
        }
        Self::open_database(root.join(DATABASE_FILENAME))
    }

    /// Open a database at an explicit path.
    ///
    /// Most callers should use [`Storage::open`]. This entry point is useful
    /// for tooling that already resolved the collection's database path.
    pub fn open_database(path: impl AsRef<Path>) -> Fallible<Self> {
        let path = path.as_ref();
        create_private_database_if_missing(path)?;
        #[cfg(unix)]
        let conn = Connection::open_with_flags(
            database_path_without_ancestor_symlinks(path)?,
            OpenFlags::default() | OpenFlags::SQLITE_OPEN_NOFOLLOW,
        )?;
        #[cfg(not(unix))]
        let conn = Connection::open(path)?;
        Self::from_connection(conn)
    }

    fn from_connection(mut conn: Connection) -> Fallible<Self> {
        conn.set_db_config(DbConfig::SQLITE_DBCONFIG_ENABLE_FKEY, true)?;

        let version: i64 = conn.query_row("pragma user_version;", [], |row| row.get(0))?;
        match version {
            SCHEMA_VERSION => validate_schema(&conn)?,
            1 => {
                migrate_v1_to_v2(&mut conn)?;
                validate_schema(&conn)?;
            }
            0 if database_is_empty(&conn)? => {
                let tx = conn.transaction()?;
                tx.execute_batch(include_str!("hashdrills_schema.sql"))?;
                tx.commit()?;
                validate_schema(&conn)?;
            }
            unsupported => {
                return fail(format!(
                    "unsupported hashdrills database schema version {unsupported}; \
                     this build supports version {SCHEMA_VERSION}"
                ));
            }
        }

        let foreign_keys: i64 = conn.query_row("pragma foreign_keys;", [], |row| row.get(0))?;
        if foreign_keys != 1 {
            return fail("failed to enable SQLite foreign-key enforcement");
        }

        Ok(Self { conn })
    }

    /// Add specs that do not yet have scheduler rows.
    ///
    /// Registration never resets an existing schedule. The spec's content
    /// hash, rather than a generated instance, is the scheduler identity.
    pub fn register_specs(
        &mut self,
        specs: &[DrillSpec],
        registered_at: Timestamp,
    ) -> Fallible<usize> {
        let tx = self.conn.transaction()?;
        let mut inserted: usize = 0;
        for spec in specs {
            let spec_hash = spec.hash();
            let question_template = spec.question.to_string();
            let answer_template = spec.answer.to_string();
            let source_file = spec.path.to_string_lossy();
            let was_inserted = tx.execute(
                "insert or ignore into specs \
                 (spec_hash, deck_name, goal, question_template, answer_template, \
                  source_file, source_provenance, line_start, line_end, registered_at, \
                  review_count) \
                 values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, 0);",
                params![
                    spec_hash,
                    &spec.deck_name,
                    &spec.goal,
                    &question_template,
                    &answer_template,
                    source_file.as_ref(),
                    &spec.source,
                    spec.range.0 as i64,
                    spec.range.1 as i64,
                    registered_at
                ],
            )?;
            inserted += was_inserted;

            if was_inserted == 0 {
                let existing_contract: (Option<String>, String, String) = tx.query_row(
                    "select goal, question_template, answer_template \
                     from specs where spec_hash = ?1;",
                    params![spec_hash],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )?;
                let current_contract = (
                    spec.goal.clone(),
                    question_template.clone(),
                    answer_template.clone(),
                );
                if existing_contract != current_contract {
                    return fail(format!(
                        "drill spec hash collision or incompatible canonicalization for {spec_hash}"
                    ));
                }

                // Deck and source location are discovery metadata, not part
                // of identity. Keep them current when a note is moved without
                // disturbing its established schedule.
                tx.execute(
                    "update specs set \
                         deck_name = ?1, source_file = ?2, source_provenance = ?3, \
                         line_start = ?4, line_end = ?5 \
                     where spec_hash = ?6;",
                    params![
                        &spec.deck_name,
                        source_file.as_ref(),
                        &spec.source,
                        spec.range.0 as i64,
                        spec.range.1 as i64,
                        spec_hash
                    ],
                )?;
            }
        }
        tx.commit()?;
        Ok(inserted)
    }

    /// Return every unseen, due, or overdue spec.
    pub fn all_due(&self, today: Date) -> Fallible<HashSet<SpecHash>> {
        due_query(&self.conn, None, today)
    }

    /// Return every unseen, due, or overdue spec in one logical deck.
    pub fn due_for_deck(&self, deck_name: &str, today: Date) -> Fallible<HashSet<SpecHash>> {
        due_query(&self.conn, Some(deck_name), today)
    }

    /// Count reviewed due and overdue specs by logical deck. Unseen inventory
    /// is reported separately and admitted by [`Storage::due_queue`].
    pub fn due_counts_by_deck(&self, today: Date) -> Fallible<BTreeMap<String, usize>> {
        let mut stmt = self.conn.prepare(
            "select deck_name, count(*) \
             from specs \
             where due_date is not null and due_date <= ?1 \
             group by deck_name \
             order by deck_name;",
        )?;
        let rows = stmt.query_map(params![today], |row| {
            let deck_name: String = row.get(0)?;
            let count: i64 = row.get(1)?;
            Ok((deck_name, count as usize))
        })?;
        let mut counts = BTreeMap::new();
        for row in rows {
            let (deck_name, count) = row?;
            counts.insert(deck_name, count);
        }
        Ok(counts)
    }

    /// Reviewed due-debt counts limited to specs present in the current parse.
    /// Database rows for edited-away/orphaned specs are deliberately ignored.
    pub fn due_counts_for_specs(
        &self,
        today: Date,
        specs: &[DrillSpec],
    ) -> Fallible<BTreeMap<String, usize>> {
        let mut counts = BTreeMap::new();
        for spec in specs {
            if let Some(Performance::Reviewed(performance)) = self.performance_opt(spec.hash())? {
                if performance.due_date <= today {
                    *counts.entry(spec.deck_name.clone()).or_insert(0) += 1;
                }
            }
        }
        Ok(counts)
    }

    /// Select an ordered practice queue, prioritizing reviewed due debt before
    /// admitting unseen inventory. Limits are applied after the exact logical
    /// deck filter; `new_limit` affects only unseen specs.
    pub fn due_queue(
        &self,
        today: Date,
        deck_name: Option<&str>,
        new_limit: Option<usize>,
        total_limit: Option<usize>,
    ) -> Fallible<Vec<DueSpecRow>> {
        let mut stmt = self.conn.prepare(
            "select spec_hash, deck_name, last_reviewed_at, stability, difficulty, \
                    interval_raw, interval_days, due_date, review_count \
             from specs \
             where (due_date is null or due_date <= ?1) \
               and (?2 is null or deck_name = ?2) \
             order by (due_date is null), due_date, registered_at, spec_hash;",
        )?;
        let rows = stmt.query_map(params![today, deck_name], |row| {
            Ok(DueSpecRow {
                spec_hash: row.get(0)?,
                deck_name: row.get(1)?,
                performance: performance_from_row_at(row, 2)?,
            })
        })?;

        let mut due_rows = Vec::new();
        for row in rows {
            due_rows.push(row?);
        }
        Ok(apply_queue_limits(due_rows, new_limit, total_limit))
    }

    /// Select a queue limited to specs present in the current filesystem
    /// parse, so orphaned scheduler rows cannot consume queue limits.
    pub fn due_queue_for_specs(
        &self,
        today: Date,
        specs: &[DrillSpec],
        deck_name: Option<&str>,
        new_limit: Option<usize>,
        total_limit: Option<usize>,
    ) -> Fallible<Vec<DueSpecRow>> {
        let current: HashSet<SpecHash> = specs.iter().map(DrillSpec::hash).collect();
        let mut rows = self.due_queue(today, deck_name, None, None)?;
        rows.retain(|row| current.contains(&row.spec_hash));
        Ok(apply_queue_limits(rows, new_limit, total_limit))
    }

    pub fn performance_opt(&self, spec_hash: SpecHash) -> Fallible<Option<Performance>> {
        load_performance(&self.conn, spec_hash)
    }

    pub fn performance(&self, spec_hash: SpecHash) -> Fallible<Performance> {
        self.performance_opt(spec_hash)?.ok_or_else(|| {
            ErrorReport::new(format!(
                "no scheduler state found for drill spec {spec_hash}"
            ))
        })
    }

    pub fn stats(&self, today: Date) -> Fallible<StorageStats> {
        let total_specs = count(&self.conn, "select count(*) from specs;", params![])?;
        let unseen_specs = count(
            &self.conn,
            "select count(*) from specs where due_date is null;",
            params![],
        )?;
        let due_specs = count(
            &self.conn,
            "select count(*) from specs where due_date is not null and due_date <= ?1;",
            params![today],
        )?;
        let frozen_instances = count(&self.conn, "select count(*) from instances;", params![])?;
        let attempts = count(&self.conn, "select count(*) from attempts;", params![])?;
        let reviews = count(&self.conn, "select count(*) from reviews;", params![])?;
        let completed_sessions = count(
            &self.conn,
            "select count(*) from sessions where ended_at is not null;",
            params![],
        )?;
        Ok(StorageStats {
            total_specs,
            unseen_specs,
            due_specs,
            frozen_instances,
            attempts,
            reviews,
            completed_sessions,
        })
    }

    /// Minimal statistics limited to the currently parsed authored specs.
    /// Historical/orphaned rows remain in the database for auditability but
    /// do not inflate the active collection's totals.
    pub fn stats_for_specs(&self, today: Date, specs: &[DrillSpec]) -> Fallible<StorageStats> {
        let current: HashSet<SpecHash> = specs.iter().map(DrillSpec::hash).collect();
        let total_specs = current.len();
        let mut unseen_specs = 0usize;
        let mut due_specs = 0usize;
        for spec_hash in &current {
            match self.performance_opt(*spec_hash)? {
                None | Some(Performance::New) => {
                    unseen_specs += 1;
                }
                Some(Performance::Reviewed(performance)) if performance.due_date <= today => {
                    due_specs += 1;
                }
                Some(Performance::Reviewed(_)) => {}
            }
        }

        let frozen_instances = sum_current_counts(
            &self.conn,
            "select spec_hash, count(*) from instances group by spec_hash;",
            &current,
        )?;
        let attempts = sum_current_counts(
            &self.conn,
            "select instances.spec_hash, count(*) \
             from attempts join instances using (instance_id) \
             group by instances.spec_hash;",
            &current,
        )?;
        let reviews = sum_current_counts(
            &self.conn,
            "select instances.spec_hash, count(*) \
             from reviews \
             join attempts using (attempt_id) \
             join instances using (instance_id) \
             group by instances.spec_hash;",
            &current,
        )?;
        let completed_sessions = count_current_completed_sessions(&self.conn, &current)?;
        Ok(StorageStats {
            total_specs,
            unseen_specs,
            due_specs,
            frozen_instances,
            attempts,
            reviews,
            completed_sessions,
        })
    }

    /// Return undo-correct review history for the active authored collection.
    ///
    /// The heatmap covers 53 Sunday-start weeks. Grade counts cover today and
    /// the preceding 29 days. Append-only attempts and undone reviews are not
    /// treated as successful activity.
    pub fn completion_history_for_specs(
        &self,
        today: Date,
        specs: &[DrillSpec],
    ) -> Fallible<CompletionHistory> {
        let current: HashSet<SpecHash> = specs.iter().map(DrillSpec::hash).collect();
        let today_inner = today.into_inner();
        let sunday_offset = i64::from(today_inner.weekday().num_days_from_sunday());
        let activity_start =
            Date::new(today_inner - Duration::days(sunday_offset) - Duration::weeks(52));
        let recent_start = Date::new(today_inner - Duration::days(29));
        let tomorrow = Date::new(today_inner + Duration::days(1));

        let mut activity_by_day: BTreeMap<Date, usize> = BTreeMap::new();
        let mut recent_grades = GradeCounts::default();
        let mut stmt = self.conn.prepare(
            "select instances.spec_hash, substr(reviews.reviewed_at, 1, 10), \
                    reviews.grade, count(*) \
             from reviews \
             join attempts using (attempt_id) \
             join instances using (instance_id) \
             left join review_undos using (review_id) \
             where review_undos.undo_id is null \
               and reviews.reviewed_at >= ?1 \
               and reviews.reviewed_at < ?2 \
             group by instances.spec_hash, substr(reviews.reviewed_at, 1, 10), \
                      reviews.grade \
             order by reviews.reviewed_at;",
        )?;
        let rows = stmt.query_map(params![activity_start, tomorrow], |row| {
            let spec_hash: SpecHash = row.get(0)?;
            let date: String = row.get(1)?;
            let grade: Grade = row.get(2)?;
            let count: i64 = row.get(3)?;
            Ok((spec_hash, date, grade, count as usize))
        })?;
        for row in rows {
            let (spec_hash, date, grade, count) = row?;
            if !current.contains(&spec_hash) {
                continue;
            }
            let date = Date::try_from(date)?;
            *activity_by_day.entry(date).or_insert(0) += count;
            if date >= recent_start {
                recent_grades.add(grade, count);
            }
        }

        let mut horizons = SchedulingHorizons::default();
        for spec_hash in current {
            let Some(Performance::Reviewed(performance)) = self.performance_opt(spec_hash)? else {
                continue;
            };
            match performance.interval_days {
                ..=6 => horizons.under_7_days += 1,
                7..=29 => horizons.days_7_to_29 += 1,
                30..=89 => horizons.days_30_to_89 += 1,
                90.. => horizons.days_90_plus += 1,
            }
        }

        Ok(CompletionHistory {
            stats: self.stats_for_specs(today, specs)?,
            today,
            activity_start,
            activity: activity_by_day
                .into_iter()
                .map(|(date, reviews)| ReviewActivityDay { date, reviews })
                .collect(),
            recent_grades,
            horizons,
        })
    }

    /// Start a session and return its database identity.
    pub fn begin_session(&self, started_at: Timestamp) -> Fallible<i64> {
        let session_id: i64 = self.conn.query_row(
            "insert into sessions (started_at) values (?1) returning session_id;",
            params![started_at],
            |row| row.get(0),
        )?;
        Ok(session_id)
    }

    /// Finish an active session. A session may only be finished once.
    pub fn finish_session(&self, session_id: i64, ended_at: Timestamp) -> Fallible<()> {
        let updated = self.conn.execute(
            "update sessions set ended_at = ?1 \
             where session_id = ?2 and ended_at is null;",
            params![ended_at, session_id],
        )?;
        if updated != 1 {
            return fail(format!("active session {session_id} was not found"));
        }
        Ok(())
    }

    /// Freeze the generated question and hidden grading material before the
    /// learner can answer it.
    pub fn freeze_instance(
        &mut self,
        session_id: i64,
        spec_hash: SpecHash,
        instance: &GeneratedInstance,
        frozen_at: Timestamp,
    ) -> Fallible<i64> {
        let tx = self.conn.transaction()?;
        require_active_session(&tx, session_id)?;
        let instance_id: i64 = tx.query_row(
            "insert into instances (\
                 session_id, spec_hash, displayed_question, target, rubric, \
                 model, protocol_version, frozen_at\
             ) values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) \
             returning instance_id;",
            params![
                session_id,
                spec_hash,
                &instance.question,
                &instance.target,
                &instance.rubric,
                &instance.model,
                i64::from(instance.protocol_version),
                frozen_at
            ],
            |row| row.get(0),
        )?;
        tx.commit()?;
        Ok(instance_id)
    }

    pub fn frozen_instance(&self, instance_id: i64) -> Fallible<FrozenInstanceRow> {
        self.conn
            .query_row(
                "select instance_id, session_id, spec_hash, displayed_question, \
                        target, rubric, model, protocol_version, frozen_at \
                 from instances where instance_id = ?1;",
                params![instance_id],
                frozen_instance_from_row,
            )
            .optional()?
            .ok_or_else(|| ErrorReport::new(format!("frozen instance {instance_id} was not found")))
    }

    pub fn attempt(&self, attempt_id: i64) -> Fallible<AttemptRow> {
        self.conn
            .query_row(
                "select attempt_id, instance_id, answered_at, user_response, learner_grade \
                 from attempts where attempt_id = ?1;",
                params![attempt_id],
                |row| {
                    Ok(AttemptRow {
                        attempt_id: row.get(0)?,
                        instance_id: row.get(1)?,
                        answered_at: row.get(2)?,
                        user_response: row.get(3)?,
                        learner_grade: row.get(4)?,
                    })
                },
            )
            .optional()?
            .ok_or_else(|| ErrorReport::new(format!("attempt {attempt_id} was not found")))
    }

    pub fn evaluation_for_attempt(&self, attempt_id: i64) -> Fallible<Option<EvaluationRow>> {
        type RawEvaluation = (i64, i64, Timestamp, String, String, String, String, i64);
        let raw: Option<RawEvaluation> = self
            .conn
            .query_row(
                "select evaluation_id, attempt_id, evaluated_at, verdict, feedback, \
                        evidence_json, evaluator_model, evaluator_protocol_version \
                 from evaluations where attempt_id = ?1;",
                params![attempt_id],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                        row.get(6)?,
                        row.get(7)?,
                    ))
                },
            )
            .optional()?;
        let Some(raw) = raw else {
            return Ok(None);
        };
        let verdict = verdict_from_str(&raw.3)?;
        let evidence: Vec<String> = serde_json::from_str(&raw.5)?;
        let evaluator_protocol_version: u32 = raw.7.try_into().map_err(|_| {
            ErrorReport::new(format!(
                "invalid evaluator protocol version {} in attempt {attempt_id}",
                raw.7
            ))
        })?;
        Ok(Some(EvaluationRow {
            evaluation_id: raw.0,
            attempt_id: raw.1,
            evaluated_at: raw.2,
            verdict,
            feedback: raw.4,
            evidence,
            evaluator_model: raw.6,
            evaluator_protocol_version,
        }))
    }

    pub fn review_for_attempt(&self, attempt_id: i64) -> Fallible<Option<ReviewRow>> {
        let review = self
            .conn
            .query_row(
                "select review_id, attempt_id, reviewed_at, grade, stability, \
                        difficulty, interval_raw, interval_days, due_date, review_count, \
                        resolution, true \
                 from reviews \
                 where attempt_id = ?1 \
                   and not exists (\
                       select 1 from review_undos \
                       where review_undos.review_id = reviews.review_id\
                   ) \
                 order by review_id desc limit 1;",
                params![attempt_id],
                review_row_from_row,
            )
            .optional()?;
        Ok(review)
    }

    /// Return an accepted review whether it is still effective or was undone.
    pub fn review(&self, review_id: i64) -> Fallible<ReviewRow> {
        self.conn
            .query_row(
                "select reviews.review_id, attempt_id, reviewed_at, grade, stability, \
                        difficulty, interval_raw, interval_days, due_date, review_count, \
                        resolution, (review_undos.undo_id is null) \
                 from reviews \
                 left join review_undos using (review_id) \
                 where reviews.review_id = ?1;",
                params![review_id],
                review_row_from_row,
            )
            .optional()?
            .ok_or_else(|| ErrorReport::new(format!("review {review_id} was not found")))
    }

    /// Persist the first submitted response and the learner's pre-feedback
    /// grade before any optional model evaluation. The row is immutable.
    pub fn record_answer(
        &mut self,
        instance_id: i64,
        user_response: &str,
        learner_grade: Grade,
        answered_at: Timestamp,
    ) -> Fallible<RecordedAttempt> {
        let tx = self.conn.transaction()?;
        let attempt_id =
            insert_answer(&tx, instance_id, user_response, learner_grade, answered_at)?;
        tx.commit()?;
        Ok(RecordedAttempt { attempt_id })
    }

    /// Compatibility name for the web runtime: an "attempt" here is still
    /// only the immutable answer plus learner grade, never an evaluation.
    pub fn record_attempt(
        &mut self,
        instance_id: i64,
        user_response: &str,
        learner_grade: Grade,
        answered_at: Timestamp,
    ) -> Fallible<RecordedAttempt> {
        self.record_answer(instance_id, user_response, learner_grade, answered_at)
    }

    /// Attach the model's independent second check. A learner-selected Forgot
    /// rejects this operation, and the UNIQUE attempt key permits it once.
    pub fn record_evaluation(
        &mut self,
        attempt_id: i64,
        evaluation: &Evaluation,
        evaluated_at: Timestamp,
    ) -> Fallible<()> {
        let tx = self.conn.transaction()?;
        insert_evaluation(&tx, attempt_id, evaluation, evaluated_at)?;
        tx.commit()?;
        Ok(())
    }

    pub fn attach_evaluation(
        &mut self,
        attempt_id: i64,
        evaluation: &Evaluation,
        evaluated_at: Timestamp,
    ) -> Fallible<()> {
        self.record_evaluation(attempt_id, evaluation, evaluated_at)
    }

    /// Accept one final FSRS grade with its resolution provenance. Inserting
    /// the immutable review and updating the scheduler snapshot are atomic.
    pub fn accept_grade(
        &mut self,
        attempt_id: i64,
        grade: Grade,
        resolution: ReviewResolution,
        reviewed_at: Timestamp,
    ) -> Fallible<RecordedReview> {
        let tx = self.conn.transaction()?;
        let review = insert_review_and_update(&tx, attempt_id, grade, resolution, reviewed_at)?;
        tx.commit()?;
        Ok(review)
    }

    /// Undo the latest effective review for its spec. The accepted review is
    /// retained, an immutable undo record is appended, the scheduler snapshot
    /// is restored to the preceding effective review (or New), and the owning
    /// session is reopened so a replacement grade can be accepted and closed.
    pub fn undo_review(&mut self, review_id: i64, undone_at: Timestamp) -> Fallible<()> {
        let tx = self.conn.transaction()?;
        undo_review_and_restore(&tx, review_id, undone_at)?;
        tx.commit()?;
        Ok(())
    }
}

/// Pre-create a new on-disk database without granting group or world access.
///
/// `create_new` makes the existence check atomic. In particular, another
/// process creating the database first cannot cause us to change permissions
/// on a file that was already present. SQLite's special in-memory path remains
/// untouched.
#[cfg(unix)]
fn create_private_database_if_missing(path: &Path) -> Fallible<()> {
    if sqlite_uses_virtual_path(path) {
        return Ok(());
    }

    match OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
    {
        Ok(file) => {
            // `mode` is filtered through the process umask. Restore owner
            // read/write access while keeping group and world bits clear.
            file.set_permissions(Permissions::from_mode(0o600))?;
            Ok(())
        }
        Err(error) if error.kind() == ErrorKind::AlreadyExists => {
            require_regular_database_path(path)
        }
        Err(error) => Err(error.into()),
    }
}

/// Resolve symlinks in parent directories before asking SQLite to reject a
/// symlink in the final database component. SQLite's `NOFOLLOW` flag rejects
/// symlinks anywhere in the supplied path; resolving only the parent keeps
/// legitimate collection paths such as macOS's `/var` working without ever
/// resolving the database filename itself.
#[cfg(unix)]
fn database_path_without_ancestor_symlinks(path: &Path) -> Fallible<PathBuf> {
    if sqlite_uses_virtual_path(path) {
        return Ok(path.to_path_buf());
    }

    let file_name = path.file_name().ok_or_else(|| {
        ErrorReport::new(format!(
            "database path does not name a file: {}",
            path.display()
        ))
    })?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let parent = if parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent
    };
    Ok(parent.canonicalize()?.join(file_name))
}

#[cfg(unix)]
fn sqlite_uses_virtual_path(path: &Path) -> bool {
    let path_bytes = path.as_os_str().as_bytes();
    path_bytes.is_empty() || path_bytes == b":memory:" || path_bytes.starts_with(b"file:")
}

#[cfg(not(unix))]
fn create_private_database_if_missing(path: &Path) -> Fallible<()> {
    if path.as_os_str().is_empty() || path == Path::new(":memory:") {
        return Ok(());
    }

    match path.symlink_metadata() {
        Ok(_) => require_regular_database_path(path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn require_regular_database_path(path: &Path) -> Fallible<()> {
    let metadata = path.symlink_metadata()?;
    let file_type = metadata.file_type();
    if file_type.is_symlink() {
        return fail(format!(
            "refusing to open database path {}: symbolic links are not allowed",
            path.display()
        ));
    }
    if !file_type.is_file() {
        return fail(format!(
            "refusing to open database path {}: expected a regular file",
            path.display()
        ));
    }
    Ok(())
}

fn insert_answer(
    tx: &Transaction<'_>,
    instance_id: i64,
    user_response: &str,
    learner_grade: Grade,
    answered_at: Timestamp,
) -> Fallible<i64> {
    let session_id: i64 = tx
        .query_row(
            "select session_id from instances where instance_id = ?1;",
            params![instance_id],
            |row| row.get(0),
        )
        .optional()?
        .ok_or_else(|| ErrorReport::new(format!("frozen instance {instance_id} was not found")))?;
    require_active_session(tx, session_id)?;

    let attempt_id: i64 = tx.query_row(
        "insert into attempts (\
             instance_id, answered_at, user_response, learner_grade\
         ) values (?1, ?2, ?3, ?4) \
         returning attempt_id;",
        params![instance_id, answered_at, user_response, learner_grade],
        |row| row.get(0),
    )?;
    Ok(attempt_id)
}

fn insert_evaluation(
    tx: &Transaction<'_>,
    attempt_id: i64,
    evaluation: &Evaluation,
    evaluated_at: Timestamp,
) -> Fallible<()> {
    let exists: bool = tx.query_row(
        "select exists(select 1 from attempts where attempt_id = ?1);",
        params![attempt_id],
        |row| row.get(0),
    )?;
    if !exists {
        return fail(format!("attempt {attempt_id} was not found"));
    }

    // Schema v1/v2 retains this column for replaying historical evaluations,
    // but the current model contract intentionally asks for no evidence list.
    let evidence_json = "[]";
    tx.execute(
        "insert into evaluations (\
             attempt_id, evaluated_at, verdict, feedback, evidence_json, \
             evaluator_model, evaluator_protocol_version\
         ) values (?1, ?2, ?3, ?4, ?5, ?6, ?7);",
        params![
            attempt_id,
            evaluated_at,
            evaluation.verdict.as_str(),
            &evaluation.feedback,
            evidence_json,
            &evaluation.model,
            i64::from(evaluation.protocol_version)
        ],
    )?;
    Ok(())
}

fn insert_review_and_update(
    tx: &Transaction<'_>,
    attempt_id: i64,
    grade: Grade,
    resolution: ReviewResolution,
    reviewed_at: Timestamp,
) -> Fallible<RecordedReview> {
    let (spec_hash, learner_grade): (SpecHash, Option<Grade>) = tx
        .query_row(
            "select instances.spec_hash, attempts.learner_grade \
             from attempts \
             join instances using (instance_id) \
             where attempts.attempt_id = ?1;",
            params![attempt_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?
        .ok_or_else(|| ErrorReport::new(format!("attempt {attempt_id} was not found")))?;
    validate_review_resolution(tx, attempt_id, learner_grade, grade, resolution)?;

    let current = load_performance(tx, spec_hash)?.ok_or_else(|| {
        ErrorReport::new(format!(
            "no scheduler state found for drill spec {spec_hash}"
        ))
    })?;
    let performance = update_performance(current, grade, reviewed_at);

    let review_id: i64 = tx.query_row(
        "insert into reviews (\
             attempt_id, reviewed_at, grade, resolution, stability, difficulty, \
             interval_raw, interval_days, due_date, review_count\
         ) values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10) \
         returning review_id;",
        params![
            attempt_id,
            reviewed_at,
            grade,
            resolution.as_str(),
            performance.stability,
            performance.difficulty,
            performance.interval_raw,
            performance.interval_days,
            performance.due_date,
            performance.review_count as i64
        ],
        |row| row.get(0),
    )?;

    let updated = tx.execute(
        "update specs set \
             last_reviewed_at = ?1, stability = ?2, difficulty = ?3, \
             interval_raw = ?4, interval_days = ?5, due_date = ?6, \
             review_count = ?7 \
         where spec_hash = ?8;",
        params![
            performance.last_reviewed_at,
            performance.stability,
            performance.difficulty,
            performance.interval_raw,
            performance.interval_days,
            performance.due_date,
            performance.review_count as i64,
            spec_hash
        ],
    )?;
    if updated != 1 {
        return fail(format!(
            "scheduler state disappeared for drill spec {spec_hash}"
        ));
    }

    Ok(RecordedReview {
        review_id,
        performance,
    })
}

fn validate_review_resolution(
    tx: &Transaction<'_>,
    attempt_id: i64,
    learner_grade: Option<Grade>,
    grade: Grade,
    resolution: ReviewResolution,
) -> Fallible<()> {
    let verdict: Option<String> = tx
        .query_row(
            "select verdict from evaluations where attempt_id = ?1;",
            params![attempt_id],
            |row| row.get(0),
        )
        .optional()?;
    let had_undone_review: bool = tx.query_row(
        "select exists(\
             select 1 from reviews \
             join review_undos using (review_id) \
             where reviews.attempt_id = ?1\
         );",
        params![attempt_id],
        |row| row.get(0),
    )?;

    let valid = match resolution {
        ReviewResolution::LearnerForgot => {
            learner_grade == Some(Grade::Forgot) && grade == Grade::Forgot && verdict.is_none()
        }
        ReviewResolution::AiConfirmed => {
            learner_grade.is_some_and(|learner| learner != Grade::Forgot && learner == grade)
                && verdict.as_deref() == Some(Verdict::Pass.as_str())
        }
        ReviewResolution::AiRejected => {
            learner_grade.is_some_and(|learner| learner != Grade::Forgot)
                && grade == Grade::Forgot
                && matches!(verdict.as_deref(), Some("partial") | Some("fail"))
        }
        ReviewResolution::UserOverride => {
            learner_grade.is_some_and(|learner| learner != Grade::Forgot && learner == grade)
                && matches!(verdict.as_deref(), Some("partial") | Some("fail"))
        }
        ReviewResolution::UndoRegrade => {
            had_undone_review
                && match learner_grade {
                    // A forgotten answer deliberately has no AI evaluation,
                    // so undo cannot turn it into a claimed success. That
                    // requires a fresh instance and checked attempt.
                    Some(Grade::Forgot) => grade == Grade::Forgot,
                    Some(_) | None => true,
                }
        }
        ReviewResolution::LegacyV1 => false,
    };
    if !valid {
        return fail(format!(
            "grade {} is inconsistent with review resolution {} for attempt {attempt_id}",
            grade.as_str(),
            resolution.as_str()
        ));
    }
    Ok(())
}

fn undo_review_and_restore(
    tx: &Transaction<'_>,
    review_id: i64,
    undone_at: Timestamp,
) -> Fallible<()> {
    let target: Option<(SpecHash, i64, Timestamp)> = tx
        .query_row(
            "select instances.spec_hash, instances.session_id, reviews.reviewed_at \
             from reviews \
             join attempts using (attempt_id) \
             join instances using (instance_id) \
             left join review_undos using (review_id) \
             where reviews.review_id = ?1 and review_undos.undo_id is null;",
            params![review_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?;
    let Some((spec_hash, session_id, reviewed_at)) = target else {
        return fail(format!(
            "effective review {review_id} was not found; it may already be undone"
        ));
    };
    if undone_at.into_inner() < reviewed_at.into_inner() {
        return fail(format!(
            "review {review_id} cannot be undone before it was accepted"
        ));
    }

    let latest_review_id: i64 = tx.query_row(
        "select reviews.review_id \
         from reviews \
         join attempts using (attempt_id) \
         join instances using (instance_id) \
         left join review_undos using (review_id) \
         where instances.spec_hash = ?1 and review_undos.undo_id is null \
         order by reviews.review_id desc limit 1;",
        params![spec_hash],
        |row| row.get(0),
    )?;
    if latest_review_id != review_id {
        return fail(format!(
            "review {review_id} is not the latest effective review for drill spec {spec_hash}"
        ));
    }

    tx.execute(
        "insert into review_undos (review_id, undone_at) values (?1, ?2);",
        params![review_id, undone_at],
    )?;

    let previous: Option<ReviewedPerformance> = tx
        .query_row(
            "select reviewed_at, stability, difficulty, interval_raw, interval_days, \
                    due_date, review_count \
             from reviews \
             join attempts using (attempt_id) \
             join instances using (instance_id) \
             where instances.spec_hash = ?1 \
               and not exists (\
                   select 1 from review_undos \
                   where review_undos.review_id = reviews.review_id\
               ) \
             order by reviews.review_id desc limit 1;",
            params![spec_hash],
            |row| {
                Ok(ReviewedPerformance {
                    last_reviewed_at: row.get(0)?,
                    stability: row.get(1)?,
                    difficulty: row.get(2)?,
                    interval_raw: row.get(3)?,
                    interval_days: row.get(4)?,
                    due_date: row.get(5)?,
                    review_count: row.get::<_, i64>(6)? as usize,
                })
            },
        )
        .optional()?;
    write_performance_snapshot(tx, spec_hash, previous)?;

    let reopened = tx.execute(
        "update sessions set ended_at = null where session_id = ?1;",
        params![session_id],
    )?;
    if reopened != 1 {
        return fail(format!(
            "session {session_id} for review {review_id} was not found"
        ));
    }
    Ok(())
}

fn write_performance_snapshot(
    tx: &Transaction<'_>,
    spec_hash: SpecHash,
    performance: Option<ReviewedPerformance>,
) -> Fallible<()> {
    let updated = if let Some(performance) = performance {
        tx.execute(
            "update specs set \
                 last_reviewed_at = ?1, stability = ?2, difficulty = ?3, \
                 interval_raw = ?4, interval_days = ?5, due_date = ?6, \
                 review_count = ?7 \
             where spec_hash = ?8;",
            params![
                performance.last_reviewed_at,
                performance.stability,
                performance.difficulty,
                performance.interval_raw,
                performance.interval_days,
                performance.due_date,
                performance.review_count as i64,
                spec_hash
            ],
        )?
    } else {
        tx.execute(
            "update specs set \
                 last_reviewed_at = null, stability = null, difficulty = null, \
                 interval_raw = null, interval_days = null, due_date = null, \
                 review_count = 0 \
             where spec_hash = ?1;",
            params![spec_hash],
        )?
    };
    if updated != 1 {
        return fail(format!(
            "scheduler state disappeared for drill spec {spec_hash}"
        ));
    }
    Ok(())
}

fn verdict_from_str(value: &str) -> Fallible<Verdict> {
    match value {
        "pass" => Ok(Verdict::Pass),
        "partial" => Ok(Verdict::Partial),
        "fail" => Ok(Verdict::Fail),
        "uncertain" => Ok(Verdict::Uncertain),
        "invalid" => Ok(Verdict::Invalid),
        _ => fail(format!("invalid persisted evaluation verdict: {value}")),
    }
}

fn migrate_v1_to_v2(conn: &mut Connection) -> Fallible<()> {
    // SQLite cannot toggle foreign-key enforcement inside a transaction.
    // Disable it only around the table rebuild, restore it on every path, and
    // run `foreign_key_check` before returning the connection to callers.
    conn.set_db_config(DbConfig::SQLITE_DBCONFIG_ENABLE_FKEY, false)?;
    let migration = (|| -> Fallible<()> {
        let tx = conn.transaction()?;
        let attempts_before = count(&tx, "select count(*) from attempts;", params![])?;
        let reviews_before = count(&tx, "select count(*) from reviews;", params![])?;

        tx.execute_batch(MIGRATE_V1_TO_V2)?;

        let attempts_after = count(&tx, "select count(*) from attempts;", params![])?;
        let evaluations_after = count(&tx, "select count(*) from evaluations;", params![])?;
        let reviews_after = count(&tx, "select count(*) from reviews;", params![])?;
        if attempts_after != attempts_before
            || evaluations_after != attempts_before
            || reviews_after != reviews_before
        {
            return fail(format!(
                "schema migration refused to lose history: attempts {attempts_before}->{attempts_after}, \
                 evaluations 0->{evaluations_after}, reviews {reviews_before}->{reviews_after}"
            ));
        }
        tx.commit()?;
        Ok(())
    })();
    let restore = conn
        .set_db_config(DbConfig::SQLITE_DBCONFIG_ENABLE_FKEY, true)
        .map(|_| ());

    migration?;
    restore?;

    let violation: Option<String> = conn
        .query_row(
            "select \"table\" from pragma_foreign_key_check limit 1;",
            [],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(table) = violation {
        return fail(format!(
            "schema migration produced a foreign-key violation in table {table}"
        ));
    }
    Ok(())
}

fn database_is_empty(conn: &Connection) -> Fallible<bool> {
    let count: i64 = conn.query_row(
        "select count(*) from sqlite_schema \
         where type = 'table' and name not like 'sqlite_%';",
        [],
        |row| row.get(0),
    )?;
    Ok(count == 0)
}

fn validate_schema(conn: &Connection) -> Fallible<()> {
    let version: i64 = conn.query_row("pragma user_version;", [], |row| row.get(0))?;
    if version != SCHEMA_VERSION {
        return fail(format!(
            "unsupported hashdrills database schema version {version}; \
             this build supports version {SCHEMA_VERSION}"
        ));
    }
    for table in [
        "specs",
        "sessions",
        "instances",
        "attempts",
        "evaluations",
        "reviews",
        "review_undos",
    ] {
        let exists: bool = conn.query_row(
            "select exists(\
                 select 1 from sqlite_schema where type = 'table' and name = ?1\
             );",
            params![table],
            |row| row.get(0),
        )?;
        if !exists {
            return fail(format!(
                "hashdrills database schema version {SCHEMA_VERSION} is incomplete: \
                 missing table {table}"
            ));
        }
    }
    Ok(())
}

fn due_query(
    conn: &Connection,
    deck_name: Option<&str>,
    today: Date,
) -> Fallible<HashSet<SpecHash>> {
    let mut due = HashSet::new();
    match deck_name {
        Some(deck_name) => {
            let mut statement = conn.prepare(
                "select spec_hash from specs \
                 where deck_name = ?1 and (due_date is null or due_date <= ?2);",
            )?;
            let rows = statement.query_map(params![deck_name, today], |row| row.get(0))?;
            for row in rows {
                due.insert(row?);
            }
        }
        None => {
            let mut statement = conn.prepare(
                "select spec_hash from specs \
                 where due_date is null or due_date <= ?1;",
            )?;
            let rows = statement.query_map(params![today], |row| row.get(0))?;
            for row in rows {
                due.insert(row?);
            }
        }
    }
    Ok(due)
}

fn apply_queue_limits(
    rows: Vec<DueSpecRow>,
    new_limit: Option<usize>,
    total_limit: Option<usize>,
) -> Vec<DueSpecRow> {
    let mut queue = Vec::new();
    let mut admitted_new = 0usize;
    for row in rows {
        if row.performance.is_new() {
            if new_limit.is_some_and(|limit| admitted_new >= limit) {
                continue;
            }
            admitted_new += 1;
        }
        if total_limit.is_some_and(|limit| queue.len() >= limit) {
            break;
        }
        queue.push(row);
    }
    queue
}

fn sum_current_counts(
    conn: &Connection,
    sql: &str,
    current: &HashSet<SpecHash>,
) -> Fallible<usize> {
    let mut statement = conn.prepare(sql)?;
    let rows = statement.query_map([], |row| {
        let spec_hash: SpecHash = row.get(0)?;
        let count: i64 = row.get(1)?;
        Ok((spec_hash, count as usize))
    })?;
    let mut counts: HashMap<SpecHash, usize> = HashMap::new();
    for row in rows {
        let (spec_hash, count) = row?;
        counts.insert(spec_hash, count);
    }
    Ok(current
        .iter()
        .map(|spec_hash| counts.get(spec_hash).copied().unwrap_or(0))
        .sum())
}

fn count_current_completed_sessions(
    conn: &Connection,
    current: &HashSet<SpecHash>,
) -> Fallible<usize> {
    let mut statement = conn.prepare(
        "select distinct sessions.session_id, instances.spec_hash \
         from sessions \
         join instances using (session_id) \
         where sessions.ended_at is not null;",
    )?;
    let rows = statement.query_map([], |row| {
        let session_id: i64 = row.get(0)?;
        let spec_hash: SpecHash = row.get(1)?;
        Ok((session_id, spec_hash))
    })?;
    let mut sessions = HashSet::new();
    for row in rows {
        let (session_id, spec_hash) = row?;
        if current.contains(&spec_hash) {
            sessions.insert(session_id);
        }
    }
    Ok(sessions.len())
}

fn load_performance(conn: &Connection, spec_hash: SpecHash) -> Fallible<Option<Performance>> {
    let mut stmt = conn.prepare(
        "select last_reviewed_at, stability, difficulty, interval_raw, \
                interval_days, due_date, review_count \
         from specs where spec_hash = ?1;",
    )?;
    let performance = stmt
        .query_row(params![spec_hash], performance_from_row)
        .optional()?;
    Ok(performance)
}

fn performance_from_row(row: &Row<'_>) -> rusqlite::Result<Performance> {
    performance_from_row_at(row, 0)
}

fn performance_from_row_at(row: &Row<'_>, offset: usize) -> rusqlite::Result<Performance> {
    let last_reviewed_at: Option<Timestamp> = row.get(offset)?;
    let stability: Option<Stability> = row.get(offset + 1)?;
    let difficulty: Option<Difficulty> = row.get(offset + 2)?;
    let interval_raw: Option<f64> = row.get(offset + 3)?;
    let interval_days: Option<i64> = row.get(offset + 4)?;
    let due_date: Option<Date> = row.get(offset + 5)?;
    let review_count: i64 = row.get(offset + 6)?;

    match (
        last_reviewed_at,
        stability,
        difficulty,
        interval_raw,
        interval_days,
        due_date,
    ) {
        (None, None, None, None, None, None) if review_count == 0 => Ok(Performance::New),
        (
            Some(last_reviewed_at),
            Some(stability),
            Some(difficulty),
            Some(interval_raw),
            Some(interval_days),
            Some(due_date),
        ) if review_count > 0 => Ok(Performance::Reviewed(ReviewedPerformance {
            last_reviewed_at,
            stability,
            difficulty,
            interval_raw,
            interval_days,
            due_date,
            review_count: review_count as usize,
        })),
        _ => Err(rusqlite::Error::InvalidQuery),
    }
}

fn review_row_from_row(row: &Row<'_>) -> rusqlite::Result<ReviewRow> {
    let reviewed_at: Timestamp = row.get(2)?;
    let resolution: String = row.get(10)?;
    let resolution = review_resolution_from_str(&resolution).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(10, rusqlite::types::Type::Text, Box::new(error))
    })?;
    Ok(ReviewRow {
        review_id: row.get(0)?,
        attempt_id: row.get(1)?,
        reviewed_at,
        grade: row.get(3)?,
        resolution,
        effective: row.get(11)?,
        performance: ReviewedPerformance {
            last_reviewed_at: reviewed_at,
            stability: row.get(4)?,
            difficulty: row.get(5)?,
            interval_raw: row.get(6)?,
            interval_days: row.get(7)?,
            due_date: row.get(8)?,
            review_count: row.get::<_, i64>(9)? as usize,
        },
    })
}

fn review_resolution_from_str(value: &str) -> Fallible<ReviewResolution> {
    match value {
        "learner_forgot" => Ok(ReviewResolution::LearnerForgot),
        "ai_confirmed" => Ok(ReviewResolution::AiConfirmed),
        "ai_rejected" => Ok(ReviewResolution::AiRejected),
        "user_override" => Ok(ReviewResolution::UserOverride),
        "undo_regrade" => Ok(ReviewResolution::UndoRegrade),
        "legacy_v1" => Ok(ReviewResolution::LegacyV1),
        _ => fail(format!("invalid persisted review resolution: {value}")),
    }
}

fn frozen_instance_from_row(row: &Row<'_>) -> rusqlite::Result<FrozenInstanceRow> {
    let protocol_version: i64 = row.get(7)?;
    let protocol_version: u32 = protocol_version.try_into().map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(
            7,
            rusqlite::types::Type::Integer,
            Box::new(error),
        )
    })?;
    Ok(FrozenInstanceRow {
        instance_id: row.get(0)?,
        session_id: row.get(1)?,
        spec_hash: row.get(2)?,
        displayed_question: row.get(3)?,
        target: row.get(4)?,
        rubric: row.get(5)?,
        model: row.get(6)?,
        protocol_version,
        frozen_at: row.get(8)?,
    })
}

fn require_active_session(tx: &Transaction<'_>, session_id: i64) -> Fallible<()> {
    let active: bool = tx.query_row(
        "select exists(\
             select 1 from sessions where session_id = ?1 and ended_at is null\
         );",
        params![session_id],
        |row| row.get(0),
    )?;
    if !active {
        return fail(format!("active session {session_id} was not found"));
    }
    Ok(())
}

fn count<P>(conn: &Connection, sql: &str, params: P) -> Fallible<usize>
where
    P: rusqlite::Params,
{
    let count: i64 = conn.query_row(sql, params, |row| row.get(0))?;
    Ok(count as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::PROTOCOL_VERSION;
    use crate::spec::Template;

    fn memory_storage() -> Fallible<Storage> {
        Storage::from_connection(Connection::open_in_memory()?)
    }

    fn timestamp(value: &str) -> Fallible<Timestamp> {
        Timestamp::try_from(value.to_string())
    }

    fn date(value: &str) -> Fallible<Date> {
        Date::try_from(value.to_string())
    }

    #[test]
    fn initializes_versioned_schema_with_foreign_keys() -> Fallible<()> {
        let storage = memory_storage()?;
        let version: i64 = storage
            .conn
            .query_row("pragma user_version;", [], |row| row.get(0))?;
        let foreign_keys: i64 = storage
            .conn
            .query_row("pragma foreign_keys;", [], |row| row.get(0))?;
        assert_eq!(version, SCHEMA_VERSION);
        assert_eq!(foreign_keys, 1);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn newly_created_database_is_owner_only() -> Fallible<()> {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir()?;
        let path = directory.path().join(DATABASE_FILENAME);
        let _storage = Storage::open(directory.path())?;

        let mode = path.metadata()?.permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn existing_database_permissions_are_preserved() -> Fallible<()> {
        use std::fs::File;
        use std::fs::Permissions;
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir()?;
        let path = directory.path().join(DATABASE_FILENAME);
        let file = File::create(&path)?;
        file.set_permissions(Permissions::from_mode(0o640))?;

        let _storage = Storage::open(directory.path())?;

        let mode = path.metadata()?.permissions().mode() & 0o777;
        assert_eq!(mode, 0o640);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn collection_database_symlink_is_rejected() -> Fallible<()> {
        use std::fs::File;
        use std::os::unix::fs::symlink;

        let collection = tempfile::tempdir()?;
        let outside = tempfile::tempdir()?;
        let target = outside.path().join("outside.db");
        File::create(&target)?;
        symlink(&target, collection.path().join(DATABASE_FILENAME))?;

        let error = Storage::open(collection.path()).err().unwrap();
        assert!(error.to_string().contains("symbolic links are not allowed"));
        assert_eq!(target.metadata()?.len(), 0);
        Ok(())
    }

    #[test]
    fn non_regular_collection_database_is_rejected() -> Fallible<()> {
        let collection = tempfile::tempdir()?;
        std::fs::create_dir(collection.path().join(DATABASE_FILENAME))?;

        let error = Storage::open(collection.path()).err().unwrap();
        assert!(error.to_string().contains("expected a regular file"));
        Ok(())
    }

    #[test]
    fn rejects_unsupported_schema_version() -> Fallible<()> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("pragma user_version = 3;")?;
        let error = Storage::from_connection(conn).err().unwrap();
        assert!(
            error
                .to_string()
                .contains("unsupported hashdrills database schema version 3")
        );
        Ok(())
    }

    #[test]
    fn registration_and_due_queries_include_unseen_specs() -> Fallible<()> {
        let mut storage = memory_storage()?;
        let spec = fixture_spec("Math", "math.md", b"multiplication");
        let at = timestamp("2026-07-31T09:00:00.000")?;
        assert_eq!(storage.register_specs(std::slice::from_ref(&spec), at)?, 1);
        assert_eq!(storage.register_specs(std::slice::from_ref(&spec), at)?, 0);

        assert_eq!(storage.performance(spec.hash())?, Performance::New);
        assert!(storage.all_due(date("2026-07-31")?)?.contains(&spec.hash()));
        assert!(
            storage
                .due_for_deck("Math", date("2026-07-31")?)?
                .contains(&spec.hash())
        );
        assert!(
            storage
                .due_for_deck("Other", date("2026-07-31")?)?
                .is_empty()
        );
        assert!(storage.due_counts_by_deck(date("2026-07-31")?)?.is_empty());

        let mut moved = spec.clone();
        moved.deck_name = "Arithmetic".to_string();
        moved.path = "/new/location/math.md".into();
        moved.source = Some("updated provenance".to_string());
        moved.range = (10, 12);
        assert_eq!(storage.register_specs(&[moved], at)?, 0);
        assert!(
            storage
                .due_for_deck("Math", date("2026-07-31")?)?
                .is_empty()
        );
        assert!(
            storage
                .due_for_deck("Arithmetic", date("2026-07-31")?)?
                .contains(&spec.hash())
        );
        let metadata: (String, String, Option<String>, i64, i64) = storage.conn.query_row(
            "select deck_name, source_file, source_provenance, line_start, line_end \
             from specs where spec_hash = ?1;",
            params![spec.hash()],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        )?;
        assert_eq!(
            metadata,
            (
                "Arithmetic".to_string(),
                "/new/location/math.md".to_string(),
                Some("updated provenance".to_string()),
                10,
                12
            )
        );
        Ok(())
    }

    #[test]
    fn frozen_instances_are_exact_and_immutable() -> Fallible<()> {
        let mut storage = memory_storage()?;
        let spec = fixture_spec("Math", "math.md", b"instance");
        let at = timestamp("2026-07-31T09:00:00.000")?;
        storage.register_specs(std::slice::from_ref(&spec), at)?;
        let session_id = storage.begin_session(at)?;
        let generated = fixture_instance();
        let instance_id = storage.freeze_instance(session_id, spec.hash(), &generated, at)?;

        let frozen = storage.frozen_instance(instance_id)?;
        assert_eq!(frozen.displayed_question, generated.question);
        assert_eq!(frozen.target, generated.target);
        assert_eq!(frozen.rubric, generated.rubric);
        assert_eq!(frozen.model, generated.model);
        assert_eq!(frozen.protocol_version, generated.protocol_version);

        let update = storage.conn.execute(
            "update instances set displayed_question = 'changed' where instance_id = ?1;",
            params![instance_id],
        );
        assert!(update.is_err());
        assert_eq!(
            storage.frozen_instance(instance_id)?.displayed_question,
            generated.question
        );
        Ok(())
    }

    #[test]
    fn answer_evaluation_and_schedule_update_are_distinct_durable_steps() -> Fallible<()> {
        let mut storage = memory_storage()?;
        let spec = fixture_spec("Math", "math.md", b"graded");
        let at = timestamp("2026-07-31T09:00:00.000")?;
        storage.register_specs(std::slice::from_ref(&spec), at)?;
        let session_id = storage.begin_session(at)?;
        let generated = fixture_instance();
        let instance_id = storage.freeze_instance(session_id, spec.hash(), &generated, at)?;
        let evaluation = fixture_evaluation(Verdict::Pass);

        let attempt = storage.record_answer(instance_id, "12", Grade::Good, at)?;
        let stored_attempt = storage.attempt(attempt.attempt_id)?;
        assert_eq!(stored_attempt.user_response, "12");
        assert_eq!(stored_attempt.learner_grade, Some(Grade::Good));
        assert!(
            storage
                .evaluation_for_attempt(attempt.attempt_id)?
                .is_none()
        );
        assert_eq!(storage.performance(spec.hash())?, Performance::New);

        storage.record_evaluation(attempt.attempt_id, &evaluation, at)?;
        let stored_evaluation = storage.evaluation_for_attempt(attempt.attempt_id)?.unwrap();
        assert!(stored_evaluation.evidence.is_empty());
        assert_eq!(stored_evaluation.evaluator_model, evaluation.model);
        assert!(
            storage
                .record_evaluation(attempt.attempt_id, &evaluation, at)
                .is_err()
        );

        let recorded = storage.accept_grade(
            attempt.attempt_id,
            Grade::Good,
            ReviewResolution::AiConfirmed,
            at,
        )?;
        let scheduled = recorded.performance;
        assert_eq!(
            storage
                .review_for_attempt(attempt.attempt_id)?
                .unwrap()
                .performance,
            scheduled
        );
        assert_eq!(
            storage.review(recorded.review_id)?.resolution,
            ReviewResolution::AiConfirmed
        );
        assert_eq!(
            storage.performance(spec.hash())?,
            Performance::Reviewed(scheduled)
        );
        assert_eq!(storage.stats(at.date())?.attempts, 1);
        assert_eq!(storage.stats(at.date())?.reviews, 1);
        assert!(!storage.all_due(at.date())?.contains(&spec.hash()));
        assert!(storage.all_due(scheduled.due_date)?.contains(&spec.hash()));

        // The same frozen instance cannot acquire a second first answer, and
        // the rejected transaction cannot advance FSRS a second time.
        assert!(
            storage
                .record_answer(instance_id, "a replacement", Grade::Easy, at)
                .is_err()
        );
        assert_eq!(
            storage.performance(spec.hash())?,
            Performance::Reviewed(scheduled)
        );
        assert_eq!(storage.stats(at.date())?.attempts, 1);
        Ok(())
    }

    #[test]
    fn completion_history_excludes_undone_and_orphaned_reviews() -> Fallible<()> {
        let mut storage = memory_storage()?;
        let active = fixture_spec("Math", "active.md", b"active-history");
        let orphaned = fixture_spec("Math", "orphaned.md", b"orphaned-history");
        let at = timestamp("2026-07-31T09:00:00.000")?;
        storage.register_specs(&[active.clone(), orphaned.clone()], at)?;
        let session_id = storage.begin_session(at)?;

        let active_instance =
            storage.freeze_instance(session_id, active.hash(), &fixture_instance(), at)?;
        let active_attempt = storage.record_answer(active_instance, "12", Grade::Good, at)?;
        storage.record_evaluation(
            active_attempt.attempt_id,
            &fixture_evaluation(Verdict::Pass),
            at,
        )?;
        let original = storage.accept_grade(
            active_attempt.attempt_id,
            Grade::Good,
            ReviewResolution::AiConfirmed,
            at,
        )?;
        storage.undo_review(original.review_id, at)?;
        storage.accept_grade(
            active_attempt.attempt_id,
            Grade::Hard,
            ReviewResolution::UndoRegrade,
            at,
        )?;

        let orphaned_instance =
            storage.freeze_instance(session_id, orphaned.hash(), &fixture_instance(), at)?;
        let orphaned_attempt = storage.record_answer(orphaned_instance, "12", Grade::Easy, at)?;
        storage.record_evaluation(
            orphaned_attempt.attempt_id,
            &fixture_evaluation(Verdict::Pass),
            at,
        )?;
        storage.accept_grade(
            orphaned_attempt.attempt_id,
            Grade::Easy,
            ReviewResolution::AiConfirmed,
            at,
        )?;

        let history = storage.completion_history_for_specs(at.date(), &[active])?;
        assert_eq!(history.stats.total_specs, 1);
        assert_eq!(history.stats.unseen_specs, 0);
        assert_eq!(
            history.activity,
            vec![ReviewActivityDay {
                date: at.date(),
                reviews: 1,
            }]
        );
        assert_eq!(history.recent_grades.forgot, 0);
        assert_eq!(history.recent_grades.hard, 1);
        assert_eq!(history.recent_grades.good, 0);
        assert_eq!(history.recent_grades.easy, 0);
        assert_eq!(history.horizons.under_7_days, 1);
        assert_eq!(history.horizons.days_7_to_29, 0);
        assert_eq!(history.horizons.days_30_to_89, 0);
        assert_eq!(history.horizons.days_90_plus, 0);
        Ok(())
    }

    #[test]
    fn forgot_skips_ai_and_can_update_schedule_directly() -> Fallible<()> {
        let mut storage = memory_storage()?;
        let spec = fixture_spec("Math", "math.md", b"forgot");
        let at = timestamp("2026-07-31T09:00:00.000")?;
        storage.register_specs(std::slice::from_ref(&spec), at)?;
        let session_id = storage.begin_session(at)?;
        let generated = fixture_instance();
        let instance_id = storage.freeze_instance(session_id, spec.hash(), &generated, at)?;
        let evaluation = fixture_evaluation(Verdict::Pass);

        let attempt = storage.record_answer(instance_id, "", Grade::Forgot, at)?;
        assert!(
            storage
                .record_evaluation(attempt.attempt_id, &evaluation, at)
                .is_err()
        );
        assert_eq!(storage.performance(spec.hash())?, Performance::New);
        assert_eq!(storage.stats(at.date())?.attempts, 1);
        assert_eq!(storage.stats(at.date())?.reviews, 0);

        let review = storage.accept_grade(
            attempt.attempt_id,
            Grade::Forgot,
            ReviewResolution::LearnerForgot,
            at,
        )?;
        assert_eq!(
            storage.review(review.review_id)?.resolution,
            ReviewResolution::LearnerForgot
        );
        assert!(
            storage
                .evaluation_for_attempt(attempt.attempt_id)?
                .is_none()
        );
        assert_eq!(storage.stats(at.date())?.reviews, 1);

        storage.undo_review(review.review_id, at)?;
        assert!(
            storage
                .accept_grade(
                    attempt.attempt_id,
                    Grade::Good,
                    ReviewResolution::UndoRegrade,
                    at,
                )
                .is_err()
        );
        storage.accept_grade(
            attempt.attempt_id,
            Grade::Forgot,
            ReviewResolution::UndoRegrade,
            at,
        )?;
        Ok(())
    }

    #[test]
    fn uncertain_or_invalid_ai_results_never_schedule() -> Fallible<()> {
        for verdict in [Verdict::Uncertain, Verdict::Invalid] {
            let mut storage = memory_storage()?;
            let identity = format!("abstain-{}", verdict.as_str());
            let spec = fixture_spec("Math", "math.md", identity.as_bytes());
            let at = timestamp("2026-07-31T09:00:00.000")?;
            storage.register_specs(std::slice::from_ref(&spec), at)?;
            let session_id = storage.begin_session(at)?;
            let instance_id =
                storage.freeze_instance(session_id, spec.hash(), &fixture_instance(), at)?;
            let attempt = storage.record_answer(instance_id, "12", Grade::Good, at)?;
            storage.record_evaluation(attempt.attempt_id, &fixture_evaluation(verdict), at)?;

            for resolution in [
                ReviewResolution::AiConfirmed,
                ReviewResolution::AiRejected,
                ReviewResolution::UserOverride,
            ] {
                let accepted_grade = if resolution == ReviewResolution::AiRejected {
                    Grade::Forgot
                } else {
                    Grade::Good
                };
                assert!(
                    storage
                        .accept_grade(attempt.attempt_id, accepted_grade, resolution, at)
                        .is_err()
                );
            }
            assert_eq!(storage.performance(spec.hash())?, Performance::New);
            assert!(storage.review_for_attempt(attempt.attempt_id)?.is_none());
        }
        Ok(())
    }

    #[test]
    fn undo_preserves_audit_restores_new_and_allows_regrade() -> Fallible<()> {
        let mut storage = memory_storage()?;
        let spec = fixture_spec("Math", "math.md", b"undo");
        let at = timestamp("2026-07-31T09:00:00.000")?;
        let later = timestamp("2026-07-31T09:01:00.000")?;
        storage.register_specs(std::slice::from_ref(&spec), at)?;
        let session_id = storage.begin_session(at)?;
        let instance_id =
            storage.freeze_instance(session_id, spec.hash(), &fixture_instance(), at)?;
        let attempt = storage.record_answer(instance_id, "12", Grade::Good, at)?;
        storage.record_evaluation(attempt.attempt_id, &fixture_evaluation(Verdict::Pass), at)?;
        let first = storage.accept_grade(
            attempt.attempt_id,
            Grade::Good,
            ReviewResolution::AiConfirmed,
            at,
        )?;
        storage.finish_session(session_id, later)?;

        storage.undo_review(first.review_id, later)?;
        assert_eq!(storage.performance(spec.hash())?, Performance::New);
        assert!(!storage.review(first.review_id)?.effective);
        assert!(storage.review_for_attempt(attempt.attempt_id)?.is_none());
        let ended_at: Option<Timestamp> = storage.conn.query_row(
            "select ended_at from sessions where session_id = ?1;",
            params![session_id],
            |row| row.get(0),
        )?;
        assert!(ended_at.is_none());

        let replacement = storage.accept_grade(
            attempt.attempt_id,
            Grade::Hard,
            ReviewResolution::UndoRegrade,
            later,
        )?;
        assert_ne!(replacement.review_id, first.review_id);
        assert!(storage.review(replacement.review_id)?.effective);
        assert_eq!(
            storage.performance(spec.hash())?,
            Performance::Reviewed(replacement.performance)
        );
        assert_eq!(storage.stats(at.date())?.reviews, 2);
        storage.finish_session(session_id, later)?;
        Ok(())
    }

    #[test]
    fn undo_restores_the_previous_effective_snapshot() -> Fallible<()> {
        let mut storage = memory_storage()?;
        let spec = fixture_spec("Math", "math.md", b"undo-chain");
        let first_at = timestamp("2026-07-28T09:00:00.000")?;
        let second_at = timestamp("2026-07-31T09:00:00.000")?;
        storage.register_specs(std::slice::from_ref(&spec), first_at)?;
        let session_id = storage.begin_session(first_at)?;

        let first_instance =
            storage.freeze_instance(session_id, spec.hash(), &fixture_instance(), first_at)?;
        let first_attempt = storage.record_answer(first_instance, "12", Grade::Good, first_at)?;
        storage.record_evaluation(
            first_attempt.attempt_id,
            &fixture_evaluation(Verdict::Pass),
            first_at,
        )?;
        let first_review = storage.accept_grade(
            first_attempt.attempt_id,
            Grade::Good,
            ReviewResolution::AiConfirmed,
            first_at,
        )?;

        let second_instance =
            storage.freeze_instance(session_id, spec.hash(), &fixture_instance(), second_at)?;
        let second_attempt =
            storage.record_answer(second_instance, "12", Grade::Hard, second_at)?;
        storage.record_evaluation(
            second_attempt.attempt_id,
            &fixture_evaluation(Verdict::Pass),
            second_at,
        )?;
        let second_review = storage.accept_grade(
            second_attempt.attempt_id,
            Grade::Hard,
            ReviewResolution::AiConfirmed,
            second_at,
        )?;
        assert_ne!(second_review.performance, first_review.performance);

        storage.undo_review(second_review.review_id, second_at)?;
        assert_eq!(
            storage.performance(spec.hash())?,
            Performance::Reviewed(first_review.performance)
        );
        assert!(!storage.review(second_review.review_id)?.effective);
        assert!(storage.review(first_review.review_id)?.effective);
        Ok(())
    }

    #[test]
    fn migrates_v1_without_inventing_or_losing_attempt_evidence() -> Fallible<()> {
        let mut storage = memory_storage()?;
        let spec = fixture_spec("Math", "math.md", b"migration");
        let at = timestamp("2026-07-31T09:00:00.000")?;
        storage.register_specs(std::slice::from_ref(&spec), at)?;
        let session_id = storage.begin_session(at)?;
        let reviewed_instance =
            storage.freeze_instance(session_id, spec.hash(), &fixture_instance(), at)?;
        let unreviewed_instance =
            storage.freeze_instance(session_id, spec.hash(), &fixture_instance(), at)?;

        storage.conn.execute_batch(
            "drop table review_undos; \
             drop table reviews; \
             drop table evaluations; \
             drop table attempts; \
             create table attempts (\
                 attempt_id integer primary key, \
                 instance_id integer not null unique references instances (instance_id), \
                 answered_at text not null, \
                 user_response text not null, \
                 verdict text not null check (\
                     verdict in ('pass', 'partial', 'fail', 'uncertain', 'invalid')\
                 ), \
                 feedback text not null, \
                 evidence_json text not null check (json_valid(evidence_json)), \
                 evaluator_model text not null, \
                 evaluator_protocol_version integer not null check (\
                     evaluator_protocol_version > 0\
                 )\
             ) strict; \
             create table reviews (\
                 review_id integer primary key, \
                 attempt_id integer not null unique references attempts (attempt_id), \
                 reviewed_at text not null, \
                 grade text not null check (grade in ('forgot', 'hard', 'good', 'easy')), \
                 stability real not null, \
                 difficulty real not null, \
                 interval_raw real not null, \
                 interval_days integer not null, \
                 due_date text not null, \
                 review_count integer not null check (review_count > 0)\
             ) strict; \
             pragma user_version = 1;",
        )?;

        let performance = update_performance(Performance::New, Grade::Forgot, at);
        storage.conn.execute(
            "insert into attempts (\
                 attempt_id, instance_id, answered_at, user_response, verdict, feedback, \
                 evidence_json, evaluator_model, evaluator_protocol_version\
             ) values (41, ?1, ?2, 'wrong', 'fail', 'legacy feedback', \
                       '[\"legacy evidence\"]', 'legacy-model', 1);",
            params![reviewed_instance, at],
        )?;
        storage.conn.execute(
            "insert into attempts (\
                 attempt_id, instance_id, answered_at, user_response, verdict, feedback, \
                 evidence_json, evaluator_model, evaluator_protocol_version\
             ) values (42, ?1, ?2, 'unknown', 'invalid', 'legacy invalid', \
                       '[\"legacy invalid evidence\"]', 'legacy-model', 1);",
            params![unreviewed_instance, at],
        )?;
        storage.conn.execute(
            "insert into reviews (\
                 review_id, attempt_id, reviewed_at, grade, stability, difficulty, \
                 interval_raw, interval_days, due_date, review_count\
             ) values (51, 41, ?1, 'forgot', ?2, ?3, ?4, ?5, ?6, ?7);",
            params![
                at,
                performance.stability,
                performance.difficulty,
                performance.interval_raw,
                performance.interval_days,
                performance.due_date,
                performance.review_count as i64
            ],
        )?;
        storage.conn.execute(
            "update specs set \
                 last_reviewed_at = ?1, stability = ?2, difficulty = ?3, \
                 interval_raw = ?4, interval_days = ?5, due_date = ?6, review_count = ?7 \
             where spec_hash = ?8;",
            params![
                performance.last_reviewed_at,
                performance.stability,
                performance.difficulty,
                performance.interval_raw,
                performance.interval_days,
                performance.due_date,
                performance.review_count as i64,
                spec.hash()
            ],
        )?;

        let Storage { conn } = storage;
        let migrated = Storage::from_connection(conn)?;
        let version: i64 = migrated
            .conn
            .query_row("pragma user_version;", [], |row| row.get(0))?;
        assert_eq!(version, 2);
        assert_eq!(migrated.attempt(41)?.learner_grade, Some(Grade::Forgot));
        assert_eq!(migrated.attempt(42)?.learner_grade, None);
        let evaluation = migrated.evaluation_for_attempt(41)?.unwrap();
        assert_eq!(evaluation.verdict, Verdict::Fail);
        assert_eq!(evaluation.feedback, "legacy feedback");
        assert_eq!(evaluation.evidence, vec!["legacy evidence"]);
        let review = migrated.review(51)?;
        assert_eq!(review.resolution, ReviewResolution::LegacyV1);
        assert!(review.effective);
        assert_eq!(migrated.stats(at.date())?.attempts, 2);
        assert_eq!(migrated.stats(at.date())?.reviews, 1);
        Ok(())
    }

    // These fixtures intentionally use the domain constructors so the tests
    // exercise the same boundary as the command/runtime layers.
    fn fixture_spec(deck_name: &str, source: &str, identity: &[u8]) -> DrillSpec {
        let identity = String::from_utf8_lossy(identity);
        DrillSpec::new(
            deck_name,
            Some("fixture provenance".to_string()),
            source.into(),
            (1, 3),
            Some(format!("Practice {identity}")),
            Template::parse(&format!("Question for {identity}?"))
                .expect("fixture question is valid"),
            Template::parse(&format!("Answer for {identity}.")).expect("fixture answer is valid"),
        )
    }

    fn fixture_instance() -> GeneratedInstance {
        GeneratedInstance {
            question: "What is 3 x 4?".to_string(),
            target: "12".to_string(),
            rubric: "The response is exactly the integer 12.".to_string(),
            model: "test-model".to_string(),
            protocol_version: PROTOCOL_VERSION,
        }
    }

    fn fixture_evaluation(verdict: Verdict) -> Evaluation {
        Evaluation {
            verdict,
            feedback: "Specific feedback".to_string(),
            model: "test-evaluator".to_string(),
            protocol_version: PROTOCOL_VERSION,
        }
    }
}
