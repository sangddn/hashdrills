create table specs (
    spec_hash text primary key,
    deck_name text not null,
    goal text,
    question_template text not null,
    answer_template text not null,
    source_file text not null,
    source_provenance text,
    line_start integer not null,
    line_end integer not null,
    registered_at text not null,
    last_reviewed_at text,
    stability real,
    difficulty real,
    interval_raw real,
    interval_days integer,
    due_date text,
    review_count integer not null default 0,
    check (line_start > 0 and line_end >= line_start),
    check (
        (
            review_count = 0
            and last_reviewed_at is null
            and stability is null
            and difficulty is null
            and interval_raw is null
            and interval_days is null
            and due_date is null
        )
        or
        (
            review_count > 0
            and last_reviewed_at is not null
            and stability is not null
            and difficulty is not null
            and interval_raw is not null
            and interval_days is not null
            and due_date is not null
        )
    )
) strict;

create index specs_due_date_idx on specs (due_date);
create index specs_deck_due_date_idx on specs (deck_name, due_date);

-- The authored G/Q/A contract is content-addressed. Discovery metadata and
-- scheduler state may evolve, but the contract beneath one hash may not.
create trigger spec_contract_is_immutable
before update of spec_hash, goal, question_template, answer_template, registered_at
on specs
begin
    select raise(abort, 'a registered drill contract is immutable');
end;

create table sessions (
    session_id integer primary key,
    started_at text not null,
    ended_at text,
    check (ended_at is null or ended_at >= started_at)
) strict;

create table instances (
    instance_id integer primary key,
    session_id integer not null
        references sessions (session_id),
    spec_hash text not null
        references specs (spec_hash),
    displayed_question text not null,
    target text not null,
    rubric text not null,
    model text not null,
    protocol_version integer not null check (protocol_version > 0),
    frozen_at text not null
) strict;

create index instances_spec_hash_idx on instances (spec_hash);
create index instances_session_id_idx on instances (session_id);

-- A generated instance is the exact problem the learner saw.  Once frozen,
-- neither application code nor a cascading parent operation may rewrite it.
create trigger instances_are_immutable_update
before update on instances
begin
    select raise(abort, 'generated instances are immutable');
end;

create trigger instances_are_immutable_delete
before delete on instances
begin
    select raise(abort, 'generated instances are immutable');
end;

create table attempts (
    attempt_id integer primary key,
    instance_id integer not null unique
        references instances (instance_id),
    answered_at text not null,
    user_response text not null,
    learner_grade text check (
        learner_grade is null
        or learner_grade in ('forgot', 'hard', 'good', 'easy')
    )
) strict;

create index attempts_answered_at_idx on attempts (answered_at);

-- An attempt is the first submitted response and the learner's rating of
-- retrieval difficulty. `learner_grade` is nullable only so v1 attempts that
-- never reached grading can be migrated without inventing evidence. New
-- application writes always provide it.
create trigger attempts_require_learner_grade
before insert on attempts
when new.learner_grade is null
begin
    select raise(abort, 'new attempts require a learner grade');
end;

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
    attempt_id integer not null unique
        references attempts (attempt_id),
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

-- Claiming Forgot is complete evidence by itself: no model call or result may
-- be attached. Other learner grades may receive exactly one evaluation.
create trigger evaluations_reject_forgotten_attempts
before insert on evaluations
when (
    select learner_grade from attempts where attempt_id = new.attempt_id
) = 'forgot'
begin
    select raise(abort, 'forgotten attempts must not have an AI evaluation');
end;

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
    attempt_id integer not null
        references attempts (attempt_id),
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

-- Accepted reviews are immutable evidence. An undo is represented by a
-- separate append-only row, so replacing a grade never erases history.
create table review_undos (
    undo_id integer primary key,
    review_id integer not null unique
        references reviews (review_id),
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

-- There may be many historical reviews for an attempt, but at most one that
-- has not subsequently been undone.
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

pragma user_version = 2;
-- Derived from Hashcards (Copyright 2025–2026 Fernando Borretti).
-- Modified in 2026 for Hashdrills by Sang Doan.
