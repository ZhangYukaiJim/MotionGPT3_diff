## ADDED Requirements

### Requirement: Paired-motion difference captioning uses text-generation metrics
The system SHALL provide a quantitative evaluation path for `m2t_diff` that scores generated difference captions with text-generation metrics rather than reusing single-motion retrieval metrics unchanged.

#### Scenario: Evaluate `m2t_diff` captions with text metrics
- **WHEN** a config-driven validation or test run evaluates `m2t_diff` predictions against reference captions
- **THEN** the system reports text-generation metrics that are directly defined over predicted and reference text, including `Bert_F1`, `ROUGE_L`, and `CIDEr`

#### Scenario: Optional BLEU scores remain secondary
- **WHEN** `m2t_diff` text metrics include BLEU-style overlap scores
- **THEN** the system treats them as supplementary text metrics and does not require them as the only success signal for paired-motion evaluation

### Requirement: Paired-motion difference captioning reports decoding diagnostics
The system SHALL report decoding diagnostics for `m2t_diff` that help detect degenerate outputs such as blank captions or excessively short generations.

#### Scenario: Blank-caption failures are surfaced explicitly
- **WHEN** an `m2t_diff` evaluation run produces empty or whitespace-only predictions
- **THEN** the system reports an empty-output rate so failures are visible in validation and test summaries

#### Scenario: Caption length trends are visible
- **WHEN** an `m2t_diff` evaluation run completes
- **THEN** the system reports average generated caption length so unusually short or collapsed outputs can be detected without manually reading every sample

### Requirement: Single-motion retrieval metrics are not silently reused for paired-motion captions
The system SHALL avoid presenting single-motion motion-text retrieval metrics as if they were valid paired-motion comparison metrics for `m2t_diff` unless a pair-aware evaluator is explicitly implemented.

#### Scenario: Existing single-motion retrieval metrics are disabled for `m2t_diff`
- **WHEN** a developer configures evaluation for `m2t_diff`
- **THEN** the system does not silently compute or headline single-motion matching-score or R-precision metrics as the primary paired-motion evaluation result

#### Scenario: Future pair-aware retrieval stays separate
- **WHEN** a team wants a stronger motion-text alignment metric for `m2t_diff`
- **THEN** the system treats that work as a separate pair-aware metric capability rather than conflating it with existing single-motion evaluators
