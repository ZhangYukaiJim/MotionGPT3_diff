## MODIFIED Requirements

### Requirement: Two-motion evaluation can run through existing non-web entry points
The system SHALL support evaluating or smoke-testing the two-motion text task through config-driven scripts and helper entry points before any Gradio integration is required.

#### Scenario: `m2t_diff` reports paired-motion-appropriate quantitative metrics
- **WHEN** a developer runs a config-driven validation or test workflow for `m2t_diff`
- **THEN** the system can report text-generation metrics and decoding diagnostics that are appropriate for paired-motion difference captions without requiring web UI tooling

#### Scenario: Existing single-motion retrieval metrics are not treated as pair-aware evaluation
- **WHEN** a developer reviews quantitative `m2t_diff` results from the standard evaluation entry points
- **THEN** the reported primary metrics exclude single-motion retrieval scores unless a dedicated pair-aware retrieval evaluator has been added
