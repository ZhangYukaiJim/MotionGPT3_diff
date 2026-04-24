## MODIFIED Requirements

### Requirement: Two-motion evaluation can run through existing non-web entry points
The system SHALL support evaluating or smoke-testing the two-motion text task through config-driven scripts and helper entry points before any Gradio integration is required, including optional qualitative artifact export for `m2t_diff` test runs.

#### Scenario: Run qualitative paired-motion generation from scripts
- **WHEN** a developer runs a dedicated two-motion helper script with two `.npy` feature files and a config path
- **THEN** the system returns generated text using the configured checkpoint and the same motion-token pipeline used in finetuning

#### Scenario: Training artifacts remain in the standard experiment layout
- **WHEN** the two-motion stage-4 finetuning task is trained
- **THEN** logs, configs, and checkpoints are written under the existing `experiments/<model>/<NAME>/` layout used by other MotionGPT3 training runs

#### Scenario: `m2t_diff` reports paired-motion-appropriate quantitative metrics
- **WHEN** a developer runs a config-driven validation or test workflow for `m2t_diff`
- **THEN** the system can report text-generation metrics and decoding diagnostics that are appropriate for paired-motion difference captions without requiring web UI tooling

#### Scenario: Existing single-motion retrieval metrics are not treated as pair-aware evaluation
- **WHEN** a developer reviews quantitative `m2t_diff` results from the standard evaluation entry points
- **THEN** the reported primary metrics exclude single-motion retrieval scores unless a dedicated pair-aware retrieval evaluator has been added

#### Scenario: `m2t_diff` test runs can export qualitative paired-motion artifacts
- **WHEN** a developer enables qualitative export for a config-driven `m2t_diff` test run
- **THEN** the standard non-web evaluation entry point can save paired-motion videos and caption files for the exported test samples alongside quantitative results
