## Purpose

Define config-driven qualitative test export for MotionFix `m2t_diff` evaluation.

## Requirements

### Requirement: `m2t_diff` test runs can materialize paired-motion qualitative artifacts
The system SHALL provide a config-driven `test.py` path for `m2t_diff` that can save qualitative artifacts for test samples, including source motion video, target motion video, side-by-side source-target video, predicted text, and ground-truth text.

#### Scenario: Visualize all `m2t_diff` test samples
- **WHEN** a MotionFix `m2t_diff` test run enables test visualization and iterates over the configured test split
- **THEN** the run writes one source video, one target video, one side-by-side video, one predicted-text file, and one ground-truth-text file per exported sample under the standard test sample output directory

#### Scenario: Visualization stays off unless requested
- **WHEN** a MotionFix `m2t_diff` test run does not enable test visualization
- **THEN** the system does not materialize paired-motion videos solely as a side effect of running `test.py`

### Requirement: `m2t_diff` test visualization reuses MotionFix render cache helpers
The system SHALL materialize `m2t_diff` test videos through the existing MotionFix render-cache helpers so repeated test exports can reuse cached source, target, and side-by-side renders.

#### Scenario: Cached source and target renders are reused
- **WHEN** an exported `m2t_diff` test sample refers to a MotionFix pair whose cached source or target render already exists in the MotionFix render cache
- **THEN** the test visualization path reuses the cached render artifact instead of rerendering that motion from scratch

#### Scenario: Cached side-by-side render is reused
- **WHEN** an exported `m2t_diff` test sample refers to a MotionFix pair whose cached source-target comparison render already exists
- **THEN** the test visualization path reuses that cached side-by-side artifact for the sample output

### Requirement: `m2t_diff` test artifact names are stable and sample-aligned
The system SHALL save `m2t_diff` test visualization artifacts with stable per-sample naming based on the sample id so videos and text files for the same pair can be matched without reading additional metadata.

#### Scenario: Test artifact filenames identify the sample and artifact role
- **WHEN** a MotionFix `m2t_diff` test sample with id `<id>` is exported
- **THEN** the output directory contains filenames derived from `<id>` that distinguish source motion, target motion, side-by-side comparison, predicted text, and ground-truth text
