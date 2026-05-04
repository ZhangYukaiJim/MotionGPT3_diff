## ADDED Requirements

### Requirement: Prototype script can resolve a MotionFix motion pair from cache-backed inputs
The system SHALL provide a standalone prototype workflow for MotionFix `m2t_diff` qualitative inspection that can load one paired-motion sample either from explicit source and target feature files or from a MotionFix sample id resolved through the existing cache manifest layout.

#### Scenario: Load a MotionFix pair by sample id
- **WHEN** a developer runs the prototype script with a MotionFix-aware config and a sample id that exists in the cached MotionFix manifests
- **THEN** the script loads the corresponding source and target feature arrays, normalizes them with the existing MotionFix datamodule conventions, and recovers root-centered joint sequences shaped `[T, 22, 3]` for both motions

#### Scenario: Load a MotionFix pair from direct feature files
- **WHEN** a developer runs the prototype script with explicit source and target `.npy` feature paths
- **THEN** the script loads those feature arrays, applies the same MotionFix normalization assumptions used by the cache-backed path, and recovers root-centered joint sequences shaped `[T, 22, 3]` for both motions without requiring a sample export directory

### Requirement: Prototype script exports explicit temporal-alignment validation artifacts
The system SHALL export visual validation artifacts for constrained-DTW alignment before relying on the alignment for mesh heat rendering.

#### Scenario: Export alignment diagnostics for one motion pair
- **WHEN** the prototype script computes constrained DTW over root-centered joint features for one motion pair
- **THEN** it writes an output directory containing at least a frame-cost matrix image with the DTW path overlaid, a path-index plot, a scalar trace plot, a serialized alignment-path file, and an aligned side-by-side preview video derived from matched frame pairs

#### Scenario: Alignment path remains monotonic and endpoint aligned
- **WHEN** constrained DTW completes successfully for one source sequence and one target sequence
- **THEN** the exported alignment path is a monotonic sequence of source-target frame index pairs that starts at the first frame of both motions and ends at the last frame of both motions

### Requirement: Prototype script exports a source-focused vertex heat-overlay mesh video
The system SHALL export a vertex-level heat-overlay video for MotionFix `m2t_diff` qualitative inspection in which the source motion is rendered with per-vertex heat values derived from aligned frame pairs and the target motion is rendered as a translucent ghost reference.

#### Scenario: Render a vertex heat overlay from aligned SMPL meshes
- **WHEN** the prototype script has recovered aligned root-centered frame pairs for one MotionFix sample
- **THEN** it converts both aligned motions into same-topology SMPL meshes, computes one scalar heat value per source vertex for each aligned frame pair, and writes a video in which the source mesh color varies with that heat value while the target mesh remains a translucent ghost

#### Scenario: Heat overlay uses robust clipping and smoothing
- **WHEN** the prototype script computes per-vertex displacement magnitudes across aligned frame pairs
- **THEN** it applies temporal smoothing and robust clipping before color mapping so the rendered heat video is not dominated by single-frame spikes or one extreme vertex outlier
