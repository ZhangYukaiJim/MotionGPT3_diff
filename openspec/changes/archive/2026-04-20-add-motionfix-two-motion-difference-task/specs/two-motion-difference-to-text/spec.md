## ADDED Requirements

### Requirement: Two-motion-to-text task accepts paired motion inputs
The system SHALL support a text-generation task that consumes two ordered motion inputs, where the first motion is treated as the source motion and the second motion is treated as the target motion.

#### Scenario: Encode paired motions for text generation
- **WHEN** a paired-motion batch provides normalized source and target motion feature tensors plus their lengths
- **THEN** the model encodes source and target motions separately with the existing MotionGPT3 motion-token pipeline and preserves their order in the combined LM input

#### Scenario: Different motion lengths are supported
- **WHEN** source and target motions in the same sample have different frame lengths
- **THEN** the task path uses the provided source and target lengths without requiring the two motions to share the same number of frames

### Requirement: Two-motion prompts identify both motion roles
The system SHALL provide prompt templates and placeholder handling for two-motion text generation that distinguish source motion content from target motion content.

#### Scenario: Comparison prompt uses both motion placeholders
- **WHEN** a two-motion text generation helper constructs a comparison prompt
- **THEN** the prompt can reference both source and target motion placeholders in a stable order suitable for qualitative prompting and finetuning supervision

#### Scenario: Prompt-only debugging is available outside the web UI
- **WHEN** a developer wants to inspect a pair of local motion feature files without launching Gradio
- **THEN** the system provides a scriptable path to generate text from those two motions using the released or finetuned checkpoint

### Requirement: Stage-4 finetuning remains compatible with the released checkpoint
The system SHALL provide a stage-4 finetuning configuration for the two-motion-to-text task that can load `checkpoints/motiongpt3.ckpt` without dropping checkpoint parameters caused by classifier-free-guidance structure mismatches.

#### Scenario: Load released checkpoint for finetuning
- **WHEN** stage-4 finetuning is started from `checkpoints/motiongpt3.ckpt`
- **THEN** the instantiated model includes the guidance-related structure required to load the checkpoint state dict without rejecting `lm.fake_latent`

#### Scenario: Guidance configuration is explicit
- **WHEN** a stage-4 finetuning config is created for the new task
- **THEN** the config explicitly defines guidance settings required for checkpoint compatibility instead of relying on implicit defaults

### Requirement: Two-motion evaluation can run through existing non-web entry points
The system SHALL support evaluating or smoke-testing the two-motion text task through config-driven scripts and helper entry points before any Gradio integration is required.

#### Scenario: Run qualitative paired-motion generation from scripts
- **WHEN** a developer runs a dedicated two-motion helper script with two `.npy` feature files and a config path
- **THEN** the system returns generated text using the configured checkpoint and the same motion-token pipeline used in finetuning

#### Scenario: Training artifacts remain in the standard experiment layout
- **WHEN** the two-motion stage-4 finetuning task is trained
- **THEN** logs, configs, and checkpoints are written under the existing `experiments/<model>/<NAME>/` layout used by other MotionGPT3 training runs
