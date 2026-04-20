## Purpose

Define the MotionFix paired-motion data workflow used by MotionGPT3 training, evaluation, and preprocessing.

## Requirements

### Requirement: MotionFix dataset root validation
The system SHALL require a configured MotionFix dataset root that contains `motionfix.pth.tar` and `splits.json` before any MotionFix-backed training, evaluation, or preprocessing run starts.

#### Scenario: Required MotionFix files are present
- **WHEN** a MotionFix preprocessing or datamodule setup command is started with a dataset root containing `motionfix.pth.tar` and `splits.json`
- **THEN** the system proceeds with dataset loading using those files

#### Scenario: Split metadata is missing
- **WHEN** a MotionFix preprocessing or datamodule setup command is started with a dataset root that does not contain `splits.json`
- **THEN** the system fails fast with an explicit setup error that identifies the missing file and does not guess train, validation, or test splits

### Requirement: MotionFix motions convert into MotionGPT3-compatible features
The system SHALL convert each MotionFix `motion_source` and `motion_target` sequence into the same HumanML-compatible motion feature representation used by the existing MotionGPT3 VAE pipeline, preserving frame order and per-sample pairing.

#### Scenario: Convert a paired MotionFix sample
- **WHEN** preprocessing reads one MotionFix sample containing `motion_source` and `motion_target` joint sequences shaped `[T, 22, 3]`
- **THEN** the system produces one converted source feature array and one converted target feature array in the MotionGPT3 motion feature space with matching sample identity metadata

#### Scenario: Preserve source-target ordering
- **WHEN** preprocessing writes converted caches for a MotionFix sample
- **THEN** the system records source and target outputs with stable ordering so downstream training can distinguish the first motion from the second motion

### Requirement: HumanML normalization compatibility is preserved for converted MotionFix features
The system SHALL normalize converted MotionFix motion features using the HumanML mean and standard deviation expected by the pretrained MotionGPT3 VAE instead of computing a replacement normalization for the first integration path.

#### Scenario: Reuse HumanML normalization during conversion
- **WHEN** MotionFix features are prepared for VAE encoding in the initial implementation
- **THEN** the system applies the same HumanML normalization values used by the current HumanML3D MotionGPT3 pipeline

#### Scenario: MotionFix-specific normalization is not silently substituted
- **WHEN** the MotionFix integration is used with the pretrained HumanML-based VAE
- **THEN** the system does not silently swap in dataset-local mean or standard deviation values for the primary normalized training path

### Requirement: MotionFix paired samples are available through a config-driven datamodule
The system SHALL provide a config-driven MotionFix datamodule path that yields paired source features, paired target features, text supervision, and both source and target lengths for train, validation, and test splits.

#### Scenario: Build MotionFix datamodule from experiment config
- **WHEN** a training or evaluation config selects the MotionFix dataset target
- **THEN** `train.py` and `test.py` can build the datamodule through the existing OmegaConf-driven dataset construction path

#### Scenario: Batch collation preserves paired fields
- **WHEN** the MotionFix dataloader collates a batch
- **THEN** each batch contains text supervision and separate source and target motion tensors with their corresponding lengths so the model can encode both motions deterministically
