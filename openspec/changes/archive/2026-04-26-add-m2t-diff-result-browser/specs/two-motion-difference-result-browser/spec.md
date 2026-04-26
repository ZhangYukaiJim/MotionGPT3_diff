## ADDED Requirements

### Requirement: MotionFix `m2t_diff` sample folders are browsable through a separate local result browser
The system SHALL provide a separate local browser workflow for MotionFix `m2t_diff` qualitative sample folders that reads existing exported artifacts from a selected `samples_<TIME>` directory without rerunning model inference.

#### Scenario: Browser loads a sample directory
- **WHEN** a user launches the result browser and points it at a MotionFix `m2t_diff` sample directory
- **THEN** the browser enumerates available samples from the existing exported files and presents them for interactive review without requiring checkpoint loading or generation

### Requirement: The main browser view is a paginated gallery-first sample browser
The system SHALL present the main browser screen as a paginated gallery of samples where each sample card is keyed by sample id and includes the side-by-side preview video when available, generated text, ground-truth text, and favorite state.

#### Scenario: User reviews one gallery page of samples
- **WHEN** the browser opens a sample directory that contains more samples than fit on one page
- **THEN** the browser shows only the configured number of sample cards for the current page and provides controls to move between pages

#### Scenario: Sample card remains usable with missing preview video
- **WHEN** a sample is present in the directory but its side-by-side preview video is missing because the export is partial or interrupted
- **THEN** the gallery still shows the sample id and available text fields and marks the preview as unavailable instead of omitting the sample entirely

### Requirement: The browser provides a detail view for the selected sample
The system SHALL provide a detail view synchronized with the selected gallery sample that exposes larger source, target, and side-by-side videos when available, along with the full generated text and full ground-truth text.

#### Scenario: Selecting a gallery sample updates the detail panel
- **WHEN** a user selects a sample from the gallery
- **THEN** the browser updates the detail view to show that sample's larger available videos and full text content

### Requirement: The browser supports page and sample navigation controls
The system SHALL provide both visible controls and keyboard shortcuts for moving between gallery pages and navigating among samples during qualitative review.

#### Scenario: User changes pages with keyboard navigation
- **WHEN** the user triggers a configured page-navigation keyboard shortcut while the browser is focused
- **THEN** the browser moves to the previous or next gallery page and updates the visible sample cards accordingly

#### Scenario: User changes the selected sample with navigation controls
- **WHEN** the user activates a previous-sample or next-sample control from the current selection
- **THEN** the browser updates the detail view to the adjacent sample, crossing page boundaries when needed

### Requirement: Favorites persist independently from exported sample artifacts
The system SHALL allow users to mark and unmark samples as favorites, and SHALL persist favorite state in a deterministic sidecar metadata file associated with the browsed sample directory rather than by mutating exported sample artifacts.

#### Scenario: Favorite state survives browser restart
- **WHEN** a user marks one or more samples as favorites, closes the browser, and later reopens the same sample directory
- **THEN** the browser reloads the previously saved favorites and displays the same samples as favorited

#### Scenario: Favorite toggle updates persistent state for one sample
- **WHEN** a user toggles favorite state for a selected sample
- **THEN** the browser updates both the current UI state and the persisted sidecar metadata so the new favorite status is durable
