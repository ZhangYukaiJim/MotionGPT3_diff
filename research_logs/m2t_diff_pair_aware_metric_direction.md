# Longer-Term Direction: Pair-Aware Metric For `m2t_diff`

## Motivation

The current recommended metric direction for MotionFix `m2t_diff` is to score generated captions as text against reference captions. That is the most defensible first quantitative layer, but it does not directly answer the harder question:

- Does the generated caption actually describe the relationship between the source motion and the target motion?

Existing repo evaluators such as TM2T and TMR assume a single motion-text relationship. They do not natively evaluate whether text matches a motion pair.

## Proposed Longer-Term Metric

Build a pair-aware motion-text retrieval evaluator for `m2t_diff`.

High-level idea:
- encode source motion into a source-motion latent
- encode target motion into a target-motion latent
- combine them into a pair representation with stable ordering
- encode the caption into a text latent
- evaluate retrieval between text latents and pair latents

This would give a motion-aware score aligned with the actual paired-motion task.

## Candidate Representation Strategies

### 1. Concatenated pair embedding

- `pair_latent = concat(src_latent, tgt_latent)`

Pros:
- simplest
- preserves source/target order explicitly
- easy to prototype with existing encoders

Cons:
- may overfit to encoder scale/layout quirks
- pair dimensionality doubles unless projected back down

### 2. Difference-aware composition

- `pair_latent = proj(concat(src_latent, tgt_latent, tgt_latent - src_latent))`

Pros:
- makes directional change explicit
- likely better aligned with edit-style captions

Cons:
- introduces more design choices
- requires learning or defining the projection layer

### 3. Cross-attentive pair encoder

- encode source and target separately, then run a small pair encoder over both streams before retrieval

Pros:
- most expressive
- can capture relational structure beyond simple subtraction

Cons:
- largest implementation scope
- no longer a lightweight evaluator-only change

## Recommended First Prototype

Start with a frozen-encoder prototype:

1. Reuse an existing motion encoder backbone for each motion.
2. Build a pair embedding as:
   - `concat(src, tgt, tgt - src)`
3. Project that into the same text-latent dimensionality.
4. Train only the lightweight projection / retrieval head.
5. Evaluate retrieval in both directions:
   - text -> motion pair
   - motion pair -> text

This gives a useful signal without retraining the full motion-language stack.

## Suggested Metrics

For a pair-aware retrieval evaluator, report:
- `t2p_R@1`, `t2p_R@3`, `t2p_R@5`, `t2p_MedR`
- `p2t_R@1`, `p2t_R@3`, `p2t_R@5`, `p2t_MedR`
- diagonal similarity mean

Where:
- `t2p` = text to paired-motion retrieval
- `p2t` = paired-motion to text retrieval

## Dataset Considerations

MotionFix captions are often short, edit-style, and single-reference. That creates a few evaluation challenges:
- some captions may be semantically valid paraphrases of each other but lexically different
- some motion pairs may support more than one reasonable description
- retrieval labels are cleaner than n-gram overlap, but only if the pair representation is actually relation-aware

This is why a pair-aware retrieval metric is attractive long-term, but it should not be rushed into the first evaluation rollout.

## Recommended Next Research Steps

1. Create a small offline analysis script that embeds source/target motions and captions for a MotionFix subset.
2. Compare simple pair embeddings:
   - `concat(src, tgt)`
   - `concat(src, tgt, tgt-src)`
   - `concat(src, tgt, abs(tgt-src))`
3. Check nearest neighbors qualitatively for a few hand-inspected examples.
4. If the pair structure looks meaningful, define an OpenSpec change for a pair-aware retrieval evaluator.

## Boundary With Current Work

This is future work.

It should stay separate from the immediate `m2t_diff` evaluation rollout, which should focus on:
- `Bert_F1`
- `ROUGE_L`
- `CIDEr`
- empty-output rate
- average generated length
