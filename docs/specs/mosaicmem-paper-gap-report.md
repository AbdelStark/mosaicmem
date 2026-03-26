# MosaicMem Paper Gap Report

Date: 2026-03-26

Paper under review:

- MosaicMem: Hybrid Spatial Memory for Controllable Video World Models
- arXiv:2603.17117 v1
- Submitted: 2026-03-17
- URL: https://arxiv.org/abs/2603.17117

Repository under review:

- `mosaicmem`
- Local state reviewed on 2026-03-26
- Local checks run:
  - `cargo test`
  - `cargo clippy --all-targets --all-features -- -D warnings`

## Executive Summary

This repository is not a faithful implementation of the paper's actual system.
It is a synthetic, modular scaffold inspired by the paper's memory-side ideas.
That distinction is material, not semantic.

The codebase does contain real engineering value:

- clear module boundaries
- typed geometry and memory APIs
- regression tests for tensor layout and memory update behavior
- deterministic synthetic behavior for fast local iteration
- a useful sandbox for experimenting with patch storage, retrieval, alignment,
  and manipulation

But the codebase does not implement the paper's core model path:

- no real Wan 2.2 backbone
- no real depth/pose stack
- no fine-tuned memory-conditioned DiT
- no faithful PRoPE operator
- no faithful Warped RoPE operator
- no faithful Warped Latent operator
- no Mosaic Forcing autoregressive model
- no paper evaluation stack or benchmark reproduction

The correct description today is:

- a synthetic MosaicMem-style simulator and architectural scaffold

The incorrect description today is:

- a paper-faithful MosaicMem implementation
- a reproduction of the paper's quantitative claims
- a production-ready controllable video world model

## Review Method

This report compares:

- the paper's method and evaluation claims
- the local repository's actual code paths
- the repository's README/CLI claims where they imply paper fidelity

This is a strict implementation review, not a style review.

The main question is:

- does the repository implement the same mechanisms the paper depends on?

Where the answer is "no", the report says so directly.

## High-Level Verdict

Overall fidelity to the paper:

- architecture naming: partial
- conceptual decomposition: partial
- actual operators: weak to missing
- training and model path: missing
- evaluation reproduction: missing
- autoregressive paper method: missing
- production readiness: low

Short version:

- good scaffold
- not the paper

## What The Paper Actually Depends On

From the paper, the core system is not just "patch memory exists".
The key components are:

1. A real video DiT backbone based on Wan 2.2, fine-tuned with MosaicMem.
2. Patch-level spatial memory lifted into 3D from off-the-shelf 3D estimation.
3. Patch-and-compose conditioning in the queried view.
4. Memory alignment via two explicit mechanisms:
   - Warped RoPE
   - Warped Latent
5. PRoPE camera conditioning injected into attention as a projective operator.
6. Evaluation on camera control, retrieval consistency, dynamics, and video
   quality.
7. A separate autoregressive variant, Mosaic Forcing, built via causal
   distillation / forcing, not just overlap blending.

If any of these are absent or heavily simplified, method fidelity drops quickly.

## Current Repository Reality

The local repository states in `AGENTS.md` that it is:

- an alpha-quality research/demo codebase
- synthetic only
- not a real model-serving stack
- not loading external checkpoints

That description is accurate.

Important local files confirming current reality:

- `src/diffusion/backbone.rs`
- `src/diffusion/vae.rs`
- `src/geometry/depth.rs`
- `src/main.rs`
- `tests/meaningful_end_to_end.rs`

The main problem is not that the repository is synthetic.
The main problem is that the paper terminology can make the synthetic path look
closer to the paper than it actually is.

## Traceability Matrix

| Area | Paper expectation | Repo implementation | Assessment |
| --- | --- | --- | --- |
| Backbone | Fine-tuned Wan 2.2 5B DiT | Synthetic deterministic denoiser | Missing |
| VAE | Real 3D VAE with temporal compression semantics used by the model | Synthetic pooling/interpolation VAE | Missing |
| Depth / camera backend | Real off-the-shelf 3D estimator stack | Synthetic gradient depth only | Missing |
| Patch memory | Patch memory in 3D | Yes, but simplified to center-point patch metadata | Partial |
| Queried-view composition | View-specific patch-and-compose | Only for first pose in each window | Weak |
| Warped RoPE | Dense/fractional reprojection-based alignment | Quantized center-only positions | Weak |
| Warped Latent | Dense reprojection + bilinear grid sampling | Single homography from scalar patch depth | Weak |
| PRoPE | Projective attention transform in DiT attention | Heuristic rotations summarized into a scalar signal | Missing |
| Dynamic modeling | Real model prompt-following and dynamics under memory conditioning | Synthetic condition summaries | Missing |
| Evaluation | FID/FVD/RotErr/TransErr/Consistency/Dynamic | No paper evaluation harness | Missing |
| Autoregressive method | Mosaic Forcing with causal/rolling forcing | Sliding windows plus linear blending | Missing |
| Memory manipulation | Spatial patch manipulation/editing | Simple spatial ops on stored patches | Partial |

## Detailed Findings

### Critical Finding 1: No Real Model Path

The paper evaluates a fine-tuned Wan 2.2 5B TI2V model.
This repository does not include that model, a compatible attention stack, or a
real inference backend.

Current implementation:

- `SyntheticBackbone` produces outputs from local averages and condition
  summaries.
- `SyntheticVAE` is a hand-written pooling/interpolation transform.
- `SyntheticDepthEstimator` returns a radial depth gradient unrelated to image
  content.

Relevant files:

- `src/diffusion/backbone.rs`
- `src/diffusion/vae.rs`
- `src/geometry/depth.rs`

Why this matters:

- the paper's claims about pose adherence, dynamics, prompt following, and
  revisit retrieval all depend on a trained generative model
- a synthetic denoiser cannot validate those claims
- green tests here prove internal consistency of the scaffold, not paper
  reproduction

Severity:

- critical

### Critical Finding 2: PRoPE Is Not The Paper's PRoPE

The paper uses PRoPE as a projective attention operator:

- relative projective transforms between cameras
- injected into attention Q/K/V through a structured operator
- aware of temporal compression details

Current implementation:

- computes cosine/sine pairs from arbitrary elements of `K * R` and the
  translation vector
- never applies PRoPE inside the actual attention operator
- passes those cosine/sine pairs into the synthetic backbone
- the synthetic backbone reduces them to a single scalar summary

Relevant files:

- `src/attention/prope.rs`
- `src/pipeline/inference.rs`
- `src/diffusion/backbone.rs`

Specific issues:

- there is no implementation of `P_i P_j^-1`
- there is no block-diagonal projective transform applied to Q/K/V
- temporal compression is handled by integer division of frame indices, not by
  the paper's per-subframe camera treatment
- `PRoPE::apply` is effectively only used in tests, not in the inference path

Impact:

- the repository does not implement the paper's camera-control mechanism
- any claim that camera conditioning here matches the paper would be inaccurate

Severity:

- critical

### Critical Finding 3: Window Generation Retrieves Memory Only For The First Pose

The paper's patch-and-compose behavior is view-dependent.
Memory must be retrieved and aligned for the queried view.

Current implementation in `generate_window`:

- retrieves memory only for `poses[0]`
- applies Warped Latent only relative to `poses[0]`
- builds one rasterized memory canvas
- replicates that same memory canvas across all latent timesteps

Relevant files:

- `src/pipeline/inference.rs`
- `src/memory/mosaic.rs`

Why this matters:

- within-window camera motion does not receive frame-specific retrieval
- patch positions are not recomputed for later window frames
- the memory signal is effectively "first-view memory with temporal replication"
  rather than per-view memory conditioning

This is a foundational mismatch with the paper.

Severity:

- critical

### Critical Finding 4: The "Autoregressive" Path Is Not Mosaic Forcing

The paper's AR section is a different model:

- causal distillation
- causal forcing
- rolling forcing
- real-time AR generation behavior

Current repo behavior:

- split trajectory into overlapping windows
- generate each window independently
- linearly blend overlap frames
- update memory after each window

Relevant file:

- `src/pipeline/autoregressive.rs`

This is not a minor implementation shortcut.
It is a different algorithmic regime.

Impact:

- the repo cannot honestly claim to implement the paper's autoregressive method
- any AR results from this code would not be comparable to the paper's Mosaic
  Forcing results

Severity:

- critical

### High Finding 5: Warped RoPE Is A Coarse Center-Only Approximation

The paper's Warped RoPE is about alignment from geometric reprojection with
fractional coordinates and dense spatial-temporal correspondence.

Current implementation:

- uses the already projected patch center only
- quantizes that to integer bins
- derives `t` from absolute timestamp difference scaled into a fixed range
- repeats the exact same warped position for every token in the patch

Relevant files:

- `src/attention/warped_rope.rs`
- `src/attention/memory_cross.rs`

What is missing:

- intra-patch coordinate field
- fractional coordinate preservation
- dense reprojection per token or per sampling point
- richer temporal semantics than absolute age-to-bin scaling

Impact:

- local geometry inside a patch is not represented in the positional alignment
- large patches and oblique views will misalign badly
- the operator has the paper's name but not the paper's resolution or behavior

Severity:

- high

### High Finding 6: Warped Latent Is A Single-Plane Homography Shortcut

The paper describes dense reprojection and bilinear resampling using reprojected
coordinates.

Current implementation:

- computes one planar homography using one scalar patch depth
- assumes a fronto-parallel plane normal
- applies the same homography to the whole patch

Relevant file:

- `src/attention/warped_latent.rs`

What is missing:

- dense source-to-target coordinate field
- per-location geometry
- handling for non-planar patches and strong parallax
- explicit occlusion handling

Impact:

- works only as a rough approximation for small, locally planar patches
- likely unstable for exactly the kinds of viewpoint changes the paper is about

Severity:

- high

### High Finding 7: Memory Geometry Is Too Coarse

Stored patch metadata:

- one 3D center
- one depth scalar
- one source rectangle
- one latent tile

Relevant file:

- `src/memory/store.rs`

The paper's behavior needs more than that for faithful alignment.

Current limitations:

- retrieval visibility is based on center visibility only
- source patch geometry is not stored densely
- patch footprint in the target view is not modeled
- center-based projection is too weak for accurate patch placement

Impact:

- retrieval can overestimate visible utility of a patch
- target-view composition can be wrong even when the right patch was retrieved
- warping methods do not have the geometry they need

Severity:

- high

### High Finding 8: Coverage And Composition Are Simplified Beyond The Paper

Current coverage:

- marks only grid cells hit by projected patch centers

Current canvas composition:

- splats tokens around the projected center
- averages overlaps by visibility weight

Relevant files:

- `src/memory/retrieval.rs`
- `src/memory/mosaic.rs`

What this misses:

- actual projected footprint masks
- occlusion-aware visibility / z-buffer logic
- stronger composition semantics for patch overlap and invalid regions
- frame-varying memory coverage within a window

Impact:

- coverage can significantly misrepresent how much of the view is actually
  grounded by memory
- conditioning masks become weaker and less truthful than intended

Severity:

- high

### Medium Finding 9: Cross-Attention Exists Structurally, But Not As A Trained Operator

The memory cross-attention module is a structurally plausible placeholder:

- Q/K/V projections
- multi-head split/concat
- Warped RoPE on memory keys
- gated residual output

Relevant file:

- `src/attention/memory_cross.rs`

But in practice:

- weights are random initialization only
- no training path updates them
- generation query tokens are padded latent vectors, not a trained DiT token
  stream with the paper's attention context
- only memory keys receive Warped RoPE

Impact:

- this is a useful placeholder interface
- it is not evidence that the paper's learned memory conditioning behavior is
  implemented

Severity:

- medium

### Medium Finding 10: Memory Budgeting And Retrieval Utility Are Simplistic

Current behavior:

- retrieval score is center proximity plus depth, then optional temporal decay
- diversity is a greedy 2D penalty heuristic
- budget enforcement removes oldest patches

Relevant files:

- `src/memory/store.rs`
- `src/memory/retrieval.rs`

This is reasonable for a scaffold, but not a faithful utility model for a paper
about selective patch memory.

Impact:

- no learned memory utility
- no richer quality/confidence signal
- no direct support for sparse memory ablations or overlap removal policies in
  the way the paper discusses them

Severity:

- medium

### Medium Finding 11: The Prompt Path Is Functionally Placeholder

The CLI accepts a prompt, but the `generate` path creates a zero text embedding.

Relevant file:

- `src/main.rs`

Impact:

- prompt-following behavior is not real
- dynamic text-driven claims from the paper are not reproducible here

Severity:

- medium

### Medium Finding 12: The Tests Overstate Validation

The test suite is good for layout and plumbing regressions.
It is not a strong validator of paper fidelity.

Examples:

- the meaningful end-to-end test disables cross-attention by setting the memory
  gate to zero for both pipelines
- the integration test for cross-attention does not assert viewpoint outputs are
  actually different; it only checks that at least one summed output is non-zero
- tests around PRoPE validate the local heuristic implementation, not the paper
  operator

Relevant files:

- `tests/meaningful_end_to_end.rs`
- `tests/integration.rs`

Severity:

- medium

### Medium Finding 13: The Current CLI And Docs Need Stricter Truthfulness

The repository contains accurate caveats in `AGENTS.md` and `CONTRIBUTING.md`,
but the README and CLI naming still read close to a real implementation unless
the user reads carefully.

Examples:

- "Rust implementation of MosaicMem"
- prompt-oriented generate command with placeholder text conditioning
- paper component names used for approximations

This is a documentation and product-truthfulness issue, not just wording.

Severity:

- medium

## Areas That Are Legitimately Good And Worth Keeping

These are real strengths and should survive the rebuild:

- modular architecture split by camera, geometry, memory, attention, diffusion,
  and pipeline
- deterministic seeds for synthetic testing
- typed public errors across most library code
- explicit tests for `[B, C, T, H, W]` layout handling and frame extraction
- useful memory manipulation entry points for later scene editing work
- serializable memory snapshots for inspection and debugging
- a fast local scaffold for reference tests before integrating expensive models

The right move is not to throw this repo away.
The right move is to stop pretending the scaffold already proves the method.

## Production Readiness Assessment

Current production readiness:

- low

Reasons:

- no real checkpoint loading
- no GPU inference stack
- no model versioning
- no evaluation harness
- no performance envelope for real workloads
- memory update failures are downgraded to warnings in rollout generation
- KD-tree rebuild on every insert will not scale
- duplicated depth estimation / encode work in update paths

Relevant files:

- `src/pipeline/autoregressive.rs`
- `src/memory/store.rs`
- `src/pipeline/inference.rs`

This repo is suitable for:

- local experimentation
- API exploration
- synthetic regression tests
- architecture planning

This repo is not suitable for:

- paper reproduction claims
- production video generation
- model quality claims
- latency or memory benchmarking for a real system

## Tradeoffs Behind The Current Design

The simplifications are not irrational.
They buy:

- fast local tests
- no heavyweight model dependencies
- deterministic outputs
- clean modular interfaces
- research ergonomics

But they cost:

- inability to validate the paper's core claims
- inability to study actual camera control behavior
- inability to study true prompt-following under memory
- inability to reproduce the paper's evaluation

This tradeoff was acceptable for an alpha scaffold.
It is not acceptable if the goal is now "make it real".

## Definition Of Done For Calling The Repo "Real"

The repo can only be called a real MosaicMem implementation when all of the
following are true:

1. It runs a real video generation backbone compatible with the paper's setup.
2. It uses real depth and camera estimation backends.
3. PRoPE is implemented in attention as a projective operator, not a heuristic
   side channel.
4. Warped RoPE uses dense/fractional reprojection semantics.
5. Warped Latent uses dense reprojection-based bilinear resampling, not just a
   single homography shortcut.
6. Memory retrieval and composition are queried-view and frame-aware.
7. The evaluation harness reproduces the paper's metric categories.
8. The autoregressive path is Mosaic Forcing or is explicitly named something
   else.
9. README and CLI claims are aligned with what the code can actually do.

Until then, this repo should be described as:

- a synthetic scaffold for a future MosaicMem implementation

## Recommended Immediate Actions

1. Preserve the current synthetic stack as a fast test backend.
2. Add a second, real backend path instead of mutating the synthetic path into a
   half-real hybrid.
3. Make paper-faithfulness a tracked objective with explicit acceptance tests.
4. Split the roadmap into:
   - paper-faithful reproduction
   - production hardening
5. Tighten docs so claims do not outrun implementation.

## Final Verdict

The repository is technically coherent and internally tested.
It is also materially below the paper in every place where the paper's results
actually come from.

That is not fatal.
It just needs to be named honestly and rebuilt deliberately.

The next document in this folder is the execution plan for doing exactly that.
