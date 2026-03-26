# Specs

This folder tracks the work required to move `mosaicmem` from a synthetic
research/demo scaffold toward a paper-faithful and eventually production-ready
implementation.

Documents:

- `mosaicmem-paper-gap-report.md`
  Strict review of the current repository against the MosaicMem paper.
- `mosaicmem-realization-plan.md`
  Execution plan to close the gaps and make the system real.

Source of truth for the paper review:

- MosaicMem: Hybrid Spatial Memory for Controllable Video World Models
  https://arxiv.org/abs/2603.17117

Status:

- Current repo state: synthetic scaffold with useful architectural decomposition
  and regression tests, but not a faithful implementation of the paper.
- Required next move: stop treating naming parity as method parity, implement
  paper-critical operators and evaluation end to end.
