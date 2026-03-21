# Multi-Agent Orchestration Protocol — mosaicmem-rs

<applicability>
This project benefits from multi-agent coordination when:
— Implementing features spanning multiple modules (e.g., new attention mechanism + pipeline integration)
— Parallelizing independent module work (geometry changes + attention changes)
— Performing large refactors across 3+ modules
For single-module tasks, a single agent is sufficient.
</applicability>

<roles>
| Role          | Model Tier | Responsibility                                 | Boundaries                                |
|---------------|------------|-------------------------------------------------|-------------------------------------------|
| Orchestrator  | Frontier   | Decompose tasks, plan across modules, review    | NEVER writes implementation code           |
| Implementer   | Mid-tier   | Write Rust code, run tests, fix clippy warnings | NEVER makes architectural decisions        |
| Reviewer      | Frontier   | Validate correctness, catch unsafe patterns     | NEVER implements fixes (sends back)        |
| Geometry      | Mid-tier   | Camera, projection, point cloud, fusion work    | Only operates in camera/ and geometry/     |
| Memory        | Mid-tier   | Store, retrieval, mosaic, manipulation work     | Only operates in memory/                   |
| Attention     | Mid-tier   | RoPE, cross-attention, positional encoding      | Only operates in attention/                |
</roles>

<delegation_protocol>
1. ANALYZE: Identify which modules are affected by the task.
2. DECOMPOSE: Break into per-module sub-tasks with clear interfaces.
3. CLASSIFY:
   — Single module, well-defined → Delegate to specialist (Geometry/Memory/Attention)
   — Cross-module integration → Delegate to Implementer with explicit file list
   — Architectural (new trait, API change) → Orchestrator handles directly
4. PLAN: Define execution order. Modules with no shared types can parallelize.
5. DELEGATE: Issue task with full context (see task format below).
6. VERIFY: After each sub-task, run `cargo test` and `cargo clippy`.
7. INTEGRATE: Merge results, ensure cross-module types align.
</delegation_protocol>

<task_format>
## Task: [Title]

**Objective**: [What "done" looks like — one sentence]

**Context**:
- Files to read: [exact paths]
- Files to modify: [exact paths]
- Files to create: [paths with naming convention]
- Related types: [struct/trait names to be aware of]
- Relevant RFC: [RFC-00X if applicable]

**Acceptance criteria**:
- [ ] `cargo test` — all tests pass (46+ tests)
- [ ] `cargo clippy` — no warnings
- [ ] `cargo fmt -- --check` — no formatting issues
- [ ] New code has `#[cfg(test)] mod tests` with 3+ tests

**Constraints**:
- Do NOT modify: [files outside scope]
- Do NOT change: trait signatures without Orchestrator approval

**Handoff**: Report completion with summary of changes and test results.
</task_format>

<parallel_execution>
Safe to parallelize (no shared files):
— camera/ + attention/ + diffusion/
— geometry/pointcloud.rs + memory/manipulation.rs
— Any two specialist agents in non-overlapping modules

Must serialize:
— Changes to trait definitions (affects all implementors)
— Pipeline module changes (depends on all other modules)
— mod.rs re-export changes (affects downstream imports)
— Cargo.toml dependency changes

Conflict protocol:
1. Before starting, list files to be modified
2. If overlap with another agent's scope, coordinate via Orchestrator
3. After parallel completion, run `cargo test` on merged result
</parallel_execution>

<escalation>
Escalate to human when:
— Trait interface changes needed that affect 3+ modules
— Performance requirements unclear (e.g., real-time vs. batch)
— ONNX model integration questions (model format, input shapes)
— Design decisions not covered by existing RFCs
— Any change to SPECIFICATION.md or Cargo.toml

Format:
**ESCALATION**: [one-line summary]
**Blocker**: [specific issue]
**Options**: [2-3 alternatives with tradeoffs]
**Recommendation**: [preferred option and why]
</escalation>
