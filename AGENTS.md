# VibeSim Workspace

This repository pins four Git submodules and provides shared build entry points.
Read the relevant README and local agent instructions before editing.

| Directory | Responsibility |
| --- | --- |
| `VibeSim/` | Simulator, Analyzer, launcher, profiling, and model/GPU definitions |
| `VibeSimAgent/` | Agent execution, workspaces, conversations, and job lifecycle |
| `VibeSimUI/app/` | Analysis and Agent browser UI |
| `vibesim-intro/` | Introduction website |

- Start with `README.md`; use `reproduce.md` for setup and `VibeSim/doc/README.md`
  for simulator design. Consult module READMEs for implementation details.
- Before starting any task, you MUST search the available skills for applicable
  workflows, read the matching `SKILL.md` files, and follow them. Check both
  `VibeSim/skills/` and user-installed skills; `skill-of-skills/SKILL.md` maps
  VibeSim workflows. General-purpose skills belong in the user's skill directory.
- `CLAUDE.md` links to this file. `.agents/skills` and `.claude/skills` link to
  `VibeSim/skills/`. Edit the source, not a separate copy.
- Check the owning repository, branch, and existing changes. Preserve unrelated
  work; commit component changes separately from root submodule-pointer updates.
- Maintain root `worktree.md` with each worktree's path, repository, branch,
  purpose, and current status; update it when creating, moving, or retiring a
  worktree, or when its task status changes.
- Run long-running jobs in named `tmux` sessions. Record the session name,
  working directory, command, and log path in the worktree's `progress.md`.
- Follow each component's environment and test instructions; use `uv` in
  `VibeSim/`. Test the affected behavior and limit formatting to intended files.
- Run physical GPU checks with elevation; sandbox device failures do not prove
  that the host GPU is unavailable.
- Read and maintain `goal.md`, `progress.md`, and `notes.md` in the working tree.
  Keep these records out of commits unless requested.
