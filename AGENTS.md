# Repository agent instructions

## Project alignment

- The active objective is retrospective supervised prediction of speech ideas from news/context headlines published in the seven days before the target speech.
- `docs/idea-layer/README.md` is the canonical task specification. Keep the root README and `CONTEXT.md` aligned with it.
- A target event's transcript is its outcome/label, never its own prediction input. Earlier context/label pairs may train predictions for later events.
- Speaker-history recurrence, prior-speech-only forecasting, continuation, speech planning, live capture, and Trump-post archives are not the active objective. Do not extend those prototypes unless explicitly asked.
- Preserve ignored local data and artifacts. Never stage or push them.

## Required delivery workflow

Unless the user explicitly asks for a local-only change, treat finishing a repository task as completing delivery to the repository's default branch—not merely editing files or creating a local commit.

1. Inspect the branch, upstream, working tree, ignored local artifacts, and existing PRs before changing or publishing anything. Preserve unrelated user work.
2. Run the relevant tests and checks. Review the final diff for secrets, local data, generated artifacts, and unintended changes; verify the worktree is clean before publishing.
3. Make clear, logical commits with descriptive messages. Do not leave task changes uncommitted.
4. Push the topic branch, open a PR to the repository's default branch, and include a concise summary, test results, and material limitations.
5. Check PR mergeability, required reviews, and CI status. When checks and repository policy permit, merge the PR (preserving commit history unless the repository convention says otherwise). If blocked by policy, failed checks, missing review, or authentication, report the blocker instead of claiming completion.
6. After merge, delete the remote and local topic branches, fast-forward the local default branch, and verify the PR state, merge commit, clean worktree, and branch deletion.
7. Report the commit(s), PR, merge, branch cleanup, checks, and any remaining blocker. Never imply a change was pushed or merged until verified.

If the user requests an exception (for example, do not push, do not open a PR, or do not merge), follow that instruction and state where the work remains.
