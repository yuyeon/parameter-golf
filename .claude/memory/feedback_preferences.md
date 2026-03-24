---
name: feedback_preferences
description: User feedback on working style and tool usage for this project
type: feedback
---

Use micromamba on the RTX 3080 machine; plain python on the A40 instance (micromamba unavailable there).
**Why:** User rejected pyenv usage; micromamba is their package manager of choice. But A40 instance doesn't have it.
**How to apply:** Scripts auto-detect micromamba availability and fall back to plain python.

Verify hardware specs with actual commands before quoting numbers.
**Why:** User called out GPU spec estimates that weren't verified.
**How to apply:** Run actual commands, don't quote from memory.

Don't create README/documentation files unless explicitly asked.
**Why:** The repo has its own upstream README from OpenAI.
**How to apply:** Only create/modify docs when explicitly requested.

Run long experiments with nohup so they survive terminal disconnects.
**Why:** User explicitly requested this multiple times — they disconnect SSH sessions while experiments run.
**How to apply:** Always use `nohup bash scripts/<script>.sh > logs/<name>_nohup.log 2>&1 &` for any run >5 min.

Update memory regularly during long workflows.
**Why:** User explicitly requested this so sessions can be resumed on a different machine.
**How to apply:** Write memory after each milestone, not just at end of session.

Do not conclude something is "expected behavior" too quickly — treat anomalies as potential bugs first.
**Why:** User pushed back on dismissing the Baseline/10L_SW inversion as "expected early-training effect" when it was actually an eval-mode confound.
**How to apply:** Investigate suspicious results before explaining them away. Separate confounding factors.

Do not treat unmatched-budget comparisons as valid.
**Why:** Phase 2 timeout bug caused runs to complete different step counts, corrupting the ranking.
**How to apply:** Always verify steps_completed == target_steps before including a run in matched-budget analysis.

Keep session files in the repo so work can resume on a different machine.
**Why:** User explicitly requested this for continuity across SSH sessions and machines.
**How to apply:** Commit memory files, scripts, and configs. Keep artifacts/ gitignored but document what they contain.
