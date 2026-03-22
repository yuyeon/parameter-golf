---
name: feedback_preferences
description: User feedback on working style and tool usage for this project
type: feedback
---

Use micromamba, not pyenv, for Python environments.
**Why:** User rejected pyenv usage; micromamba is their package manager of choice.
**How to apply:** Always use `micromamba run -n parameter-golf` to run Python commands.

Verify hardware specs with actual commands before quoting numbers.
**Why:** User called out GPU spec estimates that weren't verified. Use nvidia-smi, not web searches from memory.
**How to apply:** Run `nvidia-smi --query-gpu=...` before making claims about the user's GPU.

Don't create README/documentation files unless explicitly asked.
**Why:** Standard Claude Code guidance; the repo has its own README from OpenAI.
**How to apply:** Only create docs when the user's task spec calls for it (e.g., docs/PROXY_FRAMEWORK.md was explicitly requested).
