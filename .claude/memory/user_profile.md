---
name: user_profile
description: User's hardware setup, preferences, and working style for parameter-golf project
type: user
---

- Has an RTX 3080 12GB GPU (confirmed via nvidia-smi) — primary target for experiments
- Also has access to an A40 48GB (as of 2026-03-22) — for parallel experiment screening
- Running WSL2 on Windows (Linux 6.6.87.2-microsoft-standard-WSL2) for 3080, separate env for A40
- Uses micromamba (not pyenv) for Python environment management; env name: parameter-golf
- Prefers concise, direct communication
- Wants to verify claims with actual commands/measurements rather than estimates from memory
- Gets impatient with long-running tasks — prefers shorter iteration cycles
- PC crashes under high VRAM load (~95% usage) — display freezes, requires hard restart
