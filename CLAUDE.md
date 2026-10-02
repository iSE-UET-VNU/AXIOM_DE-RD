# Rules for Claude in this repo

- Never run experiments, API calls, background jobs or scripts that write results without explicit approval for that exact run.
- Before writing any evaluation or QA script, list every setting (top-k, models, judge, prompt, k, weights) and get confirmation. Never copy a setting from old code without asking.
- QA context is top 10 unless the user says otherwise.
- Show every file change before making it. Never delete, move or overwrite without approval.
- No comments in code. No docstrings unless asked.
- Keep replies short.
