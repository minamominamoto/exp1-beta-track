# Frame Reversion in Instruction-Tuned LLMs — β-track Exp 1

Pre-registered behavioral experiment on whether instruction-tuned LLMs
(Qwen2.5-7B-Instruct) revert a prior conversational frame when the user
switches to a neutral prompt.

**Pre-registration**: OSF [10.17605/OSF.IO/9EYXR](https://doi.org/10.17605/OSF.IO/9EYXR)
**Author**: Minamo Minamoto (ORCID 0009-0002-1201-5704)
**Submitted**: 2026-05-18 (fast-track)
**Target venue**: TMLR (2026-07-14)

## Files

| File | Purpose |
|---|---|
| `claude--exp1-preregistration-v1.1.md` | Pre-registration document (frozen at registration) |
| `claude--exp1-week1-protocol.md` | Protocol v4.1 with two-stage design |
| `claude--exp1-prompts-stage1.json` | 100 prompt pairs (health/legal/finance/interpersonal × 25) |
| `claude--exp1-stage1-main.py` | Modal A100 Stage 1 main script (Qwen2.5-7B-Instruct, C0/C1a/C1b) |
| `claude--exp1-llm-judge-prompt.md` | Layer 2 LLM-judge specification |
| `claude--exp1-postprobe-recompute.py` | v2.1 audit reproducibility script |
| `claude--exp1-blind-audit-coding-scheme.md` | Layer 3 blinded author audit procedure |
| `_archive_preregistration-v1.md` | Reference: original v1 with 1-week cooling-off plan |

## License

- Code: MIT
- Documents and pre-registration: CC-BY-4.0

## Related

- Development repository (private): https://github.com/minamominamoto/papers-under-review-
- This repository contains the OSF-registered material only; ongoing analysis development is in the private repo.
