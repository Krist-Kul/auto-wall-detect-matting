# ADR 0004 — Model infrastructure

**Status:** Accepted (grilling)
**Date:** 2026-06-18

## Decision
- Install **GroundingDINO and SAM via HuggingFace `transformers`** into the existing
  `birefnet` conda env (Python 3.10, torch 2.12, RTX 5070). No custom CUDA-op build.
- Use the **Large/Huge SAM** backbone (best edge quality; offline mask generation).

## Candidate model IDs (verify before download, do not assume)
- GroundingDINO: `IDEA-Research/grounding-dino-base` (HF `GroundingDinoForObjectDetection`).
- SAM (huge): `facebook/sam-vit-huge` (HF `SamModel` / `SamProcessor`).
- Confirm both load on torch 2.12 / sm_120 and fit GPU memory alongside BiRefNet; if the
  RTX 5070 (sm_120) needs it, fall back to CPU for DINO or load models sequentially.

## Consequences
- Adds `transformers` (+ deps) and ~2–3 GB of weights to the HF cache.
- One-time download latency; everything cached afterwards.
- Three models now resident (BiRefNet + DINO + SAM); may need sequential load / `.to(cpu)`
  juggling if VRAM is tight.
