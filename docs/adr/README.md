# Architecture Decision Records

Decisions for the wall/floor surface-mask pipeline, captured during the grilling session.
See `../pipeline-design.md` for the assembled design and `../glossary.md` for terms.

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-detector-scope.md) | Detector scope: GroundingDINO vs existing auto_detect bboxes | Accepted (amended by 0002) |
| [0002](0002-completeness-hybrid.md) | Object completeness: hybrid BiRefNet catch-all + GroundingDINO | Accepted |
| [0003](0003-sam-roles.md) | SAM's two roles: occluder cut-masks and surface masks | Accepted |
| [0004](0004-infra.md) | Model infrastructure (HF transformers, SAM-huge) | Accepted |
| [0005](0005-subtraction-rule.md) | Object subtraction rule (all objects from both surfaces) | Accepted |
| [0006](0006-outputs-rug-scope.md) | Outputs, rug deliverable, run scope | Accepted |
| [0007](0007-dino-prompts-acceptance.md) | DINO prompt list & acceptance criteria | Accepted |
| [0008](0008-structure-first-normal-clustering.md) | Structure-first: BiRefNet-invert + normal/depth plane clustering (`main_normal`) | Accepted |

Status legend: Proposed → Accepted → Superseded.
