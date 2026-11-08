# Sources and licenses

Original Aawden code and fixtures use the repository MIT license. No model weights are bundled.

| Material | Attribution / revision | License scope |
|---|---|---|
| CircuitKIT | [Lexsi Labs](https://github.com/Lexsi-Labs/CircuitKIT), e71d5e17fe6e2abfde47ec9a7fbaf31649601b78 | Pinned source declares LSAL-1.2; consult its LICENSE.md |
| AuditKIT | [Lexsi Labs](https://github.com/Lexsi-Labs/AuditKIT), 71316da345784ff220d5ed59337441a59ab31526 | Pinned source declares LSAL-1.2; consult its LICENSE.md |
| Aya Expanse 8B | [Cohere Labs model card](https://huggingface.co/CohereLabs/aya-expanse-8b), 5062468bf9bc0c6035fd64e06274333ec127d980 | CC-BY-NC-4.0, model access conditions and Cohere Labs Acceptable Use Policy. Loaded on an authorized host for research. Base weights are not redistributed in the code bundle. |
| PyTorch, Transformers, TransformerLens, safetensors | Optional research dependencies; versions in research lock | Retain their respective licenses |
| Research handoff | User-supplied RESSOURCES/CYBER_DEFENSE_WEIGHT_SPACE_DEVELOPER_GUIDE.md | Input specification; not relicensed |

Sponsor source archives are kept in the ignored .vendor directory for local inspection and AuditKIT execution. Their archives are SHA-256 verified by scripts/fetch_toolkits.py. Submission packaging excludes .vendor, private inputs, credentials and model weights.

The direct projection implementation and original executable benchmark are authored for Aawden. The supplied guide informed their protocol. Published weight-space research described in that guide is background, not a result reproduced by Aawden.

