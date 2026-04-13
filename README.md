# PlantVillage — Production Pipeline (`latest_changes/`)

A modular, production-grade plant disease diagnosis system built on top of
Qwen2.5-VL-7B-Instruct (fine-tuned with LoRA) and CLIP ViT-B/32.

---

## Folder structure

```
latest_changes/
├── config.py          # All constants (paths, thresholds, generation params)
├── pipeline.py        # Single-call run_pipeline() function
├── app.py             # Streamlit frontend
├── models/
│   └── loader.py      # load_vlm() and load_clip()
└── utils/
    ├── ood.py         # CLIP-based OOD detection
    ├── inference.py   # Qwen2.5-VL generation
    ├── parser.py      # Regex field extractor
    └── validator.py   # Plant mismatch check
```

---

## Pipeline flow

```
INPUT IMAGE
    │
    ▼
[1] OOD CHECK (CLIP cosine similarity)
    │  score < 0.27 → STOP: return OOD error
    ▼
[2] VLM INFERENCE (Qwen2.5-VL)
    │
    ▼
[3] PARSE FIELDS (Plant / Condition / Severity / Pathogen)
    │
    ▼
[4] PLANT VALIDATION (selected vs detected)
    │  mismatch → STOP: return WRONG_PLANT error
    ▼
[5] RETURN SUCCESS RESULT
```

---

## Run the app

```powershell
cd C:\Users\Admin\Desktop\VLM
.\venv\Scripts\Activate
cd latest_changes
streamlit run app.py
```

---

## Configuration

Edit `config.py` to change:
- Model / adapter paths
- OOD threshold (default 0.27)
- Generation parameters
- Supported plant list

---

## Return format

**Success:**
```json
{
  "status": "success",
  "data": {
    "Plant": "Tomato",
    "Condition": "Late Blight",
    "Severity": "Moderate to Severe",
    "Pathogen": null
  },
  "raw": "Plant: Tomato. Condition: Late Blight. ..."
}
```

**OOD error:**
```json
{
  "status": "error",
  "type": "OOD",
  "message": "Image not recognised as a plant leaf.",
  "score": 0.19
}
```

**Plant mismatch:**
```json
{
  "status": "error",
  "type": "WRONG_PLANT",
  "expected": "Tomato",
  "detected": "Potato"
}
```

---

## Extending for future crops

1. Add the new crop to `SUPPORTED_PLANTS` in `config.py`.
2. Train a new LoRA adapter and point `ADAPTER_PATH` to it (or add per-crop paths).
3. The rest of the pipeline works without modification.
