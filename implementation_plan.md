# Implementation Plan: AI-Powered Image Quality & Defect Detection
### 48-Hour Internship Technical Assessment — Full Build Plan

---

## 0. Guiding Principle: Build Against the Rubric, Not Your Ego

The grading weights tell you exactly where your hours should go. Memorize this table before you write a line of code:

| Area | Weight | What it actually rewards |
|---|---|---|
| AI / ML / DL implementation | 25% | A real learned model, justified choices |
| CV understanding & feature reasoning | 15% | Correct, well-explained image features |
| Model evaluation & experimental rigor | 15% | Honest metrics, failure case analysis |
| Backend / API | 15% | Correctness, validation, error handling |
| Deployment & reproducibility | 10% | `docker compose up` just works, cold |
| Frontend functionality & usability | 10% | Clear states, not visual polish |
| Code quality & documentation | 10% | Readable, explained, no dead code |

**55% of your grade is ML understanding + evaluation rigor.** Backend + frontend + deployment together are 35%, and code quality is 10%. Spend disproportionate *personal* attention (not agent-delegated attention) on Phases 1–4 below. Everything else can be heavily agent-generated (Antigravity) as long as you review and understand it.

---

## Tech Stack (matched to your existing foundation)

| Layer | Choice |
|---|---|
| Backend | FastAPI + SQLAlchemy |
| Database | PostgreSQL (Docker), SQLite fallback for fast local dev |
| ML / CV | Python, OpenCV, scikit-image, NumPy, PyTorch, scikit-learn |
| Frontend | React + Vite |
| Deployment | Docker + Docker Compose, Nginx for the frontend build |
| Testing | pytest (backend), manual E2E pass (frontend) |

---

## Phase 0 — Project Setup (Hour 0 → 1)

**Objective:** A clean, agent-friendly repo skeleton so Antigravity has structure to work inside rather than inventing its own.

- Initialize git repo, `.gitignore` (node_modules, __pycache__, .env, *.pt, uploads/, pgdata/)
- Create top-level structure:
  ```
  /backend        FastAPI app
  /frontend       React + Vite app
  /ml             dataset generation, training, evaluation scripts
  /docker         Dockerfiles + compose
  /docs           README assets, sample images, evaluation report
  /data           generated dataset (gitignored if large)
  ```
- Backend: Python venv, `requirements.txt` pinned (fastapi, uvicorn, sqlalchemy, psycopg2-binary, pillow, python-multipart, torch, opencv-python, scikit-image, scikit-learn, pandas, numpy, pytest)
- Frontend: `npm create vite@latest frontend -- --template react`
- Write a skeleton README with section headers only (fill in later — this doubles as your submission-requirements checklist)

**Deliverable:** empty-but-runnable skeleton, `README.md` outline, requirements files committed.

---

## Phase 1 — Dataset & Synthetic Degradation Pipeline (Hour 1 → 6)

This is the single most important phase for your grade. Own this yourself — don't delegate the reasoning to the agent, only the boilerplate.

### 1.1 Source clean images
- Pull ~150–300 clean, varied images: a small public set (e.g. a DIV2K/COCO subset) or your own photos. Variety matters more than volume (indoor/outdoor, faces/objects/scenes, different lighting).

### 1.2 Define degradation functions (`ml/degrade.py`)
Each function should support a **severity parameter** (mild/moderate/severe) so your labels are graded, not just binary — this alone demonstrates more CV understanding than binary labeling.

| Issue | Method | Severity control |
|---|---|---|
| Blur | `cv2.GaussianBlur` | kernel size / sigma |
| Underexposure | multiply by gamma < 1, or subtract constant | gamma value |
| Overexposure | multiply by gamma > 1, clip highlights | gamma value |
| Noise | additive Gaussian noise, optional salt-and-pepper | sigma |
| Corruption | aggressive JPEG re-encode (quality 5–15), or truncate file bytes | quality level |
| Defect | synthetic occlusion patch, channel swap/shift, random scratch lines | patch size/count |

### 1.3 Generate the labeled dataset
- For each clean image, apply 0–3 randomly chosen degradations at random severities (images can have multiple co-occurring issues — this is realistic and lets you demonstrate multi-label classification, which is stronger than single-label).
- Write a manifest (`data/manifest.csv`): `filename, blur_severity, underexposure_severity, overexposure_severity, noise_severity, corruption_severity, defect_present, derived_quality_score`
- Derive `quality_score` as a deterministic function of applied severities (documented formula, e.g. `100 - Σ(severity_weight_i)`), so you have ground truth for regression evaluation later.

### 1.4 Split & sanity check
- 70/15/15 train/val/test split, stratified by issue type where feasible.
- Visualize a grid of ~20 degraded samples with `matplotlib` to confirm degradations look realistic, not like noise-on-noise. **Save this grid image — it goes straight into your README as evidence of thoughtful data generation.**

**Deliverable:** `data/` folder (raw + degraded + manifest), `ml/degrade.py`, `ml/generate_dataset.py`, a sample-grid PNG.

---

## Phase 2 — Feature Engineering (Hour 6 → 9)

Implement `ml/feature_extractor.py` → `extract_features(image) -> dict`. Target ~18–22 features across these families (this list itself is worth writing near-verbatim into your README as your "CV reasoning" section):

- **Sharpness:** variance of Laplacian, Tenengrad (mean Sobel gradient magnitude), high-frequency energy ratio via FFT
- **Exposure:** mean luminance, % pixels < 10, % pixels > 245, histogram skewness
- **Contrast:** RMS contrast (std/mean of intensity), Michelson contrast
- **Noise:** fast noise estimation (Immerkær's Laplacian-based sigma estimator), local variance in flat/low-texture regions
- **Color/Saturation:** mean HSV saturation, per-channel mean imbalance
- **Texture:** GLCM contrast / homogeneity / energy (`skimage.feature.graycomatrix`)
- **Compression artifacts:** blockiness score via 8×8 DCT boundary discontinuity check

Run extraction over the full dataset → `features.csv` (feature matrix X + multi-label Y + regression target). Do a 10-minute EDA pass (correlation matrix, quick feature-importance sanity check) — catches broken features before you waste training time on them.

**Deliverable:** `feature_extractor.py`, `features.csv`, a short EDA notebook or script output.

---

## Phase 3 — Model Training: The Hybrid Core (Hour 9 → 14)

This is your 25%-weight section. Build **two models**, which directly satisfies the assessment's explicit "hybrid approach combining image-quality features with a learned model" option while giving you real deep-learning credit via PyTorch (matches your existing stack):

### 3.1 Model A — Multi-label issue classifier (PyTorch MLP)
- Input: ~20-dim feature vector → hidden layers (64 → 32) → 6 sigmoid output heads (blur, underexposure, overexposure, noise, corruption, defect)
- Loss: BCE per label, Adam optimizer, early stopping on val loss
- This gives you per-issue **confidence** (the sigmoid output) directly — satisfies the "confidence" field in the required JSON response for free.

### 3.2 Model B — Convolutional autoencoder for anomaly/defect detection
- Small 3–4 layer conv encoder/decoder, trained **only on clean images**, input resized to e.g. 128×128, MSE reconstruction loss
- At inference: reconstruction error = anomaly signal for "potential visual defect"; the **pixel-wise error map** is your bonus-point quality heatmap / localization, essentially for free
- This is the part that most clearly demonstrates real deep-learning work beyond hand-built features — call this out explicitly in your README

### 3.3 Combine into the final output
- `quality_score` = weighted combination of (100 − Σ active-issue penalties from Model A) and normalized reconstruction error from Model B — document the exact formula and weights, and justify them in one paragraph
- Define `quality_label` thresholds explicitly (e.g. ACCEPTABLE ≥ 75, DEGRADED 40–74, DEFECTIVE < 40) with reasoning, not arbitrary numbers

**Deliverable:** `train_mlp.py`, `train_autoencoder.py`, saved checkpoints in `ml/models/`, a written paragraph on architecture choices and why (this paragraph is what you'll paraphrase into the README).

---

## Phase 4 — Evaluation & Explainability (Hour 14 → 17)

This is your other high-weight, easy-to-neglect section (15% + explainability requirement). Do not skip the honest parts.

- **Model A:** per-label precision/recall/F1, per-issue confusion matrix, ROC-AUC
- **Model B:** reconstruction-error distribution for clean vs. defective images, threshold selection on validation set, basic anomaly-detection metrics (precision/recall at chosen threshold)
- **quality_score regression:** MAE against your synthetic ground truth
- **Explainability, three cheap wins:**
  1. Feature importance / correlation per prediction (attach top-3 contributing features to each detected issue in the API response)
  2. Reconstruction-error heatmap overlay, upsampled to original resolution
  3. Confidence = the MLP sigmoid probability, already computed
- **Failure case analysis:** pick 5–10 borderline/misclassified examples and write a short, honest paragraph each (e.g., artistic intentional blur vs. genuine blur; a dark scene vs. true underexposure). This section alone is worth writing carefully — it's explicitly requested and most applicants under time pressure skip it.

**Deliverable:** `ml/evaluation_report.md` with metric tables, confusion matrices, and 5–10 annotated failure-case images.

---

## Phase 5 — Backend API (Hour 17 → 27)

Heavily agent-delegable — you just need to review, not hand-write, most of this.

### 5.1 Structure
```
backend/app/
  main.py
  routers/analysis.py
  models/db_models.py
  schemas/schemas.py
  services/inference.py
  db/session.py
```

### 5.2 Database schema
`AnalysisResult`: `id, filename, upload_time, quality_score, quality_label, issues (JSON), features (JSON), heatmap_path (nullable), image_path`

### 5.3 Endpoints
- `POST /api/analyze` — multipart upload → validate (format, size, corrupt-file check via PIL `verify()`) → extract features → run Model A + Model B → persist → return structured JSON matching the spec's example shape
- `GET /api/results` — paginated history list
- `GET /api/results/{id}` — full detail incl. image reference and heatmap if present
- `GET /api/health` — status + "model loaded" boolean

### 5.4 Robustness
- Explicit status codes: 400 invalid file type, 413 too large, 422 validation error, 500 with a clean error body — never a raw traceback
- Load both models once at FastAPI startup (`lifespan` context), not per-request
- CORS origins from an environment variable
- A handful of `pytest` tests for the analyze/health/results endpoints (cheap bonus points)

**Deliverable:** working FastAPI service, passes a manual Postman/curl smoke test for every endpoint including deliberately bad inputs.

---

## Phase 6 — Frontend (Hour 27 → 35)

Also heavily agent-delegable. Functionality and state-handling matter far more than visual polish per the spec — don't over-invest in styling.

- **Upload page:** drag-and-drop + file picker, image preview, "Analyze" button, loading spinner during inference
- **Result page:** quality score display (badge/gauge), issue list with severity + confidence chips, image statistics table (the raw features), heatmap overlay toggle if Model B's heatmap is available
- **History page:** grid/table of past analyses with thumbnails, click-through to full detail, basic pagination
- **API layer:** a thin fetch/axios wrapper, with distinct loading / success / error UI states everywhere a request happens
- Responsive layout with a simple, consistent style — no need for a design system

**Deliverable:** a usable 3-page React app wired to the live backend.

---

## Phase 7 — Integration Testing (Hour 35 → 38)

- Manually upload: a clean sharp image, a blurry one, over/under-exposed, noisy, a genuinely corrupted file, a non-image file, an oversized file — confirm every path returns sane output or a clean error
- Restart the backend/DB and confirm history persists
- Cross-check every field in the backend's JSON response actually renders somewhere in the frontend
- Fix whatever breaks — budget this hour for bugs, not new features

---

## Phase 8 — Dockerization & Deployment (Hour 38 → 43)

- **Backend Dockerfile:** `python:3.11-slim`, install CPU-only torch wheel (smaller, faster build), copy app + saved model checkpoints
- **Frontend Dockerfile:** multi-stage — `node` build stage → `nginx` (or `vite preview`) serve stage
- **docker-compose.yml:** `backend`, `frontend`, `db` (postgres:16-alpine) services; volumes for uploaded images and Postgres data; `.env` for `DATABASE_URL`, `MODEL_PATH`, `CORS_ORIGINS`, `MAX_UPLOAD_SIZE`
- Add `healthcheck:` directives so compose reports service health, not just "running"
- **Critical test:** `docker compose up` from a completely fresh clone, on its own, with zero manual fixes. This is what "reproducibility" (10%) is actually graded on.
- Cloud deployment (Render/Railway/Fly.io) is optional per the spec — only attempt it if Phases 0–8 finish with room to spare.

---

## Phase 9 — Documentation & Submission Packaging (Hour 43 → 47)

- **README.md:** setup instructions, architecture overview (a simple diagram is fine, even ASCII), model/training explanation (paraphrase your Phase 3 write-up), API docs with example `curl` requests and example JSON responses, database setup, deployment instructions
- Fold your Phase 4 evaluation report (or a summary + link) into the README
- `docs/sample_images/` — a small curated set showing each quality condition (clean, blurry, under/over-exposed, noisy, corrupted, defective)
- Repo cleanup: remove dead code, commented-out experiments, secrets, and any oversized generated files
- Optional but high-value: a 2–3 minute screen recording of upload → result → history, linked in the README

---

## Phase 10 — Final Buffer & Submission (Hour 47 → 48)

- Fresh clone on a clean machine (or ask someone else to try) and re-run `docker compose up` one last time
- Walk the Section 12 submission checklist line by line and confirm every item exists in the repo
- Submit

---

## Risk Mitigation — Where Applicants Usually Lose Points

| Risk | Mitigation |
|---|---|
| "Working code, but I can't explain the model" | You personally write Phase 3's architecture paragraph and Phase 4's failure-case analysis — never fully delegate these |
| CV-only solution mistaken for sufficient | Two real learned models (MLP + autoencoder) satisfies the explicit "not CV-only" requirement |
| Fails to run outside your machine | Phase 8's fresh-clone test is non-negotiable, do it twice |
| Inflated/dishonest accuracy claims | Report the real numbers, including where the model is wrong (Phase 4) — reviewers trust honest 85% over suspicious 99% |
| Missing edge-case handling | Phase 7's deliberate bad-input testing (corrupt file, wrong type, oversized) |
| Running out of time on plumbing | Delegate backend/frontend boilerplate to Antigravity aggressively; reserve your own attention for Phases 1–4 |

---

## Optional Bonus Work (only if ahead of schedule, in priority order)

1. Quality heatmap / localization — **already nearly free** from Model B's reconstruction error map
2. Confidence calibration — already partially free from the MLP sigmoid outputs; add a calibration curve to the eval report
3. Basic pytest suite for backend (cheap, already scoped in Phase 5.4)
4. Batch analysis endpoint (accept multiple files in one request)
5. Simple CI workflow (GitHub Actions running pytest on push)
6. Basic request logging / monitoring middleware

---

## Final Submission Checklist (mirrors Section 12 of the assessment)

- [ ] Complete source: frontend, backend, ML/AI components
- [ ] README: setup, model/training, API, deployment instructions
- [ ] Database setup instructions
- [ ] API documentation / example requests
- [ ] Evaluation results + technical explanation
- [ ] Sample images across quality conditions
- [ ] Docker / Docker Compose configuration
- [ ] Deployed URL (if applicable — optional)
