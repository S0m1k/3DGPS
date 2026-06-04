# Beta v0.1 — video-first 3DGS validation

**Goal**: validate that ordinary smartphone video → photorealistic 3DGS scene is viable for real-estate use case, without relying on LiDAR.

**Hypothesis to prove or kill**:
- Quality from video-only pipeline is within -1 PSNR of LiDAR-based baseline
- Time-to-result on RTX 3070 Ti is bearable for development (≤ 2h per room)
- WOW-effect is present (subjective: scene "feels real" to 3+ blind viewers)

**Hardware**: RTX 3070 Ti (8 GB), Windows 11, iPhone 13 Pro Max for capture.

---

## Phases

### Phase 1 — PC setup
- See [`pipeline/env.yml`](pipeline/env.yml)
- Smoke-test on Mip-NeRF360 "garden" public dataset

### Phase 2 — record test videos
- See [`docs/capture-guide.md`](docs/capture-guide.md)
- 3 rooms: bedroom (diffuse) / living room (medium) / kitchen (hard)
- Plain iPhone Camera app, 4K @ 30fps, ~3 min each
- Save to `captures/<room>/video.mp4`

### Phase 3 — baseline (video + COLMAP + 3DGS)
- Run [`pipeline/run_baseline.sh <room>`](pipeline/run_baseline.sh)
- Uses `ns-process-data video` (COLMAP under hood) + `ns-train splatfacto`
- Output: `results/<room>/baseline.ply` + metrics row in `results/metrics.csv`

### Phase 4 — foundation-model pose estimation
- Replace COLMAP with LingBot-Map (or VGGT as fallback)
- See [`pipeline/run_lingbot.sh <room>`](pipeline/run_lingbot.sh) — currently scaffolded, needs verification against actual LingBot-Map CLI
- Compare against baseline on same video input

### Phase 5 — smart frame selection
- Naive every-Nth vs blur-filtered + motion-based
- See [`pipeline/extract_frames.py`](pipeline/extract_frames.py)

### Phase 6 — go/no-go
- Review [`results/metrics.csv`](results/metrics.csv) + subjective notes in [`results/notes.md`](results/notes.md)
- Decision documented in `../CLAUDE.md` and (if pivot confirmed) SPEC.md rewrite kicks off

---

## Quick start

```bash
# 1. install env (one-time)
conda env create -f pipeline/env.yml
conda activate splat-beta

# 2. launch control panel (recommended)
streamlit run ui/app.py
# opens http://localhost:8501 — manage captures, runs, metrics from here

# OR work via CLI:
# 2a. record video on iPhone, transfer to captures/bedroom/video.mp4
# 2b. bash pipeline/run_baseline.sh bedroom
# 2c. drag-drop results/bedroom/baseline.ply into https://playcanvas.com/supersplat/viewer
```

## Control panel

`streamlit run ui/app.py` запускает локальный веб-интерфейс на http://localhost:8501.

Что в нём:
- **Комнаты** — список с автоопределением статуса (video / processed / trained / .ply), кнопки запуска baseline и LingBot прямо из UI, live-логи прогонов
- **Метрики** — таблица из `results/metrics.csv` с фильтрами + сравнение методов на одной комнате
- **Заметки** — редактирование `results/notes.md` прямо в браузере
- **Подсказки** — встроенная справка по workflow

---

## Metrics tracked per run

See [`results/metrics.csv`](results/metrics.csv) for schema. Each row = one run.

Key fields: room, method (`baseline_colmap` / `lingbot` / `vggt`), num_frames_extracted, train_time_sec, final_psnr, splat_count, file_size_mb, subjective_score (1–10), notes.

---

## Out of scope for beta

- iOS custom app (deferred until pivot decision)
- Backend orchestrator
- Compression beyond gsplat default
- Web viewer development (using public supersplat viewer)
- Mirrors / masks / PII
- Multi-room scenes (one room at a time)
- Outdoor / facade
