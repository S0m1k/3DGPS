# 3D-Vizualize — Spec v1 (MVP)

> Status: **draft, under reconsideration**.
> Reference codebase studied: [playcanvas/supersplat](https://github.com/playcanvas/supersplat).
>
> ⚠️ **2026-06-01 — video-first hypothesis under validation.** Current SPEC is LiDAR-first (iPhone Pro). Beta v0.1 (`beta_v0.1/`) tests whether video-only capture + foundation-model pose estimation (LingBot-Map / VGGT) can replace LiDAR with acceptable quality. If beta confirms, this SPEC will be rewritten: video as primary capture, web upload as primary UX, iOS Pro mode demoted to v1.5. Until beta data lands, treat this SPEC as the fallback path.

---

## 1. Product

Web-based photorealistic 3D viewer of residential property, captured from iPhone/iPad with LiDAR, embeddable in real-estate listings.

**Target users**:
- B2B platforms: Cian, Avito (embed in listing cards)
- Real-estate developers (застройщики)
- Private sellers / landlords / renters

**Value proposition**: WOW-grade photorealism — "you're actually inside the apartment" — runs in any browser, no install. Beats photo galleries and video tours on first impression.

---

## 2. Confirmed constraints

- **Quality > optimization**. WOW-effect must be visible on first frame.
- **Backend GPU servers are acceptable**. No on-device fitting requirement.
- **ML team is available**.
- **Primary capture surface**: iPhone Pro / iPad Pro with LiDAR (iPhone 12 Pro+).
- **Embed target**: ≤ 50 MB per apartment scene for B2B partner integrations.

---

## 3. Open assumptions (need confirmation)

These were not finalized in pre-spec discussion. v1 proceeds with the assumption shown. **Review and correct before implementation kicks off.**

| # | Assumption | Why it matters |
|--|--|--|
| A1 | Capture-to-view latency target = **30 min** wall time | Drives GPU instance type and batch sizing |
| A2 | Initial target size = **30–80 m² apartments**, upper bound 200 m² house | Affects training time, asset size budget, capture UX |
| A3 | Content protection v1 = **signed-URL CDN**, no DRM. File downloadable to client | Cheapest, fastest to ship. B2B may demand more later |
| A4 | Capture scope v1 = **interior only**. Facade + yard → v1.5 | Outdoor needs different lighting handling, more capture time |
| A5 | MVP viewer features = **passive view only** (orbit, walk, zoom). Hotspots / measure / comments → v1.5 | Smallest possible MVP for hypothesis validation |

---

## 4. Architecture overview

```
┌─────────────────┐   ┌──────────────────────┐   ┌─────────────────┐
│  iOS Capture    │   │  Orchestrator +      │   │   Web Viewer    │
│  app            │──▶│  GPU Pipeline        │──▶│   (embeddable)  │
│  (LiDAR + RGB)  │   │  (backend)           │   │                 │
└─────────────────┘   └──────────────────────┘   └─────────────────┘
        │                       │                         │
        ▼                       ▼                         ▼
   Capture bundle          Asset registry            CDN-served
   (frames, poses,         (versions, metadata,      .compressed.ply
    LiDAR mesh,            QA reports)               + service mesh
    masks, coverage)                                 + scene.json
```

Three components + control plane:

1. **iOS Capture app** — collects raw data with live coverage feedback
2. **Orchestrator** — upload, queue, state machine, asset registry, auth/tenant
3. **GPU Pipeline** — staged processing (validation → masking → training → QA → compression)
4. **Web Viewer** — embeddable iframe with stable postMessage API

---

## 5. Component 1: iOS Capture app

### Capabilities
- ARKit `ARWorldTrackingConfiguration` with `sceneReconstruction = .mesh`
- Synchronized capture: RGB (2–5 fps) + camera pose (ARKit) + LiDAR mesh + LiDAR depth (optional)
- Room-type classifier (Places365 CoreML) → initial budget estimate shown to user
- **Live coverage heatmap** (see §5.2) — the central UX feature
- Glossy-surface detector → mirror-protocol micro-instructions
- Resume capture (recover from app crash mid-session, persist coverage state)
- Capture-time validation: blur score per frame (motion-blurred frames dropped), exposure consistency warning
- PII warning prompt: app surfaces "we detected faces / documents — review before upload" (mandatory under 152-FZ)
- Manual mirror masking: tap-to-mark mirror surfaces (assists backend)
- Upload: resumable, chunked, compressed (HEVC for RGB stream, zlib for depth)

### 5.2 Coverage tracking data structure

Per architect review — **voxelized angular bins**, not per-triangle:

```swift
struct CoverageVoxel {
  position: SIMD3<Float>      // 5–10 cm grid cell center
  normalEstimate: SIMD3<Float>
  viewedAngles: BitSet64      // 42 bins ≈ icosphere subdivision over hemisphere
}
```

- Voxelize ARKit mesh into 5–10 cm grid
- 42 angular bins per voxel covers the upper hemisphere
- "Covered" = ≥ 3 bins hit with ±cos(20°) cone diversity
- Heatmap color in AR overlay = `bins_filled / 42`
- "Finish capture" button disabled until > 90% voxels covered
- Persisted to disk as `Float16[]` for resume-after-crash

### Capture session output (bundle uploaded to backend)
```
session_<uuid>/
  frames/*.heic          # RGB photos
  poses.json             # per-frame camera intrinsics + extrinsics (ARKit)
  mesh.obj               # LiDAR mesh
  depth/*.bin            # sparse depth maps (optional)
  masks/mirrors.json     # user-tagged mirror polygons
  coverage.bin           # voxel coverage state
  metadata.json          # room labels, device info, capture time
```

---

## 6. Component 2: Orchestrator (control plane)

Often missed — explicitly broken out per architect review.

**Responsibilities**:
- Resumable multi-GB upload service
- Job queue + state machine: `uploaded → validating → masking → training → qa → compressing → ready | failed_<stage>`
- Asset registry: `apartment_id → version[] → (cdn_urls, service_mesh_url, qa_report, metadata)`
- Auth & tenant model: developer accounts (B2B, higher SLA) vs. private sellers (B2C)
- Operator QA tool: preview captures *before* gating them to expensive GPU training
- Cost accounting per tenant per stage
- Retention policy: see §11 (storage)

**Stack**: Python 3.12 / FastAPI / Celery + Redis / Postgres for registry / S3-compatible storage.

---

## 7. Component 3: GPU pipeline (staged)

Pipeline order incorporates architect-suggested stages:

| # | Stage | Purpose | Failure mode |
|--|--|--|--|
| 1 | **Capture validation** | Coverage %, blur score, exposure drift, frame count. Fail fast before GPU spend | Reject session, request reshoot |
| 2 | **Pose refinement** | ARKit poses are good but not perfect. Try **GLOMAP** or hloc; fall back to COLMAP if needed. Tunable, not hardcoded | Bad poses → bad splats |
| 3 | **Mask generation** | Sky / window / mirror semantic segmentation (SAM2 or custom) — feeds loss masking | Skipped masks → fantom splats behind glass |
| 4 | **3DGS training** | `gsplat` (PyTorch). Loss masking via §3 masks. ~15–40 min on A100 for 50 m² | Floater "popcorn" artifacts, mode collapse |
| 5 | **Auto-QA** | Novel-view PSNR on held-out frames, floater detection, mirror artifact check | Below threshold → flag for operator review |
| 6 | **Compression** | LightGaussian pruning + SH quantization → PlayCanvas `.compressed.ply` | Target ≤ 50 MB/scene |
| 7 | **Service mesh extraction** *(parallel with 6)* | Simplified low-poly mesh from LiDAR (or Poisson from splat cloud) for measurements + navigation | Independent — can re-run without retraining |
| 8 | **Asset packaging** | `scene.json` + compressed splat + service mesh + thumbnail → CDN | Bundle integrity check |

**Determinism note**: gsplat training is non-deterministic across runs. Lock seeds, version weights, snapshot inputs. Two re-trainings should land within ±0.5 PSNR (acceptance §10).

---

## 8. Component 4: Web Viewer

### Engine
- PlayCanvas Engine 2.x + GSplat renderer (basis from `supersplat/src/render.ts`)
- TypeScript + PCUI + Rollup
- WebGPU primary, WebGL2 fallback

### Controls
- Orbit, fly, click-to-navigate via service mesh (raycast on click → animate camera)
- Smart camera: stays inside walls (collision against service mesh)

### Embedding contract (bake into v1 to avoid breaking changes later)

**Iframe-only**. Never inject scripts into host. Host integrates as:
```html
<iframe src="https://viewer.3d-vizualize.com/v/<token>"
        allow="fullscreen; xr-spatial-tracking"
        sandbox="allow-scripts allow-same-origin"></iframe>
```

**postMessage API v1** (frozen schema):
```js
{ v: 1, type: 'ready' | 'loaded' | 'error' | 'view_changed' | 'room_entered' | 'analytics',
  payload: {...} }
```

- Lazy-load: don't fetch splat until iframe in viewport (IntersectionObserver)
- Domain allowlist per asset (signed URL Origin/Referer check)
- Analytics events for host: dwell time, rooms viewed, hotspot taps — Cian/Avito will demand this
- CSP-compatible: no `eval`, document any required PlayCanvas exceptions

### Performance budgets
- Initial scene ≤ 50 MB
- Time-to-first-pixel ≤ 3 sec on 4G + iPhone 12
- Time-to-interactive ≤ 8 sec
- Sustained FPS ≥ 30 on iPhone 12 / mid-range Android
- Memory cap ≤ 800 MB

---

## 9. Tech stack summary

| Layer | Choice |
|--|--|
| iOS | Swift 6, ARKit, RealityKit, CoreML, Combine, swift-protobuf |
| Orchestrator | Python 3.12, FastAPI, Celery + Redis, Postgres, S3-compatible storage |
| GPU pipeline | PyTorch 2.x, gsplat, GLOMAP/COLMAP, SAM2 (masks), Poisson recon |
| GPU infra | A100 / L40S / RTX 6000 Ada (spot instances for cost) |
| Web viewer | TypeScript, PlayCanvas Engine 2.x, PCUI, Rollup, deployed as static SPA + CDN |
| CDN | CloudFront / BunnyCDN / Cloudflare R2 (cost-driven) |

---

## 10. Acceptance criteria (v1 GA)

**Functional**:
1. Capture: 50 m² apartment scanned in ≤ 15 min by a trained user
2. Processing: ≤ 30 min wall time from upload finish to viewer link (assumption A1)
3. Viewer cold-cache time-to-first-pixel ≤ 3 sec, time-to-interactive ≤ 8 sec on 4G + iPhone 12
4. Sustained FPS ≥ 30 on iPhone 12 and mid-range Android; memory ≤ 800 MB
5. Session completion rate ≥ 95% (no manual intervention)
6. Capture failure recovery: ≥ 90% of sessions recoverable after mid-session app crash

**Quality**:
7. Blind user test: ≥ 80% of respondents say "looks real" or better
8. Floater artifact count < threshold per m³ (automated; threshold calibrated during alpha)
9. Mirror / window behavior: no visible duplication artifacts on held-out test set
10. Reproducibility: two re-trainings of same capture within ±0.5 PSNR

**Cross-browser matrix** (each must pass §10.3 and §10.4):
- Safari iOS 17+
- Chrome Android (current + 2 prior)
- Chrome / Firefox / Safari desktop (current)

---

## 11. Risks (catalog)

| # | Risk | Severity | Mitigation |
|--|--|--|--|
| R1 | Capture UX failure (users miss angles → broken models) | 🔴 High | Coverage heatmap, ML coverage scoring, finish-button gating |
| R2 | Mirror artifacts (fantom splats behind reflective surfaces) | 🔴 High | Manual mask in app + auto-mask in pipeline + post-training cleanup. Real reflection rendering → v2 |
| R3 | Floater / popcorn artifacts on flat walls and ceilings | 🟠 Medium | Mask-aware training, auto-QA gate, regularization |
| R4 | HDR / exposure drift across rooms (window vs. dark hallway) | 🟠 Medium | Lock exposure during capture, warn user on drift, color-correction pass pre-training |
| R5 | iPhone thermal throttling during long sessions | 🟡 Medium | Split into per-room sessions, monitor `thermalState` |
| R6 | gsplat training non-determinism | 🟡 Medium | Seed locking, version pinning, reproducibility test in CI |
| R7 | GPU cost per scan × N scans / month | 🟠 Medium | Spot instances, batched off-peak, gsplat optimization |
| R8 | Browser fragmentation, low-end devices | 🟠 Medium | Static panorama fallback, device detection, low-quality LOD |
| R9 | iOS LiDAR penetration (~15% of iPhones) | 🟡 Medium | B2B-first focus (agents have Pro), v2 = non-LiDAR + Android |
| R10 | **PII in captures** (faces, documents, screens) | 🔴 High | Auto-blur pass + user review prompt before upload (152-FZ compliance) |
| R11 | Storage cost trajectory (50 MB × scenes × versions) | 🟡 Low-Med | Retention policy: hot 90 days, cold tier after, delete on tenant request |
| R12 | Legal: 3D asset ownership ambiguity | 🟠 Medium | Explicit ToS — platform-licensed, homeowner-owned, takedown workflow |

---

## 12. Out of scope for v1

See [ROADMAP.md](ROADMAP.md). Briefly:
- Adaptive hybrid representation (mesh / 2DGS / 3DGS per zone) — v2
- Real mirror reflection rendering — v2
- Outdoor capture (facade, yard) — v1.5
- Hotspots, measurements, comments — v1.5
- AR view ("step into your apartment via phone") — v2
- Non-LiDAR iPhone support + Android — v2

---

## 13. Critical follow-up documents (to be authored)

- `docs/pipeline.md` — stage-by-stage contracts (input format, output format, failure modes per stage)
- `docs/viewer-embed-api.md` — postMessage schema v1 (frozen contract), CSP requirements, host integration recipes
- `docs/ios-coverage.md` — voxel + angular-bin spec, persistence format, resume semantics
- `docs/asset-format.md` — `scene.json` schema, versioning, service-mesh format
