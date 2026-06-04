# 3D-Vizualize — Roadmap

> Versions are sized by scope, not calendar. Each version ships only when its acceptance criteria pass.

---

## v1.0 — MVP (single-representation 3DGS)

**Goal**: validate WOW-effect, attract investment, prove core hypothesis.

Scope: see [SPEC.md](SPEC.md). Single representation (pure 3DGS + compression), interior only, passive viewer.

**Exit criteria**: SPEC §10 acceptance criteria all green; ≥ 50 real captures from beta partners; blind test ≥ 80% "looks real".

---

## v1.5 — Service layer + outdoor

**Theme**: turn the renderer into a *product*. Real-estate agents need more than a passive viewer.

### Capture
- Outdoor zones: facade + yard
- Separate lighting handling (sun, shadows, sky removal)
- Per-zone session continuity (re-enter and continue)
- ML-driven coverage scoring with "smart next-direction" hints during capture

### Viewer features (service mesh activated)
- **Measurements**: tap two points → distance, area, ceiling height
- **Floor plan**: auto-extracted top-down 2D view, switch between 3D and 2D
- **Navigation hotspots**: room-to-room teleports, named rooms
- **Embed analytics**: dwell time, rooms viewed, hotspot taps → postMessage events for Cian/Avito

### Pipeline
- Outdoor 3DGS training profile (different regularization, sky masking)
- Service mesh becomes first-class: keyed measurements, room labels, portal graph

**Exit criteria**: 5 real listings on Cian/Avito with embedded viewer + measurement working; outdoor scenes pass §10 quality.

---

## v2.0 — Adaptive hybrid representation

**Theme**: scale economics. When you have 10k+ listings, asset size and cost per scan dominate.

### Per-zone representation selection
- Room-type classifier + surface complexity + reflectivity → pick one of:
  - **Textured low-poly mesh** (hallways, empty rooms, closets) — 0.3–1 MB/zone
  - **2D Gaussian Splatting** (diffuse rooms) — 2–5 MB/zone
  - **Full 3DGS** (kitchens, living rooms, bathrooms) — 6–15 MB/zone
- Scene format v2: zone graph + portal connections
- Boundary handling: overlap zones, shared training, natural door-frame occluders
- Target: ×2–3 size reduction without loss of WOW in hero zones

### Mirror rendering subsystem
- LiDAR-based mirror plane + contour extraction
- Runtime virtual-camera reflection (not splat-baked)
- No more fantom splats behind glass
- Same approach for shiny tile floors (planar reflection)

### Platform expansion
- Non-LiDAR iPhones via full COLMAP/GLOMAP fallback (longer processing time accepted)
- Android capture (ARCore depth API as LiDAR substitute on Pixel/Samsung devices)

### Viewer
- Hybrid render path (mesh + 2DGS + 3DGS in same scene, depth-sorted)
- Adaptive LOD: full-detail current room + immediately adjacent, lite for far rooms
- Streaming: load zones as user moves through portals

**Exit criteria**: average scene ≤ 20 MB; non-LiDAR capture quality within 1 PSNR of LiDAR; Android beta active.

---

## v3.0 — Streaming & scale

**Theme**: commercial real estate, hotels, large properties. Datasets that don't fit in single asset.

- LOD streaming protocol (WebTransport or HTTP range-streamed) for large properties
- Multi-LOD per zone (LightGaussian + foveated rendering)
- WebGPU compute paths for client-side splat decoding (offload from CPU)
- AR mode: open captured space in iOS/Android AR overlay — uses original LiDAR coordinate frame
- B2B annotation: agent adds info pins, virtual staging hook points
- Analytics dashboard for B2B partners (engagement heatmaps over the 3D space)

---

## v4.0 — AI-assistive features

**Theme**: differentiate beyond visualization — make it *interactive intelligence*.

- **Virtual staging**: select furniture in viewer → diffusion model replaces with alternative styles (modern / scandi / classic)
- **Repaint preview**: tap a wall → "see this room in [color]" — AI surface relighting
- **Natural-language search**: "show me apartments with bright kitchen and balcony" — uses 3D scene as embedding source
- **Auto-tagging**: room types, materials, furniture inventory extracted automatically for listing metadata
- **Comparable-property suggestions**: 3D similarity search across the catalog
- **Renovation visualization**: upload renovation plan → AI projects after-state onto current scene

---

## Continuous tracks (parallel to versions)

These don't ship as features — they're ongoing investments running alongside every version.

### Cost & throughput
- GPU minutes per scan target: -30% each major version
- Spot/preemptible instance strategy
- gsplat fork with custom optimizations

### Quality benchmarks
- Blind A/B vs. Matterport, vs. photo tours, vs. video tours
- Internal PSNR / SSIM / floater metrics in CI
- Real-device performance regression suite

### Compliance & legal
- 152-FZ (RU), GDPR (EU), CCPA (US) — face/document blur, takedown SLAs
- ToS evolution as B2B contracts expand
- Asset ownership / licensing clarity for tenant-uploaded content

### Capture UX research
- Studies with non-expert users (homeowners, not just photographers)
- Reduce time-to-good-capture (target: untrained user achieves §10 quality in v2)
