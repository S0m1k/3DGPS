# Beta v0.1 — open questions for the owner

> Collected by the autonomous dev session on **2026-06-04** while you were out.
> Code was written by Sonnet, reviewed by Opus. Nothing here blocks the committed
> work — these are decisions/verifications that need your hardware, accounts, or call.
> Grouped by urgency.

---

## 🔴 Blocks running anything (need your machine)

1. **Push access.** This sandbox cannot reach GitHub over git (port 443 blocked for
   `git push`, though plain HTTP fetch works). All work is committed locally to branch
   `sprint/beta-pipeline-2026-06-04`. **You need to `git push -u origin
   sprint/beta-pipeline-2026-06-04`** from your machine, then open a PR. The local repo
   lives at `C:\Users\somov\3DGPS` (freshly cloned; it's a nested git repo inside your
   home dir — that's fine, just never `git add -A` from `C:\Users\somov`).

2. **Conda env not created / not verified.** `pipeline/env.yml` has never been built on
   this machine. Before any pipeline run: `conda env create -f pipeline/env.yml` then
   `conda activate splat-beta`. On Windows, COLMAP must be installed separately and put on
   PATH (the env file's conda `colmap` package is Linux/Mac only — see note in env.yml).
   → Run the new `pipeline/smoke_test.*` first; it validates the whole env on a public
   dataset before you waste a capture on it.

3. **`bash` on PATH for the UI.** `ui/app.py` launches the `.sh` scripts via `["bash", ...]`.
   On Windows that needs Git Bash (or WSL) on PATH. Confirm `bash --version` works in the
   same shell you launch Streamlit from, or the Run buttons silently fail.

---

## 🟠 LingBot-Map path (Phase 4) — written against docs, NOT yet run

The real LingBot-Map interface was confirmed from the official repo
(github.com/Robbyant/lingbot-map, Ant Group, arXiv:2604.14141). The adapter and updated
`run_lingbot.sh` target the **documented** output format. Verify these on first real run:

4. **Model weights.** Which checkpoint? Repo ships `lingbot-map.pt` and `lingbot-map-long.pt`
   (HuggingFace `robbyant/lingbot-map`). For 3-min room walkthroughs (>500 frames) the docs
   recommend `-long.pt` + `--mode windowed`. The script defaults to that — confirm the path
   you download to and set `LINGBOT_MODEL` / `LINGBOT_REPO` env vars accordingly.

5. **`--save_predictions` output layout.** Docs say it "persists per-frame NPZs alongside
   the MP4" but do not document the exact filenames or the in-NPZ keys. The adapter assumes
   keys `extrinsic` (cam-to-world 3×4), `intrinsic` (3×3), `world_points`, `world_points_conf`
   per the `predictions` dict in `demo.py`. **If the NPZ keys differ, fix the constants at the
   top of `lingbot_to_nerfstudio.py`** (they're isolated there for exactly this reason).

6. **Coordinate convention.** Nerfstudio wants `transform_matrix` in OpenGL convention
   (cam looks −Z, +Y up). LingBot/VGGT extrinsics are typically OpenCV (+Z, −Y). The adapter
   applies the OpenCV→OpenGL flip (`c2w[:,1:3] *= -1`). **If the splat comes out mirrored or
   inside-out, this flip is the first thing to toggle** (`--no-opengl-flip`).

7. **LingBot vs VGGT.** README's Phase 4 lists VGGT as fallback. VGGT→splatfacto is a known
   path but reportedly weak on high-frequency detail (nerfstudio issue #3672). Do you want a
   parallel `run_vggt.sh`, or is LingBot the only Phase-4 contender for now? (Not built yet.)

---

## 🟡 Tuning / judgement calls (sensible defaults chosen)

8. **Frame budget = 300.** Both scripts use `--num-frames-target 300` / `--target-count 300`.
   Fine for 3-min/25 m². Larger rooms may want more. Left as-is.

9. **`extract_frames.py` new `--strategy`.** Added `blur+motion` (default stays `blur` for
   backward compat). Motion gate drops near-duplicate frames when you stood still. Thresholds
   are heuristic — may need tuning once you see real extraction counts. See its `--help`.

10. **Blur threshold 100.** Laplacian-variance cutoff. Comment says sharp indoor frames are
    200–800, motion blur <50. 100 is conservative. Tune after the first extraction prints
    its "sharp frames: X / Y" line.

11. **Seed locking.** `run_baseline.sh` now pins `--machine.seed 42` and records it. Note
    gsplat is only *approximately* deterministic even with a fixed seed (CUDA atomics);
    SPEC §7 accepts ±0.5 PSNR. The reproducibility check is documented, not automated in CI yet.

---

## 🟢 Backlog / not started (need your steer)

12. SPEC §13 follow-up docs (`docs/pipeline.md`, `viewer-embed-api.md`, `ios-coverage.md`,
    `asset-format.md`) — these are for the **product** SPEC, not the beta. The beta's
    go/no-go (Phase 6) gates whether the SPEC even stays LiDAR-first. Deliberately left
    until the pivot decision. Confirm you want them deferred.

13. No automated PSNR/quality CI yet (Continuous track "Quality benchmarks"). The beta is
    too early; metrics are logged by hand via the new `append_metrics.py`. Flag if you want
    CI sooner.
