# Capture guide — iPhone 13 Pro Max video for beta v0.1

The whole point of video-first beta is **no app required**. Use the built-in iOS Camera. But a sloppy walkthrough gives a sloppy 3D scene. Follow this.

## Settings (Camera app)

1. Open Settings → Camera → Record Video → **4K at 30 fps** (NOT 60, NOT 24)
2. Settings → Camera → Formats → **High Efficiency** (HEVC, smaller files)
3. Settings → Camera → Preserve Settings → enable everything (so it doesn't reset)
4. Lock exposure: in Camera, long-press on a wall → "AE/AF LOCK". This prevents auto-exposure from changing mid-walk. Critical.

## Lighting

- **Daylight, overcast preferred.** Direct sunlight through windows creates hard shadows that bake into the 3D model.
- All lights on, blinds in their final position. Don't change anything mid-capture.
- Avoid backlight (camera pointed at bright window) — sensor clips.

## Movement

- **Walk slowly.** Real slow. ~0.3 m/s. Speed = motion blur = bad frames discarded by `extract_frames.py`.
- **Hold steady.** Two hands. Keep phone roughly horizontal.
- **Don't twist the phone.** Move your body, not your wrist. Twisting makes pose estimation harder.

## Path through the room

For a 25 m² room, ~3 minutes is plenty:

```
1. Stand in doorway. Face the room. Record start.
2. Pan slowly L→R covering the whole room from this entry view (~10 sec).
3. Walk along the right wall, camera pointed at the opposite (left) wall and middle of room.
4. At far corner, pause, pan around 360° slowly (~15 sec).
5. Walk along far wall back to left, camera pointed at right wall.
6. Walk back toward doorway along left wall.
7. Bonus: stand in middle, pan 360° at three different heights (low, eye-level, high).
8. End by recording the ceiling for ~5 sec (often forgotten).
```

**Key principle**: every surface should be visible from at least 3 different angles. Glossy surfaces (TV, mirror-finish appliances) need more like 5–6 angles, otherwise 3DGS can't figure out the reflection properly.

## What to AVOID for beta

- Mirrors — exclude rooms with mirrors for now (or cover mirrors with a sheet)
- Moving things during capture — pets, family, you yourself walking past the camera
- Recording reflections of yourself in windows / glossy surfaces (move sideways)
- Auto-exposure drift (locked in step 4 above)
- Camera pointed straight up or down for long — disorients pose estimation

## Transfer to PC

Easiest:
- AirDrop to a Mac if available
- USB cable + Windows Photos app
- iCloud Photos → download from iCloud.com on the PC
- Send via Telegram "Saved Messages" at original quality (skip compression: "Send as File")

Save to `beta_v0.1/captures/<room>/video.mp4`.

## Test rooms for beta

| Room | Difficulty | Why |
|--|--|--|
| bedroom | easy | Diffuse surfaces (textile, paint, wood), no mirrors, predictable lighting |
| living-room | medium | Likely has TV (glossy), maybe glass coffee table, more clutter |
| kitchen | hard | Chrome appliances, glass cabinets, induction, glossy backsplash — stress test |

Capture in that order. If bedroom fails the beta hypothesis, the harder ones won't save it.

## Quick quality check before processing

After recording, scrub through the video. Reject and reshoot if:
- More than 20% of frames look motion-blurred
- Lighting changed visibly mid-recording (someone opened curtains, sun moved)
- Camera was pointed at floor / ceiling for big chunks
- You appear in any mirror or reflection
- Less than 2 minutes long

A 3-minute steady walk gives ~5400 raw frames. After blur filter (~30% drop) and even subsample (300 frames), you get a great training set.
