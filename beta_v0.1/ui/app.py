"""Beta v0.1 control panel.

Run:  streamlit run ui/app.py    (from beta_v0.1/)
"""
from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------- paths ----------
ROOT = Path(__file__).resolve().parent.parent  # beta_v0.1/
CAPTURES = ROOT / "captures"
RESULTS = ROOT / "results"
PIPELINE = ROOT / "pipeline"
DOCS = ROOT / "docs"
METRICS_CSV = RESULTS / "metrics.csv"
NOTES_MD = RESULTS / "notes.md"

st.set_page_config(
    page_title="3D-Vizualize · Beta v0.1",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- helpers ----------
def discover_rooms() -> list[str]:
    """Find rooms in captures/ and results/. Sorted, no dotfiles."""
    rooms: set[str] = set()
    for base in (CAPTURES, RESULTS):
        if base.exists():
            for p in base.iterdir():
                if p.is_dir() and not p.name.startswith("."):
                    rooms.add(p.name)
    return sorted(rooms)


def room_status(room: str) -> dict:
    cap = CAPTURES / room
    res = RESULTS / room
    video = cap / "video.mp4"
    processed = res / "processed"
    train_dir = res / "train"
    baseline_ply = res / "baseline.ply"
    lingbot_ply = res / "lingbot.ply"

    return {
        "has_video": video.exists(),
        "video_path": video,
        "video_size_mb": (video.stat().st_size / 1024 / 1024) if video.exists() else 0.0,
        "processed": processed.exists() and any(processed.iterdir()),
        "trained": train_dir.exists() and any(train_dir.rglob("config.yml")),
        "baseline_ply": baseline_ply.exists(),
        "baseline_ply_path": baseline_ply,
        "baseline_ply_mb": (baseline_ply.stat().st_size / 1024 / 1024) if baseline_ply.exists() else 0.0,
        "lingbot_ply": lingbot_ply.exists(),
        "lingbot_ply_path": lingbot_ply,
        "lingbot_ply_mb": (lingbot_ply.stat().st_size / 1024 / 1024) if lingbot_ply.exists() else 0.0,
    }


def load_metrics() -> pd.DataFrame:
    if not METRICS_CSV.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(METRICS_CSV, comment="#")
    except Exception:
        return pd.DataFrame()


def open_in_os(path: Path) -> None:
    """OS-native reveal in file explorer. Best-effort, never throws to UI."""
    try:
        if not path.exists():
            return
        sysname = platform.system()
        if sysname == "Windows":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sysname == "Darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as e:
        st.warning(f"Could not open: {e}")


def stream_command(cmd: list[str], placeholder) -> int:
    """Run subprocess, stream stdout/stderr into placeholder as code block.

    Returns exit code. Blocks until process exits.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(ROOT),
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ""):
        lines.append(line.rstrip())
        # keep last 80 lines in view to avoid huge re-renders
        placeholder.code("\n".join(lines[-80:]), language="bash")
    proc.wait()
    return proc.returncode


def status_badge(active: bool, label: str) -> str:
    """Inline Markdown badge using Streamlit colored-text syntax."""
    if active:
        return f":green[●] {label}"
    return f":gray[○] {label}"


# ---------- sidebar ----------
with st.sidebar:
    st.title("3D-Vizualize")
    st.caption("Beta v0.1 — video-first")
    st.divider()

    st.markdown("**Pipeline phase**")
    st.markdown(
        "1. PC setup  \n"
        "2. Record video  \n"
        "3. Baseline (COLMAP)  \n"
        "4. LingBot/VGGT poses  \n"
        "5. Smart frame select  \n"
        "6. go/no-go"
    )
    st.divider()

    st.markdown("**Links**")
    st.markdown(
        "- [Capture guide](docs/capture-guide.md)\n"
        "- [Supersplat viewer](https://playcanvas.com/supersplat/viewer)\n"
        "- [Beta README](README.md)"
    )
    st.divider()

    st.markdown("**Hardware**")
    st.caption("RTX 3070 Ti (8 GB) · Win 11 · iPhone 13 Pro Max")

    if st.button("Refresh", use_container_width=True):
        st.rerun()

# ---------- main ----------
st.title("Beta v0.1 — рабочая панель")

tab_rooms, tab_metrics, tab_notes, tab_help = st.tabs(
    ["Комнаты", "Метрики", "Заметки", "Подсказки"]
)

# === Tab: Rooms ===
with tab_rooms:
    rooms = discover_rooms()

    # Create-new-room widget
    with st.expander("Добавить новую комнату", expanded=not rooms):
        cols = st.columns([3, 1])
        new_room = cols[0].text_input(
            "Имя", placeholder="bedroom / living-room / kitchen", label_visibility="collapsed",
        )
        if cols[1].button("Создать", use_container_width=True):
            if new_room.strip():
                (CAPTURES / new_room.strip()).mkdir(parents=True, exist_ok=True)
                st.rerun()
            else:
                st.warning("Пустое имя")

    if not rooms:
        st.info("Пока нет комнат. Создай выше или положи `captures/<room>/video.mp4` вручную.")
    else:
        for room in rooms:
            s = room_status(room)
            with st.container(border=True):
                head_col, status_col, action_col = st.columns([2, 3, 2], vertical_alignment="top")

                with head_col:
                    st.subheader(room)
                    if s["has_video"]:
                        st.caption(f"video.mp4 · {s['video_size_mb']:.0f} MB")
                    else:
                        st.caption("video.mp4 отсутствует")

                with status_col:
                    st.markdown(status_badge(s["has_video"], "Video"))
                    st.markdown(status_badge(s["processed"], "Processed (COLMAP)"))
                    st.markdown(status_badge(s["trained"], "Trained"))
                    if s["baseline_ply"]:
                        st.markdown(f":green[●] Baseline .ply · {s['baseline_ply_mb']:.0f} MB")
                    if s["lingbot_ply"]:
                        st.markdown(f":green[●] LingBot .ply · {s['lingbot_ply_mb']:.0f} MB")

                with action_col:
                    can_run = s["has_video"]
                    is_running = st.session_state.get(f"running_{room}")

                    btn_disabled = (not can_run) or bool(is_running)

                    if st.button(
                        "Run baseline",
                        key=f"btn_baseline_{room}",
                        type="primary",
                        disabled=btn_disabled,
                        use_container_width=True,
                    ):
                        st.session_state[f"running_{room}"] = "baseline"
                        st.rerun()

                    if st.button(
                        "Run LingBot",
                        key=f"btn_lingbot_{room}",
                        disabled=btn_disabled,
                        use_container_width=True,
                    ):
                        st.session_state[f"running_{room}"] = "lingbot"
                        st.rerun()

                    if s["has_video"] and st.button(
                        "Reveal capture folder",
                        key=f"reveal_cap_{room}",
                        use_container_width=True,
                    ):
                        open_in_os(CAPTURES / room)

                    if (s["baseline_ply"] or s["lingbot_ply"]) and st.button(
                        "Reveal results folder",
                        key=f"reveal_res_{room}",
                        use_container_width=True,
                    ):
                        open_in_os(RESULTS / room)

                    if s["baseline_ply"] or s["lingbot_ply"]:
                        st.link_button(
                            "Open Supersplat viewer",
                            "https://playcanvas.com/supersplat/viewer",
                            use_container_width=True,
                            help="Открой ссылку и перетащи .ply из reveal-папки",
                        )

                # Run logs area
                method = st.session_state.get(f"running_{room}")
                if method:
                    st.markdown(f"**Запущен `{method}` для `{room}`** (не закрывай вкладку)")
                    log = st.empty()
                    script = "run_baseline.sh" if method == "baseline" else "run_lingbot.sh"
                    cmd = ["bash", str(PIPELINE / script), room]
                    code = stream_command(cmd, log)
                    st.session_state[f"running_{room}"] = None
                    if code == 0:
                        st.success(f"{method} завершён успешно")
                    else:
                        st.error(f"{method} упал с кодом {code}")
                    if st.button("OK", key=f"ack_{room}_{method}"):
                        st.rerun()

# === Tab: Metrics ===
with tab_metrics:
    df = load_metrics()
    if df.empty:
        st.info(
            "Пока пусто. После прогона добавь строку в `results/metrics.csv` "
            "(см. шаблон в файле)."
        )
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            room_filter = st.multiselect(
                "Комната",
                options=sorted(df["room"].dropna().unique()) if "room" in df.columns else [],
            )
        with col2:
            method_filter = st.multiselect(
                "Метод",
                options=sorted(df["method"].dropna().unique()) if "method" in df.columns else [],
            )
        with col3:
            sort_by = st.selectbox(
                "Сортировка",
                options=list(df.columns),
                index=list(df.columns).index("timestamp") if "timestamp" in df.columns else 0,
            )

        filtered = df.copy()
        if room_filter and "room" in filtered.columns:
            filtered = filtered[filtered["room"].isin(room_filter)]
        if method_filter and "method" in filtered.columns:
            filtered = filtered[filtered["method"].isin(method_filter)]
        if sort_by in filtered.columns:
            filtered = filtered.sort_values(sort_by, ascending=False)

        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.caption(f"Всего записей: {len(df)} · показано: {len(filtered)}")

        # Comparison shortcut: same room, different methods
        if "room" in df.columns and "method" in df.columns:
            st.divider()
            st.markdown("**Сравнение методов на одной комнате**")
            rooms_with_multi = [
                r for r in df["room"].unique()
                if df[df["room"] == r]["method"].nunique() > 1
            ]
            if rooms_with_multi:
                pick = st.selectbox("Комната", rooms_with_multi, key="cmp_room")
                cmp = df[df["room"] == pick].copy()
                key_cols = [c for c in
                            ["method", "final_psnr", "train_time_sec", "splat_count",
                             "file_size_mb", "subjective_score_1to10"]
                            if c in cmp.columns]
                st.dataframe(cmp[key_cols], use_container_width=True, hide_index=True)
            else:
                st.caption("Пока нет комнат с несколькими методами.")

# === Tab: Notes ===
with tab_notes:
    current = NOTES_MD.read_text(encoding="utf-8") if NOTES_MD.exists() else ""
    edited = st.text_area("notes.md", current, height=500, label_visibility="collapsed")
    col_s, col_r = st.columns([1, 5])
    if col_s.button("Сохранить", type="primary"):
        NOTES_MD.write_text(edited, encoding="utf-8")
        st.success(f"Записал в {NOTES_MD.name}")
    if col_r.button("Перечитать с диска"):
        st.rerun()

# === Tab: Help ===
with tab_help:
    st.markdown(
        """
### Как использовать панель

1. **Создай комнату** — на вкладке `Комнаты` через "Добавить новую комнату" или вручную
   положи видео в `captures/<room>/video.mp4`
2. **Запиши видео** на iPhone по [`docs/capture-guide.md`](docs/capture-guide.md)
3. **Run baseline** — запустит `pipeline/run_baseline.sh`, поток логов появится прямо тут
4. **Открой результат** через "Reveal results folder" → перетащи `.ply` в supersplat-viewer
5. **Запиши метрики** в `results/metrics.csv`, заметки на вкладке `Заметки`

### Что важно

- Streamlit блокирует UI пока тренировка работает — это нормально, не закрывай вкладку
- При OOM на 3070 Ti убавь `--max-num-iterations` или `--num-frames-target` в `run_baseline.sh`
- LingBot путь — пока заглушка, нужно сначала изучить актуальный CLI их репо
- Если что-то не отрисовалось — кнопка `Refresh` в сайдбаре

### Где что лежит

```
beta_v0.1/
├── ui/app.py              <- этот файл
├── pipeline/              <- shell-скрипты + Python helpers
├── captures/<room>/video.mp4
├── results/<room>/        <- processed/, train/, *.ply
├── results/metrics.csv
└── results/notes.md
```
"""
    )
