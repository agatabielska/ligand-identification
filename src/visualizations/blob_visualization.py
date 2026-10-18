import base64
import json

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from plotly.colors import sequential
import sys
from pathlib import Path
import inspect
import uuid
from typing import Any, List

# Ensure src is in sys.path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.sampling_strategies import (
    RandomSelectionTransform,
    UniformSelectionTransform,
    ProbabilisticSelectionTransform,
    SpatialNormalization,
    SpatialNormalization2,
    SpatialNormalization3,
    SpatialStandardization,
    SpatialStandardization2,
    PowerTransform,
    TemperatureScaling,
)
from visualizations.blob_kind import recognise
from visualizations.render_blob import auto_sigma, splat_to_grid

PLOT_HEIGHT = 500
PLOT_TEMPLATE = (Path(__file__).parent / "progressive_plot.html").read_text(encoding="utf-8")

# Registry of available Transform classes for the custom stack editor
TRANSFORM_CLASSES = {
    "RandomSelectionTransform": RandomSelectionTransform,
    "UniformSelectionTransform": UniformSelectionTransform,
    "ProbabilisticSelectionTransform": ProbabilisticSelectionTransform,
    "SpatialNormalization": SpatialNormalization,
    "SpatialNormalization2": SpatialNormalization2,
    "SpatialNormalization3": SpatialNormalization3,
    "SpatialStandardization": SpatialStandardization,
    "SpatialStandardization2": SpatialStandardization2,
    "PowerTransform": PowerTransform,
    "TemperatureScaling": TemperatureScaling,
}


def parse_input_value(value: str, expected_type: Any = None):
    """Try to coerce a string input into int/float/str based on expectation or content."""
    if expected_type in (int, float, str):
        try:
            if expected_type is int:
                return int(value)
            if expected_type is float:
                return float(value)
            return value
        except Exception:
            return value

    # Heuristic parsing
    try:
        return int(value)
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        pass
    return value


def instantiate_and_apply_stack(data: np.ndarray, stack: List[dict]) -> tuple[np.ndarray, list[str], bool]:
    """Given a stack (list of dict with 'class' and 'kwargs'), instantiate each Transform and apply sequentially.

    Each item in stack: {'class': 'ClassName', 'kwargs': {'param': value, ...}}
    Returns (output, notes on skipped transforms, whether a ProbabilisticSelectionTransform subsampled the blob).
    """
    out, notes, sampled = data, [], False

    for item in stack:
        cls_name = item.get("class")
        kwargs = item.get("kwargs", {}) or {}
        cls = TRANSFORM_CLASSES.get(cls_name)
        if cls is None:
            # unknown transform: skip
            continue

        # Skip transforms with incomplete kwargs (e.g., placeholders like '')
        missing = [k for k, v in kwargs.items() if v is None or (isinstance(v, str) and not v.strip())]
        if missing:
            notes.append(f"{cls_name} skipped: set {', '.join(missing)}")
            continue

        try:
            inst = cls(**kwargs)
            if hasattr(inst, "preprocess"):
                new = inst.preprocess(out)
                if cls is ProbabilisticSelectionTransform and np.count_nonzero(new) < np.count_nonzero(out):
                    sampled = True
                out = new
        except Exception as e:
            # If a transform fails, skip it but continue the pipeline
            notes.append(f"{cls_name} failed and was skipped: {e}")
            continue
    return out, notes, sampled


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


@st.cache_data(max_entries=64, show_spinner=False)
def apply_stack_cached(blob: np.ndarray, stack: List[dict]) -> tuple[np.ndarray, list[str], bool]:
    """Cached so reruns triggered by other widgets keep the same (random) samples instead of redrawing them."""
    return instantiate_and_apply_stack(blob.copy(), stack)


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


@st.cache_data(max_entries=64, show_spinner=False)
def plot_html(data: np.ndarray, title: str, mode: str = "points", filename: str = "plot", weight: str = "values") -> str:
    """Self-contained three.js page (progressive_plot.html): axes at once, then points stream into GPU buffers.

    Only compact typed arrays go to the browser (uint16 coords, float32 values, uint8 volume grid); spheres and
    isosurfaces are built client-side. weight="count" for points drawn ∝ density, so volume mode doesn't count it twice.
    """
    idx = np.argwhere(data > 0)
    raw = data[tuple(idx.T)]
    norm = (raw - raw.min()) / (np.ptp(raw) + 1e-9) if len(raw) else raw
    order = np.random.default_rng(0).permutation(len(idx))  # random arrival order: whole shape appears, then densifies
    payload: dict[str, Any] = dict(
        mode=mode, title=title, filename=filename, height=PLOT_HEIGHT, colorscale=sequential.Viridis,
        shape=list(data.shape), n=len(idx),
        coords=_b64(idx[order].astype(np.uint16)), values=_b64(norm[order].astype(np.float32)))
    if mode == "spheres":
        payload.update(radius=0.6)  # voxels, as in render_blob.spheres_trace
    elif mode == "volume" and len(idx):
        coords = idx.astype(float)
        sigma = auto_sigma(coords)
        grid, origin = splat_to_grid(coords, raw, sigma, weight)
        grid = np.rint(255 * grid / (grid.max() or 1.0)).astype(np.uint8)  # 8 bits is plenty for isolevels 0.15-0.6
        payload.update(grid=_b64(grid), grid_shape=list(grid.shape), origin=origin.tolist())
    return PLOT_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))


def show_intensity_histogram(values: np.ndarray, title: str, nbins: int = 50) -> None:
    """Histogram of intensities normalised to [0, 1]. Streamlit's own bar chart (Vega) instead of st.plotly_chart,
    which made the page parse a second full plotly.js and delayed the 3D plots by ~1.4 s."""
    counts, edges = np.histogram(values, bins=nbins, range=(0.0, 1.0))
    st.caption(title)
    st.bar_chart({"Intensity": np.round((edges[:-1] + edges[1:]) / 2, 3), "Count": counts},
                 x="Intensity", y="Count", height=250)


# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(layout="wide", page_title="3D Probability Visualizer")

st.title("3D Probability Data Visualizer")
st.markdown("Upload a raw blob (`data/cryoem_blobs`) or a preprocessed file (`data/*_transform_*`); its kind is recognised automatically")
RENDER_MODES = ["points", "spheres", "volume"]
RENDER_HELP = "points: flat markers (fastest); spheres: lit sphere per voxel; volume: Gaussian-smoothed isosurfaces"


def sync_all_render_modes():
    for i in range(3):
        st.session_state[f"render_mode_{i}"] = st.session_state.render_mode_all


def update_global_render_mode():
    # Global switch shows the common mode, or nothing when stacks differ
    # .get: callbacks run before the script body, so plots the previous (possibly interrupted) run never reached have no key
    modes = {st.session_state.get(f"render_mode_{i}", st.session_state.render_mode_all) for i in range(3)}
    st.session_state.render_mode_all = modes.pop() if len(modes) == 1 else None


st.session_state.setdefault("render_mode_all", "points")
st.sidebar.radio("3D rendering (all stacks)", RENDER_MODES, key="render_mode_all",
                 on_change=sync_all_render_modes, help=RENDER_HELP)

# File upload
uploaded_file = st.file_uploader("Choose an NPZ file", type=["npz"])

if uploaded_file is not None:
    try:
        try:
            blob, info = recognise(uploaded_file)  # raw dense or preprocessed sparse, as a dense grid
        except ValueError as e:
            st.error(str(e))
        else:
            st.success(f"Loaded data with shape: {blob.shape}")
            st.info(info["description"])
            if info["ligand"]:
                st.info(f"Ligand code from the file name: {info['ligand']} (PDB {info['pdb_id']}). Preprocessed copies "
                        "are stored under this code, or under `rare` if it isn't one of the trained classes.")
            st.info(
                f"Non-zero values: {np.count_nonzero(blob)} / {blob.size} ({np.count_nonzero(blob) / blob.size * 100:.2f}%)"
            )
            st.info(f"Value range: [{blob.min():.4f}, {blob.max():.4f}]")

            # Store transform stacks in session state (three stacks)
            if "transform_stacks" not in st.session_state:
                st.session_state.transform_stacks = [[] for _ in range(3)]

            # Helper to render a stack editor for a plot
            def render_stack_editor(plot_idx: int):
                stack = st.session_state.transform_stacks[plot_idx]

                st.markdown("**Available transforms:**")
                cols_small = st.columns([3, 1])
                with cols_small[0]:
                    to_add = st.selectbox(
                        f"Transform to add (Stack {plot_idx + 1})",
                        options=list(TRANSFORM_CLASSES.keys()),
                        key=f"add_select_{plot_idx}",
                    )
                with cols_small[1]:
                    if st.button("Add", key=f"add_btn_{plot_idx}"):
                        # append with kwargs from constructor defaults when available
                        cls = TRANSFORM_CLASSES[to_add]
                        sig = inspect.signature(cls.__init__)
                        kwargs = {}
                        for name, param in sig.parameters.items():
                            if name == "self":
                                continue
                            if param.default is not inspect._empty:
                                kwargs[name] = param.default
                            else:
                                kwargs[name] = ""
                        stack.append({"class": to_add, "kwargs": kwargs})
                        st.session_state.transform_stacks[plot_idx] = stack

                # Render existing stack items
                for idx, item in enumerate(list(stack)):
                    item.setdefault("id", uuid.uuid4().hex)  # widget keys follow the item, not its position
                    cls_name = item.get("class")
                    kwargs = item.get("kwargs", {}) or {}
                    with st.expander(f"{idx + 1}. {cls_name}", expanded=False):
                        # Show remove button
                        if st.button("Remove", key=f"remove_{item['id']}"):
                            stack.pop(idx)
                            st.rerun()

                        # show inputs for kwargs
                        for param_name, param_val in kwargs.items():
                            widget_key = f"item_{item['id']}_{param_name}"
                            val = st.text_input(
                                f"{param_name}", value=str(param_val), key=widget_key
                            )
                            cls = TRANSFORM_CLASSES.get(cls_name)
                            expected = None
                            if cls is not None:
                                sig = inspect.signature(cls.__init__)
                                p = sig.parameters.get(param_name)
                                if p is not None and p.annotation in (int, float, str):
                                    expected = p.annotation
                            parsed = parse_input_value(val, expected)
                            kwargs[param_name] = parsed

                        stack[idx]["kwargs"] = kwargs

                st.session_state.transform_stacks[plot_idx] = stack

            # Render stack editors above the plots so plots are always aligned
            st.markdown("## Transform stacks")
            editor_cols = st.columns(3)
            for i, ecol in enumerate(editor_cols):
                with ecol:
                    st.subheader(f"Stack {i + 1}")
                    render_stack_editor(i)

            plot_cols = st.columns(3)
            for i, pcol in enumerate(plot_cols):
                with pcol:
                    stack = st.session_state.transform_stacks[i]
                    st.session_state.setdefault(f"render_mode_{i}", st.session_state.render_mode_all or "points")
                    render_mode = st.radio("3D rendering", RENDER_MODES, key=f"render_mode_{i}", horizontal=True,
                                           on_change=update_global_render_mode, help=RENDER_HELP)
                    # ids stripped so identical stacks share a cache entry (same samples), as before
                    sampled_data, notes, stack_sampled = apply_stack_cached(
                        blob, [{k: v for k, v in it.items() if k != "id"} for it in stack])
                    title = f"Plot {i + 1}: {len(stack) - len(notes)} of {len(stack)} transforms applied"

                    # Show stats
                    n_points = np.count_nonzero(sampled_data)
                    st.caption(f"Points displayed: {n_points}")

                    # One final render per plot; identical HTML on a rerun leaves the iframe alone (no reload).
                    weight = "count" if info["density_sampled"] or stack_sampled else "values"
                    components.html(plot_html(sampled_data, title, render_mode, f"plot_{i + 1}_{render_mode}", weight),
                                    height=PLOT_HEIGHT + 20)

                    # Below each scatter, show a histogram of intensities for displayed points
                    intensities = sampled_data[sampled_data > 0]
                    if intensities.size > 0:
                        # Normalize to [0,1] for consistency with color scale
                        vmin = float(intensities.min())
                        vmax = float(intensities.max())
                        rng = vmax - vmin
                        if rng > 0:
                            intens_norm = (intensities - vmin) / (rng + 1e-9)
                        else:
                            intens_norm = np.zeros_like(intensities)
                    else:
                        intens_norm = intensities  # empty

                    show_intensity_histogram(intens_norm, f"Intensity distribution {i + 1}")
                    for note in notes:  # below the plots, so the three plots stay aligned
                        st.warning(note)


    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.exception(e)
else:
    st.info("👆 Upload an NPZ file to begin visualization")

    # Show instructions
    with st.expander("ℹ️ Instructions"):
        st.markdown("""
    ### How to use:
    1. Upload a raw blob (one dense 3D array, e.g. `arr_0`) or a preprocessed file (`indices`, `values`, `shape`)
    2. The app reports which kind it is (raw thresholded/unthresholded, uniform, probabilistic) and the ligand code from the file name
    3. Three interactive 3D plots will be displayed side by side
    4. Edit the transform stacks in the "Transform stacks" section above the plots.
    5. For each stack you can add transforms, remove them, and edit their kwargs. The plots below will apply the stacks in order.
        
    ### Notes on kwargs:
    - All kwargs are edited as text inputs and will be parsed into int/float when possible.
    - If a transform constructor provides default values they will be used when adding a transform.
        """)
