import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path
import inspect
from typing import Any, List

# Ensure src is in sys.path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.sampling_strategies import (
    RandomSelectionTransform, 
    UniformSelectionTransform, 
    ProbabilisticSelectionTransform, 
    SpacialNormalization,
    PowerTransform
)

# Registry of available Transform classes for the custom stack editor
TRANSFORM_CLASSES = {
    'RandomSelectionTransform': RandomSelectionTransform,
    'UniformSelectionTransform': UniformSelectionTransform,
    'ProbabilisticSelectionTransform': ProbabilisticSelectionTransform,
    'SpacialNormalization': SpacialNormalization,
    "PowerTransform": PowerTransform
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


def instantiate_and_apply_stack(data: np.ndarray, stack: List[dict]) -> np.ndarray:
    """Given a stack (list of dict with 'class' and 'kwargs'), instantiate each Transform and apply sequentially.

    Each item in stack: {'class': 'ClassName', 'kwargs': {'param': value, ...}}
    """
    out = data

    def kwargs_complete(kwargs: dict) -> bool:
        # Consider kwargs incomplete if any value is an empty string or None.
        for v in kwargs.values():
            if v is None:
                return False
            if isinstance(v, str) and v.strip() == '':
                return False
        return True

    for item in stack:
        cls_name = item.get('class')
        kwargs = item.get('kwargs', {}) or {}
        cls = TRANSFORM_CLASSES.get(cls_name)
        if cls is None:
            # unknown transform: skip
            continue

        # Skip transforms with incomplete kwargs (e.g., placeholders like '')
        if not kwargs_complete(kwargs):
            continue

        try:
            inst = cls(**kwargs)
            if hasattr(inst, 'preprocess'):
                out = inst.preprocess(out)
        except Exception as e:
            # If instantiation fails, skip the transform but continue pipeline
            print(f"Failed to instantiate {cls_name} with {kwargs}: {e}")
            continue
    return out

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_3d_scatter(data: np.ndarray, title: str, data_shape: tuple) -> go.Figure:
    """Create a 3D scatter plot from probability data with fixed axes"""
    # Get non-zero probability coordinates
    x, y, z = np.where(data > 0)
    probabilities = data[x, y, z]

    probabilities = (probabilities - np.min(probabilities)) / (np.max(probabilities) - np.min(probabilities) + 1e-9)
    
    # Create scatter plot (even if empty)
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=probabilities if len(probabilities) > 0 else [],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Probability"),
            cmin=0,
            cmax=1
        ),
        text=[f'Prob: {p:.3f}' for p in probabilities],
        hovertemplate='<b>Position</b><br>X: %{x}<br>Y: %{y}<br>Z: %{z}<br>%{text}<extra></extra>'
    )])
    
    if len(x) == 0:
        fig.add_annotation(
            text="No data to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
    
    # Fix axes to original data dimensions
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X', range=[0, data_shape[0]-1]),
            yaxis=dict(title='Y', range=[0, data_shape[1]-1]),
            zaxis=dict(title='Z', range=[0, data_shape[2]-1]),
            aspectmode='data'
        ),
        height=500,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(layout="wide", page_title="3D Probability Visualizer")

st.title("3D Probability Data Visualizer")
st.markdown("Upload an NPZ file containing a 'blob' entry with 3D probability data")

# File upload
uploaded_file = st.file_uploader("Choose an NPZ file", type=['npz'])

if uploaded_file is not None:
    try:
        # Load NPZ file
        data = np.load(uploaded_file)
        
        # Check what keys are present in the NPZ file
        keys = list(data.keys())
        
        if len(keys) != 1:
            st.error("NPZ file must contain exactly one entry, but found: " + ", ".join(keys))
        else:
            blob = data[keys[0]]
            
            # Validate data
            if blob.ndim != 3:
                st.error(f"Expected 3D array, got {blob.ndim}D array")
            elif blob.dtype != np.float64:
                st.warning(f"Expected float64, got {blob.dtype}. Converting...")
                blob = blob.astype(np.float64)
            
            st.success(f"Loaded data with shape: {blob.shape}")
            st.info(f"Non-zero values: {np.count_nonzero(blob)} / {blob.size} ({np.count_nonzero(blob)/blob.size*100:.2f}%)")
            st.info(f"Value range: [{blob.min():.4f}, {blob.max():.4f}]")
            
            # Store transform stacks in session state (three stacks)
            if 'transform_stacks' not in st.session_state:
                st.session_state.transform_stacks = [[] for _ in range(3)]

            # Helper to render a stack editor for a plot
            def render_stack_editor(plot_idx: int):
                stack = st.session_state.transform_stacks[plot_idx]

                st.markdown("**Available transforms:**")
                cols_small = st.columns([3, 1])
                with cols_small[0]:
                    to_add = st.selectbox(f"Transform to add (Stack {plot_idx+1})", options=list(TRANSFORM_CLASSES.keys()), key=f"add_select_{plot_idx}")
                with cols_small[1]:
                    if st.button("Add", key=f"add_btn_{plot_idx}"):
                        # append with kwargs from constructor defaults when available
                        cls = TRANSFORM_CLASSES[to_add]
                        sig = inspect.signature(cls.__init__)
                        kwargs = {}
                        for name, param in sig.parameters.items():
                            if name == 'self':
                                continue
                            if param.default is not inspect._empty:
                                kwargs[name] = param.default
                            else:
                                kwargs[name] = ''
                        stack.append({'class': to_add, 'kwargs': kwargs})
                        st.session_state.transform_stacks[plot_idx] = stack

                # Render existing stack items
                for idx, item in enumerate(list(stack)):
                    cls_name = item.get('class')
                    kwargs = item.get('kwargs', {}) or {}
                    with st.expander(f"{idx+1}. {cls_name}", expanded=False):
                        # Show remove button
                        if st.button("Remove", key=f"remove_{plot_idx}_{idx}"):
                            stack.pop(idx)
                            st.session_state.transform_stacks[plot_idx] = stack
                            # Some Streamlit installations don't expose experimental_rerun.
                            # Guard the call so we don't raise AttributeError.
                            if hasattr(st, "experimental_rerun") and callable(getattr(st, "experimental_rerun")):
                                st.experimental_rerun()
                            # If experimental_rerun isn't available, updating session_state is
                            # usually sufficient to trigger a rerun on next interaction.
                            # Avoid continuing to process this expander after mutating `stack`
                            # which would cause IndexError when accessing stack[idx].
                            return
                        
                        # show inputs for kwargs
                        for param_name, param_val in kwargs.items():
                            widget_key = f"plot{plot_idx}_item{idx}_{param_name}"
                            val = st.text_input(f"{param_name}", value=str(param_val), key=widget_key)
                            cls = TRANSFORM_CLASSES.get(cls_name)
                            expected = None
                            if cls is not None:
                                sig = inspect.signature(cls.__init__)
                                p = sig.parameters.get(param_name)
                                if p is not None and p.annotation in (int, float, str):
                                    expected = p.annotation
                            parsed = parse_input_value(val, expected)
                            kwargs[param_name] = parsed

                        stack[idx]['kwargs'] = kwargs

                st.session_state.transform_stacks[plot_idx] = stack

            # Render stack editors above the plots so plots are always aligned
            st.markdown("## Transform stacks")
            editor_cols = st.columns(3)
            for i, ecol in enumerate(editor_cols):
                with ecol:
                    st.subheader(f"Stack {i+1}")
                    render_stack_editor(i)

            # Now render the three plots in aligned columns (editors are above)
            plot_cols = st.columns(3)
            for i, pcol in enumerate(plot_cols):
                with pcol:
                    stack = st.session_state.transform_stacks[i]
                    sampled_data = instantiate_and_apply_stack(blob.copy(), stack)
                    title = f"Plot {i+1}: {len(stack)} transforms"

                    # Show stats
                    n_points = np.count_nonzero(sampled_data)
                    st.caption(f"Points displayed: {n_points}")

                    # Create and display plot with fixed scale
                    fig = create_3d_scatter(sampled_data, title, blob.shape)
                    st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.exception(e)
else:
    st.info("👆 Upload an NPZ file to begin visualization")
    
    # Show instructions
    with st.expander("ℹ️ Instructions"):
        st.markdown("""
    ### How to use:
    1. Upload an NPZ file containing a 'blob' entry with 3D probability data
    2. The data should be a 3D numpy array of float64 probabilities (values between 0 and 1)
    3. Three interactive 3D plots will be displayed side by side
    4. Edit the transform stacks in the "Transform stacks" section above the plots.
    5. For each stack you can add transforms, remove them, and edit their kwargs. The plots below will apply the stacks in order.
        
    ### Notes on kwargs:
    - All kwargs are edited as text inputs and will be parsed into int/float when possible.
    - If a transform constructor provides default values they will be used when adding a transform.
        """)