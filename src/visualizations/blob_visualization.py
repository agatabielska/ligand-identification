import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import Callable, Dict
import sys
from pathlib import Path

# Ensure src is in sys.path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.sampling_strategies import RandomSelectionTransform, UniformSelectionTransform

# ============================================================================
# SAMPLING FUNCTIONS - Add your custom sampling functions here
# ============================================================================

def no_sampling(data: np.ndarray) -> np.ndarray:
    """Return all non-zero probability points"""
    return data

def random_sampling(data: np.ndarray, ratio: float) -> np.ndarray:
    transform = RandomSelectionTransform(max_blob_size=int(np.count_nonzero(data) * ratio))
    return transform.preprocess(data)

def uniform_sampling(data: np.ndarray, ratio: float, method: str) -> np.ndarray:
    transform = UniformSelectionTransform(max_blob_size=int(np.count_nonzero(data) * ratio), method=method)
    return transform.preprocess(data)

# ============================================================================
# SAMPLING FUNCTIONS REGISTRY
# ============================================================================

SAMPLING_FUNCTIONS: Dict[str, Callable] = {
    "No Sampling": no_sampling,
    "Random 10%": lambda data: random_sampling(data, ratio=0.1),
    "Random 30%": lambda data: random_sampling(data, ratio=0.3),
    "Random 50%": lambda data: random_sampling(data, ratio=0.5),
    "Uniform 10% max": lambda data: uniform_sampling(data, ratio=0.1, method='max'),
    "Uniform 30% max": lambda data: uniform_sampling(data, ratio=0.3, method='max'),
    "Uniform 50% max": lambda data: uniform_sampling(data, ratio=0.5, method='max'),
    "Uniform 10% average": lambda data: uniform_sampling(data, ratio=0.1, method='average'),
    "Uniform 30% average": lambda data: uniform_sampling(data, ratio=0.3, method='average'),
    "Uniform 50% average": lambda data: uniform_sampling(data, ratio=0.5, method='average'),
    "Uniform 10% basic": lambda data: uniform_sampling(data, ratio=0.1, method='basic'),
    "Uniform 30% basic": lambda data: uniform_sampling(data, ratio=0.3, method='basic'),
    "Uniform 50% basic": lambda data: uniform_sampling(data, ratio=0.5, method='basic'),
}

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_3d_scatter(data: np.ndarray, title: str, data_shape: tuple) -> go.Figure:
    """Create a 3D scatter plot from probability data with fixed axes"""
    # Get non-zero probability coordinates
    x, y, z = np.where(data > 0)
    probabilities = data[x, y, z]
    
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
        
        if 'arr_0' not in data:
            st.error("NPZ file must contain 'arr_0' entry")
        else:
            blob = data['arr_0']
            
            # Validate data
            if blob.ndim != 3:
                st.error(f"Expected 3D array, got {blob.ndim}D array")
            elif blob.dtype != np.float64:
                st.warning(f"Expected float64, got {blob.dtype}. Converting...")
                blob = blob.astype(np.float64)
            
            st.success(f"Loaded data with shape: {blob.shape}")
            st.info(f"Non-zero values: {np.count_nonzero(blob)} / {blob.size} ({np.count_nonzero(blob)/blob.size*100:.2f}%)")
            st.info(f"Value range: [{blob.min():.4f}, {blob.max():.4f}]")
            
            # Create three columns for plots
            cols = st.columns(3)
            
            # Store sampling choices in session state
            if 'sampling_choices' not in st.session_state:
                st.session_state.sampling_choices = ["No Sampling"] * 3
            
            # Create selectboxes and plots for each column
            for i, col in enumerate(cols):
                with col:
                    st.session_state.sampling_choices[i] = st.selectbox(
                        f"Sampling Function (Plot {i+1})",
                        options=list(SAMPLING_FUNCTIONS.keys()),
                        key=f"sampling_{i}",
                        index=list(SAMPLING_FUNCTIONS.keys()).index(st.session_state.sampling_choices[i])
                    )
                    
                    # Apply sampling function
                    sampling_func = SAMPLING_FUNCTIONS[st.session_state.sampling_choices[i]]
                    sampled_data = sampling_func(blob.copy())
                    
                    # Show stats
                    n_points = np.count_nonzero(sampled_data)
                    st.caption(f"Points displayed: {n_points}")
                    
                    # Create and display plot with fixed scale
                    fig = create_3d_scatter(sampled_data, f"Plot {i+1}: {st.session_state.sampling_choices[i]}", blob.shape)
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
        4. Choose a sampling function for each plot from the dropdown menu
        5. Only sampled (non-zero) data points will be visualized
        
        ### Adding custom sampling functions:
        Edit the `SAMPLING_FUNCTIONS` dictionary in the source code to add your own sampling functions.
        Each function should take a 3D numpy array and return a filtered 3D numpy array.
        """)