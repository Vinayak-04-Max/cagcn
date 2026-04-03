import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import altair as alt


# ================================================================
# Page config
# ================================================================
st.set_page_config(
    page_title="GhostNet CAGCN Dashboard",
    layout="wide"
)

# ================================================================
# Custom Layer
# ================================================================
class ClassAdaptiveCalibration(tf.keras.layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name="alpha", shape=(self.num_classes,),
            initializer="ones", trainable=True
        )
        self.beta = self.add_weight(
            name="beta", shape=(self.num_classes,),
            initializer="zeros", trainable=True
        )
        super().build(input_shape)

    def call(self, logits):
        return tf.nn.sigmoid(self.alpha * logits + self.beta)

# ================================================================
# Cached assets
# ================================================================
@st.cache_resource
def load_assets():
    model = load_model(
        "experiment/best_model.h5",
        compile=False,
        custom_objects={"ClassAdaptiveCalibration": ClassAdaptiveCalibration}
    )
    alpha = np.load("experiment/alpha.npy")
    beta = np.load("experiment/beta.npy")
    thresholds = np.load("experiment/thresholds.npy")
    return model, alpha, beta, thresholds

model, alpha, beta, thresholds = load_assets()

# ================================================================
# Helper functions
# ================================================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def infer(X):
    backbone = tf.keras.Model(
        inputs=model.input,
        outputs=model.layers[-1].input
    )
    logits = backbone.predict(X, verbose=0)
    probs = sigmoid(alpha * logits + beta)
    preds = (probs >= thresholds).astype(int)
    return preds, probs

def colored_wafer(wafer):
    fig, ax = plt.subplots()
    ax.imshow(wafer, cmap="plasma")
    ax.axis("off")
    return fig

@st.cache_data
def load_random_wafers(k=10):
    data = np.load("Wafer_Map_Datasets_1.npz")
    X = data["arr_0"] if "arr_0" in data else data[list(data.keys())[0]]
    idx = np.random.choice(len(X), k, replace=False)
    return X[idx]

# ================================================================
# Labels
# ================================================================
label_names = [
    "Center", "Donut", "Edge_Loc", "Edge_Ring",
    "Loc", "Near_Full", "Scratch", "Random"
]

# ================================================================
# Sidebar
# ================================================================
with st.sidebar:
    st.title("🧪 Wafer Selector")

    st.markdown(
        """
        **Click a wafer image to select it.**  
        The selected wafer will be sent directly to the model.
        """
    )

    if st.button("🔄 Resample Wafers"):
        st.cache_data.clear()
        st.session_state.pop("selected_idx", None)

    wafers = load_random_wafers()

    if "selected_idx" not in st.session_state:
        st.session_state.selected_idx = None

    for i in range(len(wafers)):
        fig = colored_wafer(wafers[i])
        st.pyplot(fig, use_container_width=True)
        if st.button("Select", key=f"select_{i}"):
            st.session_state.selected_idx = i

# ================================================================
# Main Page
# ================================================================
st.title("🧠 GhostNet CAGCN – Inference Results")

if st.session_state.selected_idx is None:
    st.info("Select a wafer from the sidebar to view results")
    st.stop()

# ================================================================
# Inference
# ================================================================
with st.spinner("Running inference..."):
    X_sel = wafers[st.session_state.selected_idx].astype("float32")
    X_sel = X_sel[..., np.newaxis]
    X_sel = np.expand_dims(X_sel, axis=0)

    preds, probs = infer(X_sel)

pred_bin = preds[0]
probs_arr = np.round(probs[0], 3)

pred_labels = [
    label_names[i] for i in range(len(label_names)) if pred_bin[i] == 1
]

# ================================================================
# Text Output (Neatly Printed)
# ================================================================
st.markdown("### 📄 Prediction Summary")

st.markdown("**Predicted Labels:**")

label_text = ", ".join(pred_labels) if pred_labels else "None"

st.markdown(
    f"""
    <div style="
        background-color:#0D0887;
        color:#FDE725;
        padding:0.75rem 1rem;
        border-radius:0.5rem;
        font-weight:600;
        border-left:6px solid #FDE725;
    ">
        {label_text}
    </div>
    """,
    unsafe_allow_html=True
)




# ================================================================
# Plots Layout
# ================================================================
col1, col2 = st.columns([1, 2])


# -------------------------------
# Left: Count-of-ones bar chart
# -------------------------------
with col1:
    st.markdown("#### ✅ Prediction (Binary)")
    pred_df = pd.DataFrame({
        "label": label_names,
        "value": pred_bin
    })

    bar_chart = (
        alt.Chart(pred_df)
        .mark_bar(color="gold")
        .encode(
            x=alt.X("label:N", sort=None),
            y="value:Q"
        )
    )

    st.altair_chart(bar_chart, use_container_width=True)

# -------------------------------
# Right: Probability line graph
# -------------------------------
with col2:
    st.markdown("#### 📈 Probabilities")
    prob_df = pd.DataFrame({
        "label": label_names,
        "value": probs_arr
    })

    line_chart = (
        alt.Chart(prob_df)
        .mark_line(color="gold", point=True)
        .encode(
            x=alt.X("label:N", sort=None),
            y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1]))
        )
    )

    st.altair_chart(line_chart, use_container_width=True)
