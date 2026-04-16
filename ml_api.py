from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = FastAPI()

# ✅ Allow frontend (React) to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# Custom Layer
# ========================
class ClassAdaptiveCalibration(tf.keras.layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name="alpha",
            shape=(self.num_classes,),
            initializer="ones",
            trainable=True
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(self.num_classes,),
            initializer="zeros",
            trainable=True
        )

    def call(self, logits):
        return tf.nn.sigmoid(self.alpha * logits + self.beta)


# ========================
# Load once (IMPORTANT FIX)
# ========================
model = load_model(
    "experiment/best_model.h5",
    compile=False,
    custom_objects={"ClassAdaptiveCalibration": ClassAdaptiveCalibration}
)

alpha = np.load("experiment/alpha.npy")
beta = np.load("experiment/beta.npy")
thresholds = np.load("experiment/thresholds.npy")
dataset = np.load("Wafer_Map_Datasets_1.npz")
wafer_maps = dataset["arr_0"] if "arr_0" in dataset else dataset[list(dataset.keys())[0]]

label_names = [
    "Center", "Donut", "Edge_Loc", "Edge_Ring",
    "Loc", "Near_Full", "Scratch", "Random"
]

# Build backbone ONCE (IMPORTANT FIX)
backbone = tf.keras.Model(
    inputs=model.input,
    outputs=model.layers[-1].input
)

# ========================
# Request schema (FIX)
# ========================
class WaferRequest(BaseModel):
    wafer: list


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sample_wafers(count: int):
    safe_count = max(1, min(count, len(wafer_maps)))
    idx = np.random.choice(len(wafer_maps), safe_count, replace=False)
    sampled = wafer_maps[idx]
    return sampled.astype(np.float32).tolist()


# ========================
# API
# ========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "samples": len(wafer_maps),
        "labels": label_names
    }


@app.get("/wafers/random")
def random_wafers(count: int = Query(default=10, ge=1, le=24)):
    return {
        "count": count,
        "wafers": sample_wafers(count)
    }


@app.post("/predict")
def predict(req: WaferRequest):

    wafer = np.array(req.wafer, dtype=np.float32)

    # safety check
    if wafer.ndim == 2:
        wafer = wafer[..., np.newaxis]

    wafer = np.expand_dims(wafer, axis=0)

    logits = backbone.predict(wafer, verbose=0)
    probs = sigmoid(alpha * logits + beta)
    preds = (probs >= thresholds).astype(int)

    return {
        "labels": [
            label_names[i] for i in range(len(label_names))
            if preds[0][i] == 1
        ],
        "probabilities": probs[0].tolist(),
        "binary": preds[0].tolist()
    }
