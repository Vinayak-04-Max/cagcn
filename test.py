import numpy as np
import pandas as pd

NPZ_PATH = "Wafer_Final (1).npz"
OUTPUT_CSV = "inference.csv"

label_names = [
    "Center", "Donut", "Edge_Loc", "Edge_Ring",
    "Loc", "Near_Full", "Scratch", "Random"
]

data = np.load(NPZ_PATH)
X = data["arr_0"].astype("float32")
y = data["arr_1"].astype("int")

if X.ndim == 4:
    X = X.squeeze(-1)

X_flat = X.reshape(X.shape[0], -1)

df_X = pd.DataFrame(X_flat)
df_y = pd.DataFrame(y, columns=label_names)

df = pd.concat([df_X, df_y], axis=1)
df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Saved dataset CSV with labels: {OUTPUT_CSV}")
