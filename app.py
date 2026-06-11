import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
load_model = tf.keras.models.load_model
from sklearn.metrics import (
silhouette_score,
accuracy_score,
precision_score,
recall_score,
f1_score,
confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================

# KONFIGURASI HALAMAN

# =====================================================

st.set_page_config(
page_title="DEC Hotel Segmentation",
layout="wide"
)

st.title("🏨 Segmentasi Pelanggan Hotel Menggunakan DEC dan Prediksi Booking Cancellation")

# =====================================================

# LOAD MODEL DAN FILE PENDUKUNG

# =====================================================

@st.cache_resource
def load_models():

    encoder_model = load_model(
        "encoder_model.keras",
        compile=False
    )

    rf_model = joblib.load(
        "rf_model.pkl"
    )

    scaler = joblib.load(
        "scaler.pkl"
    )

    label_encoders = joblib.load(
        "label_encoders.pkl"
    )

    selected_features = joblib.load(
        "selected_features.pkl"
    )

    rf_features = joblib.load(
        "rf_features.pkl"
    )

    cluster_centers = joblib.load(
        "cluster_centers.pkl"
    )

    pca = joblib.load(
        "pca.pkl"
    )

    return (
        encoder_model,
        rf_model,
        scaler,
        label_encoders,
        selected_features,
        rf_features,
        cluster_centers,
        pca
    )


(
encoder_model,
rf_model,
scaler,
label_encoders,
selected_features,
rf_features,
cluster_centers,
pca
) = load_models()

# =====================================================

# FUNGSI SAFE LABEL ENCODING

# =====================================================

def safe_label_encode(series, encoder):

    mapping = {
        cls: idx
        for idx, cls in enumerate(encoder.classes_)
    }

    return series.map(
        lambda x: mapping.get(x, -1)
    )

# =====================================================

# UPLOAD DATASET

# =====================================================

uploaded_file = st.file_uploader(
"Upload Dataset Hotel Reservation (.csv)",
type=["csv"]
)

# =====================================================

# PROSES DATASET

# =====================================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

st.subheader("Preview Dataset")

st.dataframe(df.head())

# =================================================
# MENYIMPAN LABEL ASLI
# =================================================

y_true = None

if "booking_status" in df.columns:
    y_true = df["booking_status"].copy()

# =================================================
# ENCODING FITUR KATEGORIKAL
# =================================================

df_encoded = df.copy()

for col, encoder in label_encoders.items():

    if col in df_encoded.columns:

        df_encoded[col] = safe_label_encode(
            df_encoded[col],
            encoder
        )

# =================================================
# MEMBUAT FITUR UNTUK DEC
# =================================================

X_dec = df_encoded[
    selected_features
]

# =================================================
# SCALING
# =================================================

X_scaled = scaler.transform(
    X_dec
)

# =================================================
# LATENT FEATURE DARI ENCODER
# =================================================

latent_features = encoder_model.predict(
    X_scaled,
    verbose=0
)

# =================================================
# MENENTUKAN CLUSTER BERDASARKAN CENTROID DEC
# =================================================

distances = cdist(
    latent_features,
    cluster_centers
)

clusters = np.argmin(
    distances,
    axis=1
)

# =================================================
# MENAMBAHKAN CLUSTER KE DATA
# =================================================

df_encoded["cluster"] = clusters

# =================================================
# DATA UNTUK RANDOM FOREST
# =================================================

X_rf = df_encoded[
    rf_features
]

# =================================================
# PREDIKSI BOOKING STATUS
# =================================================

predictions = rf_model.predict(
    X_rf
)

prediction_label = np.where(
    predictions == 1,
    "Not_Canceled",
    "Canceled"
)

# =================================================
# DATA HASIL AKHIR
# =================================================

result_df = df.copy()

result_df["cluster"] = clusters

result_df["predicted_booking_status"] = (
    prediction_label
)

# =================================================
# MENAMPILKAN HASIL
# =================================================

st.subheader(
    "Hasil Clustering dan Prediksi"
)

st.dataframe(
    result_df
)

# =================================================
# DOWNLOAD CSV
# =================================================

csv = result_df.to_csv(
    index=False
)

st.download_button(
    label="Download Hasil CSV",
    data=csv,
    file_name="hasil_cluster_prediksi.csv",
    mime="text/csv"
)

# =================================================
# SILHOUETTE SCORE
# =================================================

sil_score = silhouette_score(
    latent_features,
    clusters
)

st.subheader(
    "Evaluasi Clustering"
)

st.metric(
    "Silhouette Score",
    round(sil_score, 4)
)

# =================================================
# SCATTER PLOT CLUSTER
# =================================================

st.subheader(
    "Scatter Plot 3 Cluster"
)

X_pca = pca.transform(
    latent_features
)

fig1, ax1 = plt.subplots(
    figsize=(10, 6)
)

scatter = ax1.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters
)

ax1.set_xlabel(
    "PCA 1"
)

ax1.set_ylabel(
    "PCA 2"
)

ax1.set_title(
    "Visualisasi Cluster DEC"
)

plt.colorbar(
    scatter
)

st.pyplot(fig1)

# =================================================
# DISTRIBUSI CLUSTER
# =================================================

st.subheader(
    "Distribusi Cluster"
)

fig2, ax2 = plt.subplots(
    figsize=(8, 5)
)

sns.countplot(
    x=clusters,
    ax=ax2
)

ax2.set_xlabel(
    "Cluster"
)

ax2.set_ylabel(
    "Jumlah Data"
)

ax2.set_title(
    "Distribusi Cluster"
)

st.pyplot(fig2)

# =================================================
# EVALUASI RANDOM FOREST
# =================================================

if y_true is not None:

    target_encoder = LabelEncoder()

    y_true_encoded = (
        target_encoder.fit_transform(
            y_true
        )
    )

    acc = accuracy_score(
        y_true_encoded,
        predictions
    )

    prec = precision_score(
        y_true_encoded,
        predictions
    )

    rec = recall_score(
        y_true_encoded,
        predictions
    )

    f1 = f1_score(
        y_true_encoded,
        predictions
    )

    st.subheader(
        "Evaluasi Prediksi Booking Cancellation"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        round(acc, 4)
    )

    col2.metric(
        "Precision",
        round(prec, 4)
    )

    col3.metric(
        "Recall",
        round(rec, 4)
    )

    col4.metric(
        "F1 Score",
        round(f1, 4)
    )

    # =============================================
    # CONFUSION MATRIX
    # =============================================

    st.subheader(
        "Confusion Matrix"
    )

    cm = confusion_matrix(
        y_true_encoded,
        predictions
    )

    fig3, ax3 = plt.subplots(
        figsize=(6, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax3
    )

    ax3.set_xlabel(
        "Predicted"
    )

    ax3.set_ylabel(
        "Actual"
    )

    st.pyplot(fig3)

else:

    st.info(
        "Silakan upload dataset CSV terlebih dahulu."
    )
