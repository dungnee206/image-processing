import os
import sys
import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image

# Thêm thư mục src vào PATH để import các module đã viết
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from dataset import CLASSES
from preprocessing import preprocess_image
from features import extract_single_image_features

st.set_page_config(
    page_title="Nhận diện Khuyết tật Bề mặt Thép (NEU-DET)",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Hệ Thống Phân Loại Khuyết Tật Bề Mặt Kim Loại")
st.markdown("---")

# Đường dẫn tới mô hình và scaler
MODEL_PATH = os.path.join("results", "best_model.pkl")
SCALER_PATH = os.path.join("results", "scaler.pkl")

@st.cache_resource
def load_trained_pipeline():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_trained_pipeline()

if model is None or scaler is None:
    st.error("⚠️ Chưa tìm thấy file `best_model.pkl` hoặc `scaler.pkl` trong thư mục `results/`. Hãy chạy `python src/train.py` trước!")
    st.stop()

st.sidebar.header("⚙️ Tùy chọn Tiền xử lý Ảnh")
prep_method = st.sidebar.selectbox(
    "Bộ lọc nâng cao chất lượng ảnh:",
    options=["clahe", "raw", "hist_eq", "median", "sobel", "laplacian"],
    format_func=lambda x: {
        "raw": "Ảnh gốc (Raw)",
        "hist_eq": "Cân bằng Histogram toàn cục",
        "clahe": "CLAHE (Cân bằng tương phản thích nghi)",
        "median": "Lọc trung vị (Median Filter)",
        "sobel": "Tách biên Sobel Gradient",
        "laplacian": "Tách biên Laplacian"
    }[x]
)

st.sidebar.info(
    "**6 Lớp khuyết tật chính:**\n"
    "1. `crazing`: Nứt rạn\n"
    "2. `inclusion`: Tạp chất\n"
    "3. `patches`: Mảng bám\n"
    "4. `pitted_surface`: Rỗ bề mặt\n"
    "5. `rolled-in_scale`: Vảy cán\n"
    "6. `scratches`: Trầy xước"
)

uploaded_file = st.file_uploader("Chọn ảnh khuyết tật để kiểm tra (.bmp, .jpg, .png):", type=["bmp", "jpg", "jpeg", "png"])

if uploaded_file is not None:
  
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    

    if img_gray.shape != (200, 200):
        img_gray = cv2.resize(img_gray, (200, 200))
        
    filtered_img = preprocess_image(img_gray, method=prep_method)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🖼️ Ảnh Gốc (Grayscale 200x200)")
        st.image(img_gray, clamp=True, use_container_width=True)
        
    with col2:
        st.subheader(f"✨ Ảnh Đấ Xử Lý ({prep_method.upper()})")
        st.image(filtered_img, clamp=True, use_container_width=True)

    # Quy trình chuẩn: Tiền xử lý bằng CLAHE -> Trích xuất đặc trưng -> Scale -> Dự đoán
    prep_for_model = preprocess_image(img_gray, method="clahe")
    feats = extract_single_image_features(prep_for_model).reshape(1, -1)
    feats_scaled = scaler.transform(feats)

    pred_idx = model.predict(feats_scaled)[0]
    pred_label = CLASSES[pred_idx]

    # Kiểm tra xem có tính toán xác suất (predict_proba) hay không
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feats_scaled)[0]

    st.markdown("---")
    st.subheader("🎯 Kết Quả Phân Loại Khuyết Tật")

    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.success(f"**Khuyết tật nhận diện:** `{pred_label.upper()}`")
        if probs is not None:
            confidence = probs[pred_idx] * 100
            st.metric(label="Độ tin cậy (Confidence)", value=f"{confidence:.2f}%")

    with res_col2:
        if probs is not None:
            st.write("📊 Phân bố xác suất trên 6 lớp:")
            prob_dict = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
            st.bar_chart(prob_dict)