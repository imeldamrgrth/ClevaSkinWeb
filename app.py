import streamlit as st
import numpy as np
import tensorflow as tf
import json
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ====== Konfigurasi Halaman ======
st.set_page_config(page_title="Diagnosa Penyakit Kulit", layout="wide", page_icon="🧴")

# ====== Custom CSS Styling ======
st.markdown("""
    <style>
    .center-container {
        max-width: 1100px;
        margin: 0 auto;
        padding: 20px;
    }
    .info-box {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .alert-box {
        background-color: #b8b369;
        color: #000;
        padding: 12px;
        border-radius: 8px;
        margin-top: 15px;
    }
    h1, h2, h3, h4 {
        font-family: 'Segoe UI', sans-serif;
    }
    ul {
        padding-left: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ====== Load Model Sekali ======
@st.cache_resource
def load_model_once():
    return load_model('best_model_finetuned.h5')

model = load_model_once()

# ====== Load Label & Edukasi ======
@st.cache_data
def load_class_labels():
    with open('class_labels.json', 'r') as f:
        return json.load(f)

with open("edukasi.json", "r", encoding="utf-8") as f:
    penyakit_info = json.load(f)

class_labels = load_class_labels()

# ====== Preprocessing ======
def preprocess_image(img):
    img_resized = img.resize((224, 224))
    img_array = img_to_array(img_resized)
    img_array = img_array.astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0), img_array

# ====== Edukasi HTML Parser ======
def render_edukasi(edukasi_items):
    html = "<ul>"
    for item in edukasi_items:
        if isinstance(item, dict) and "sub" in item:
            html += f"<li>{item['text']}<ul>"
            for sub in item["sub"]:
                html += f"<li>{sub}</li>"
            html += "</ul></li>"
        else:
            html += f"<li>{item}</li>"
    html += "</ul>"
    return html

# ====== Judul Halaman ======
st.markdown("<h2 style='text-align:center;'>🧴 Diagnosa Kondisi Kulit dengan AI</h2>", unsafe_allow_html=True)

# ====== Petunjuk Upload ======
with st.expander("📸 Petunjuk Upload Gambar", expanded=True):
    st.markdown("""
    Harap unggah foto **area kulit** dengan pencahayaan yang cukup dan **tanpa make-up, filter, atau masker**.  
    Hindari penutup seperti kacamata atau rambut menutupi area kulit.  
    <br><i>AI akan menganalisis dan memberikan hasil prediksi seakurat mungkin.</i>
    """, unsafe_allow_html=True)

# ====== Upload Gambar ======
uploaded_file = st.file_uploader("Unggah gambar kulit Anda (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

# ====== Jika Gambar Diupload ======
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    x, img_array = preprocess_image(image)

    preds = model.predict(x)[0]
    sorted_preds = sorted(zip(class_labels, preds), key=lambda x: x[1], reverse=True)
    pred_label, confidence = sorted_preds[0]
    info = penyakit_info.get(pred_label, {})

    # Deteksi apakah kategori spesial
    skip_detail = pred_label in ["Non-Skin", "Normal Skin", "Unknown"]

    # Render edukasi dan gejala (jika tidak diskip)
    edukasi_html = render_edukasi(info.get("edukasi", [])) if not skip_detail else ""
    gejala_html = render_edukasi(info.get("gejala", [])) if not skip_detail else ""

    # ====== Container Tengah ======
    st.markdown('<div class="center-container">', unsafe_allow_html=True)
    st.markdown("## ✅ Hasil Prediksi")

    # Layout Flex
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.image(image, caption="🖼️ Gambar yang Diunggah", use_container_width=True)

    with col2:
        st.markdown(f"""
        <div class="info-box">
            <h3 style="font-size: 30px; color:#ffc8dd;">🧠 Diagnosis: <b>{pred_label}</b></h3>
            <p style="font-size: 18px;"><strong>Akurasi:</strong> {confidence:.2%}</p>
        </div>

        <h4 style="margin-top:60px;">📘 Informasi Kondisi Kulit</h4>
        {"<ul style='line-height: 1.8; font-size: 18px;'>"
         f"<li>📖 <strong>Penjelasan:</strong> {info.get('penjelasan', '-')}</li>"
         f"<li>🧬 <strong>Gejala:</strong> {gejala_html}</li>"
         f"<li>💡 <strong>Edukasi:</strong> {edukasi_html}</li></ul>" if not skip_detail else f"<p style='font-size:16px;'><strong>📖 Penjelasan:</strong> {info.get('penjelasan', '-')}</p>"}

        <div class="alert-box">
            ⚠️ {info.get('warning', 'Tidak ada peringatan khusus.')}
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ====== Footer ======
st.markdown("""
    <hr style='margin-top:50px;'>
    <div style='text-align:center; color:gray; font-size:13px;'>© 2025 ClevaSkin AI</div>
""", unsafe_allow_html=True)
