import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Tải model YOLOv8s (pretrained COCO)
@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")

model = load_model()

st.title("📷 YOLOv8 Object Detection")
st.markdown(
    "Mở trang web này trên **điện thoại** và chụp ảnh bằng camera để phát hiện đối tượng."
)

# --- Chụp ảnh trực tiếp từ camera của thiết bị ---
picture = st.camera_input("Chụp ảnh")

if picture:
    # Đọc ảnh từ camera_input
    img = Image.open(picture)

    # Dự đoán
    results = model.predict(img)

    # Vẽ bounding boxes
    res_img = results[0].plot()        # numpy array BGR
    res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)

    st.image(res_img, caption="Kết quả phát hiện", use_column_width=True)

    # (Tùy chọn) Hiển thị các class và độ tin cậy
    for box in results[0].boxes:
        cls_name = model.names[int(box.cls)]
        conf = float(box.conf)
        st.write(f"- {cls_name}: {conf:.2f}")
