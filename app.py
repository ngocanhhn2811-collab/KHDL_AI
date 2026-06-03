# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import plotly.express as px

# =========================
# Cấu hình trang
# =========================
st.set_page_config(
    page_title="Phân tích Báo cáo Tài chính",
    layout="wide"
)

st.title("📊 Hệ thống Phân tích Báo cáo Tài chính")
st.markdown("""
Ứng dụng sử dụng Machine Learning để phát hiện các dấu hiệu bất thường
trong báo cáo tài chính doanh nghiệp.
""")

# =========================
# Tải dữ liệu
# =========================
DATA_FILENAME = "Financial Statement Anomaly Dataset.csv"
local_path = Path(DATA_FILENAME)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Đã tải dữ liệu từ file upload thành công.")
    except Exception as e:
        st.error(f"Không đọc được file upload: {e}")
        st.stop()

elif local_path.exists():
    try:
        df = pd.read_csv(local_path)
        st.success(f"Đã tải dữ liệu từ {DATA_FILENAME}.")
    except Exception as e:
        st.error(f"Không đọc được file CSV: {e}")
        st.stop()

else:
    st.warning(
        "Chưa có file CSV để đọc. Vui lòng upload file hoặc đặt file "
        f"'{DATA_FILENAME}' cùng thư mục với app.py."
    )
    st.stop()

st.subheader("Dữ liệu mẫu")
st.dataframe(df.head())

# =========================
# Thông tin dữ liệu
# =========================
st.subheader("Thông tin dữ liệu")

col1, col2 = st.columns(2)
with col1:
    st.metric("Số dòng", df.shape[0])
with col2:
    st.metric("Số cột", df.shape[1])

# =========================
# Xác định cột nhãn
# =========================
possible_targets = ["Anomaly", "Fraud", "Label", "Target", "Class", "Financial_Status"]

found_targets = [col for col in possible_targets if col in df.columns]

if found_targets:
    target_col = found_targets[0]
    st.info(f"Tự động chọn cột nhãn: {target_col}")
else:
    st.warning(
        "Không tìm thấy cột nhãn mặc định (Anomaly/Fraud/Label/Target/Class/Financial_Status)."
    )
    target_col = st.selectbox(
        "Chọn cột nhãn trong dữ liệu:",
        options=df.columns.tolist(),
        help="Chọn cột chứa thông tin trạng thái bất thường/gian lận."
    )

if not target_col:
    st.error("Vui lòng chọn một cột nhãn để tiếp tục.")
    st.stop()

# =========================
# Tiền xử lý dữ liệu
# =========================
y = df[target_col]
if y.dtype == object or y.dtype == bool:
    y = pd.factorize(y)[0]

X = df.drop(columns=[target_col])
X = X.select_dtypes(include=np.number)
X = X.fillna(X.mean())

if X.shape[1] == 0:
    st.error("Không có cột số để huấn luyện mô hình. Vui lòng kiểm tra dữ liệu CSV.")
    st.stop()

# =========================
# Chia tập dữ liệu
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Huấn luyện mô hình
# =========================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =========================
# Đánh giá
# =========================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Độ chính xác mô hình")
st.metric(label="Accuracy", value=f"{accuracy*100:.2f}%")

# =========================
# Biểu đồ phân bố nhãn
# =========================
st.subheader("Phân bố dữ liệu")
fig = px.histogram(df, x=target_col, title="Phân bố bất thường và bình thường")
st.plotly_chart(fig, use_container_width=True)

# =========================
# Tầm quan trọng thuộc tính
# =========================
st.subheader("Các chỉ số ảnh hưởng nhiều nhất")
importance = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
importance = importance.sort_values(by="Importance", ascending=False)
fig2 = px.bar(
    importance.head(10),
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top 10 thuộc tính quan trọng"
)
st.plotly_chart(fig2, use_container_width=True)

# =========================
# Dự đoán mẫu mới
# =========================
st.subheader("Kiểm tra doanh nghiệp")
input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(col, value=float(X[col].mean()))

if st.button("Dự đoán"):
    sample = pd.DataFrame([
        {col: input_data.get(col, float(X[col].mean())) for col in X.columns}
    ])
    sample = sample[X.columns]
    prediction = model.predict(sample)[0]
    if prediction == 1:
        st.error("⚠️ Có dấu hiệu bất thường/gian lận")
    else:
        st.success("✅ Báo cáo tài chính bình thường")
    
