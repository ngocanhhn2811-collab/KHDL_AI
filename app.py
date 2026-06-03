# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import plotly.express as px

st.set_page_config(
    page_title="Phân tích báo cáo tài chính & phát hiện gian lận",
    layout="wide"
)

st.title("📊 Hệ thống phân tích báo cáo tài chính và phát hiện gian lận")
st.markdown(
    """
    Ứng dụng này cung cấp phân tích chuyên sâu báo cáo tài chính doanh nghiệp,
    bao gồm giám sát dữ liệu, thống kê, mô hình phân loại và phát hiện bất thường.
    """
)

DEFAULT_CSV = "Financial Statement Anomaly Dataset.csv"
default_path = Path.home() / "Downloads" / DEFAULT_CSV

st.sidebar.header("Tập dữ liệu")
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])
use_default = st.sidebar.checkbox("Sử dụng dữ liệu mẫu", value=True)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Tải dữ liệu upload thành công.")
    except Exception as e:
        st.sidebar.error(f"Không đọc được file upload: {e}")
        st.stop()
elif use_default and default_path.exists():
    try:
        df = pd.read_csv(default_path)
        st.sidebar.success(f"Đã tải dữ liệu mẫu từ {DEFAULT_CSV}.")
    except Exception as e:
        st.sidebar.error(f"Không đọc được file mẫu: {e}")
        st.stop()
else:
    st.sidebar.warning(
        "Chưa có dữ liệu. Vui lòng upload file CSV hoặc đặt file mẫu vào thư mục Downloads."
    )
    st.stop()

possible_targets = [
    "Financial_Status",
    "Anomaly",
    "Fraud",
    "Label",
    "Target",
    "Class"
]

default_label = next((c for c in possible_targets if c in df.columns), None)
selected_label = st.sidebar.selectbox(
    "Chọn cột nhãn:",
    options=df.columns.tolist(),
    index=df.columns.get_loc(default_label) if default_label is not None else 0,
)

if selected_label not in df.columns:
    st.error("Cột nhãn không tồn tại trong dữ liệu.")
    st.stop()

raw_y = df[selected_label]
y, y_labels = pd.factorize(raw_y)
label_decoder = {idx: label for idx, label in enumerate(y_labels)}

X = df.drop(columns=[selected_label])
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) == 0:
    st.error("Không có cột số nào để phân tích. Vui lòng kiểm tra file CSV.")
    st.stop()

X = X[numeric_cols].fillna(X[numeric_cols].mean())

st.sidebar.header("Cấu hình mô hình")
classifier_name = st.sidebar.selectbox(
    "Chọn mô hình phân loại:",
    ["Random Forest", "Logistic Regression"],
)

test_size = st.sidebar.slider(
    "Tỷ lệ dữ liệu kiểm tra:",
    min_value=0.1,
    max_value=0.5,
    value=0.2,
    step=0.05,
)

random_state = st.sidebar.number_input(
    "Random state:",
    min_value=0,
    max_value=9999,
    value=42,
)

contamination = st.sidebar.slider(
    "Tỷ lệ bất thường (IsolationForest):",
    min_value=0.01,
    max_value=0.2,
    value=0.05,
    step=0.01,
)

st.markdown("## Tổng quan dữ liệu")
col1, col2, col3 = st.columns(3)
col1.metric("Số dòng", df.shape[0])
col2.metric("Số cột", df.shape[1])
col3.metric("Số cột số", len(numeric_cols))

with st.expander("Xem trước dữ liệu"):
    st.dataframe(df.head())

with st.expander("Thống kê số học"):
    st.dataframe(df[numeric_cols].describe())

with st.expander("Phân phối nhãn"):
    label_counts = pd.Series(y).map(label_decoder).value_counts().reset_index()
    label_counts.columns = [selected_label, "Số lượng"]
    fig_label = px.bar(
        label_counts,
        x=selected_label,
        y="Số lượng",
        title="Phân phối nhãn trong dữ liệu",
        text="Số lượng",
    )
    st.plotly_chart(fig_label, width='stretch')

with st.expander("Ma trận tương quan"):
    corr = X.corr()
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Ma trận tương quan giữa các chỉ số tài chính",
    )
    st.plotly_chart(fig_corr, width='stretch')

st.markdown("## Huấn luyện mô hình phát hiện gian lận")
label_counts = pd.Series(y).value_counts()
if label_counts.min() < 2:
    st.warning(
        "Một số lớp nhãn có quá ít mẫu để stratify khi chia tập dữ liệu. "
        "Chuyển sang chia dữ liệu không stratify."
    )
    stratify_value = None
else:
    stratify_value = y

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    stratify=stratify_value,
    random_state=random_state,
)

if classifier_name == "Random Forest":
    model = RandomForestClassifier(n_estimators=200, random_state=random_state)
else:
    model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=random_state)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("Kết quả phân loại")
metric_col1, metric_col2 = st.columns(2)
metric_col1.metric("Accuracy", f"{acc * 100:.2f}%")
metric_col2.metric("Số lớp", len(label_decoder))

with st.expander("Báo cáo phân loại"):
    st.text(
        classification_report(
            y_test,
            y_pred,
            target_names=[label_decoder[i] for i in sorted(label_decoder)],
        )
    )

with st.expander("Confusion matrix"):
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(
        cm,
        labels={"x": "Dự đoán", "y": "Thực tế"},
        x=[label_decoder[i] for i in sorted(label_decoder)],
        y=[label_decoder[i] for i in sorted(label_decoder)],
        text_auto=True,
        title="Confusion Matrix",
    )
    st.plotly_chart(fig_cm, width='stretch')

importance_values = (
    model.feature_importances_
    if classifier_name == "Random Forest"
    else np.abs(model.coef_).flatten()
)
importance_df = pd.DataFrame(
    {"Feature": numeric_cols, "Importance": importance_values}
).sort_values(by="Importance", ascending=False)

with st.expander("Các chỉ số quan trọng nhất"):
    st.dataframe(importance_df.head(10))
    fig_imp = px.bar(
        importance_df.head(10),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 10 chỉ số quan trọng nhất",
    )
    st.plotly_chart(fig_imp, width='stretch')

st.markdown("## Phát hiện bất thường không giám sát")
use_unsupervised = st.checkbox("Kích hoạt phân tích bất thường không giám sát", value=True)
if use_unsupervised:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    iso.fit(X_scaled)
    anomaly_scores = -iso.score_samples(X_scaled)
    anomaly_labels = np.where(iso.predict(X_scaled) == -1, "Bất thường", "Bình thường")

    df_anomaly = df.copy()
    df_anomaly["AnomalyScore"] = anomaly_scores
    df_anomaly["AnomalyFlag"] = anomaly_labels

    st.write("### Top 20 bản ghi có nguy cơ bất thường cao")
    st.dataframe(df_anomaly.sort_values(by="AnomalyScore", ascending=False).head(20))

    anomaly_summary = df_anomaly["AnomalyFlag"].value_counts().reset_index()
    anomaly_summary.columns = ["Trạng thái", "Số lượng"]
    fig_anom = px.bar(
        anomaly_summary,
        x="Trạng thái",
        y="Số lượng",
        title="Tỷ lệ bất thường không giám sát",
        text="Số lượng",
    )
    st.plotly_chart(fig_anom, width='stretch')

    st.write("### Điểm bất thường trung bình theo nhãn thực tế")
    df_anomaly[selected_label] = raw_y
    score_by_label = df_anomaly.groupby(selected_label)["AnomalyScore"].mean().reset_index()
    fig_score = px.bar(
        score_by_label,
        x=selected_label,
        y="AnomalyScore",
        title="Điểm bất thường trung bình theo nhãn thực tế",
    )
    st.plotly_chart(fig_score, width='stretch')

st.markdown("## Dự đoán một báo cáo tài chính mới")
with st.form("predict_form"):
    input_cols = st.columns(2)
    sample_data = {}
    for idx, col_name in enumerate(numeric_cols):
        with input_cols[idx % 2]:
            sample_data[col_name] = st.number_input(
                col_name,
                value=float(X[col_name].mean()),
                format="%.4f",
            )
    submitted = st.form_submit_button("Dự đoán")

if submitted:
    sample_df = pd.DataFrame([sample_data], columns=numeric_cols)
    predicted = model.predict(sample_df)[0]
    predicted_label = label_decoder.get(predicted, str(predicted))

    st.write(f"**Kết quả dự đoán:** {predicted_label}")
    if str(predicted_label).strip().lower() != "normal":
        st.error("⚠️ Mẫu này được dự đoán là bất thường / gian lận tiềm năng.")
    else:
        st.success("✅ Mẫu này được dự đoán là bình thường.")

    if use_unsupervised:
        sample_scaled = scaler.transform(sample_df)
        unsup_pred = iso.predict(sample_scaled)[0]
        if unsup_pred == -1:
            st.warning("IsolationForest cũng đánh giá mẫu này là bất thường.")
        else:
            st.info("IsolationForest đánh giá mẫu này là bình thường.")
