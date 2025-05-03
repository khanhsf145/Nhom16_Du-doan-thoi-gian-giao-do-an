import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Cấu hình trang (Tùy chọn) ---
st.set_page_config(
    page_title="Dự đoán thời gian giao đồ ăn",
    page_icon="🍲",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Hàm tải mô hình và preprocessor ---
# Sử dụng cache để tránh tải lại mỗi lần tương tác
@st.cache_resource # Dùng cache_resource cho các đối tượng lớn như model
def load_resources(model_path, preprocessor_path):
    if not os.path.exists(model_path):
        st.error(f"Lỗi: Không tìm thấy file mô hình tại '{model_path}'")
        return None, None
    if not os.path.exists(preprocessor_path):
        st.error(f"Lỗi: Không tìm thấy file preprocessor tại '{preprocessor_path}'")
        return None, None
    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        st.success("Đã tải thành công mô hình và preprocessor!")
        return model, preprocessor
    except Exception as e:
        st.error(f"Lỗi khi tải tài nguyên: {e}")
        return None, None

# --- Đường dẫn tới file đã lưu ---
MODEL_PATH = 'food_delivery_model.joblib'
PREPROCESSOR_PATH = 'food_delivery_preprocessor.joblib'

# --- Tải tài nguyên ---
model, preprocessor = load_resources(MODEL_PATH, PREPROCESSOR_PATH)

# --- Giao diện người dùng Streamlit ---
st.title("🍲 DỰ ĐOÁN THỜI GIAN GIAO ĐỒ ĂN")
st.markdown("""
Nhập các thông tin về đơn hàng và điều kiện giao hàng để dự đoán thời gian giao hàng dự kiến (tính bằng phút).
""")

# --- Chỉ hiển thị giao diện nhập liệu nếu tải tài nguyên thành công ---
if model is not None and preprocessor is not None:

    # --- Tạo các input widgets ---
    # Sử dụng các cột gốc mà preprocessor đã học

    # Tạo 2 cột để bố trí gọn gàng hơn (tùy chọn)
    col1, col2 = st.columns(2)

    with col1:
        distance = st.number_input("Khoảng cách giao hàng (km)", min_value=0.1, max_value=50.0, value=5.0, step=0.1, format="%.1f")
        prep_time = st.number_input("Thời gian chuẩn bị đơn (phút)", min_value=5, max_value=60, value=15, step=1)

        weather_options = ['Clear', 'Foggy', 'Rainy', 'Snowy', 'Windy']
        weather = st.selectbox("Điều kiện thời tiết", options=weather_options, index=0)

        traffic_options = ['Low', 'Medium', 'High']
        traffic = st.selectbox("Mức độ giao thông", options=traffic_options, index=1)

    with col2:
        courier_exp = st.number_input("Kinh nghiệm người giao hàng (năm)", min_value=0, max_value=30, value=2, step=1)

        time_options = ['Morning', 'Afternoon', 'Evening', 'Night']
        time_of_day = st.selectbox("Thời điểm giao hàng", options=time_options, index=1)

        vehicle_options = ['Bike', 'Scooter', 'Car']
        vehicle_type = st.selectbox("Loại phương tiện", options=vehicle_options, index=1)


    # --- Nút dự đoán ---
    predict_button = st.button("Dự đoán", type="primary")

    if predict_button:
        input_data = {
            'Distance_km': distance,
            'Weather': weather,
            'Traffic_Level': traffic,
            'Time_of_Day': time_of_day,
            'Vehicle_Type': vehicle_type,
            'Preparation_Time_min': prep_time,
            'Courier_Experience_yrs': courier_exp,
        }

        # Lấy danh sách tên cột gốc mà preprocessor đã học
        original_cols = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs',
                         'Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
        # Kiểm tra xem có cột nào thiếu trong input_data không
        missing_cols = [col for col in original_cols if col not in input_data]
        if missing_cols:
            st.warning(f"Thiếu dữ liệu đầu vào cho các cột: {missing_cols}. Sử dụng giá trị mặc định hoặc kiểm tra lại code.")

        # Tạo DataFrame từ input (chỉ 1 hàng)
        # Đảm bảo thứ tự cột khớp với lúc fit preprocessor nếu preprocessor không tự xử lý
        try:
            input_df = pd.DataFrame([input_data], columns=original_cols)
            st.write("Dữ liệu đầu vào:")
            st.dataframe(input_df)

            # --- Tiền xử lý dữ liệu input ---
            input_processed = preprocessor.transform(input_df)

            # --- Dự đoán ---
            prediction = model.predict(input_processed)
            predicted_time = round(prediction[0], 1) # Lấy kết quả đầu tiên và làm tròn

            # --- Hiển thị kết quả ---
            st.success(f"**Thời gian giao hàng được dự đoán là: {predicted_time} phút**")

        except Exception as e:
            st.error(f"Lỗi xảy ra trong quá trình xử lý hoặc dự đoán: {e}")
            st.error("Vui lòng kiểm tra lại dữ liệu đầu vào và cấu hình.")

else:
    st.warning("Không thể tải mô hình hoặc preprocessor. Vui lòng kiểm tra lại các file .joblib và đường dẫn.")