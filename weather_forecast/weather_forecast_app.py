import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import time
from datetime import datetime
from flask import Flask, render_template
import os
import pickle

# === Cấu hình Flask ===
app = Flask(__name__)

# === Đảm bảo thư mục static tồn tại
if not os.path.exists('static'):
    os.makedirs('static')

# === Cấu hình API ===
API_KEY = "d2d021ceb18c28747d865fd2b33d71ea"
CITY = "Hanoi"
URL = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"

# === Hàm lấy dữ liệu thời tiết từ API ===
def fetch_weather():
    try:
        response = requests.get(URL, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get('cod') != 200:
            raise Exception(f"API error: {data.get('message')}")
        main = data['main']
        temp = max(main['temp'], 20)  # Giới hạn nhiệt độ tối thiểu 20°C
        weather = {
            "temp": temp,
            "feels_like": main['feels_like'],
            "humidity": main['humidity'],
            "pressure": main['pressure'],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return weather
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu thời tiết: {e}")
        return None

# === Tạo dữ liệu huấn luyện mẫu ===
def collect_data(n=200):
    rows = []
    print("Đang thu thập dữ liệu thời tiết...")
    for i in range(n):
        weather = fetch_weather()
        if weather:
            current_hour = datetime.now().hour
            # Nhiệt độ tương lai giới hạn trong khoảng thực tế
            if 22 <= current_hour or current_hour < 6:  # Ban đêm
                future_temp = weather['temp'] + np.random.normal(-0.2, 0.1)  # Giảm nhiễu
            else:  # Ban ngày
                future_temp = weather['temp'] + np.random.normal(0.2, 0.1)
            future_temp = max(min(future_temp, 30), 20)  # Giới hạn 20-30°C
            if abs(future_temp - weather['temp']) > 1:  # Giới hạn thay đổi tối đa 1°C
                future_temp = weather['temp'] + (1 if future_temp > weather['temp'] else -1)
            row = [weather['temp'], weather['humidity'], weather['pressure'], future_temp, weather['timestamp']]
            rows.append(row)
        else:
            print(f"Không thể lấy dữ liệu lần {i+1}")
        time.sleep(2)  # Tránh vượt giới hạn API
    df = pd.DataFrame(rows, columns=["temp", "humidity", "pressure", "future_temp", "timestamp"])
    if df.empty or df.isna().sum().sum() > 0:
        print("Dữ liệu thu thập không hợp lệ hoặc chứa giá trị NaN")
        return pd.DataFrame(columns=["temp", "humidity", "pressure", "future_temp", "timestamp"])
    # Kiểm tra dữ liệu bất thường
    if (df['temp'] < 20).any() or (df['future_temp'] < 20).any():
        print("Dữ liệu chứa nhiệt độ bất thường (<20°C)")
        return pd.DataFrame(columns=["temp", "humidity", "pressure", "future_temp", "timestamp"])
    print("Dữ liệu thu thập:")
    print(df.describe())
    print("Mẫu dữ liệu đầu tiên:")
    print(df.head())
    return df

# === Huấn luyện mô hình AI ===
def train_model(df):
    scaler = MinMaxScaler()
    X = df[["temp", "humidity", "pressure"]]
    y = df["future_temp"]
    X_scaled = scaler.fit_transform(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print("y_train mẫu:", y_train[:10])
    print("y_test mẫu:", y_test[:10])

    model_path = 'weather_model.h5'
    scaler_path = 'scaler.pkl'

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = tf.keras.models.load_model(model_path)
            print("Tải mô hình từ weather_model.h5")
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print("Tải scaler từ scaler.pkl")
            history = None
            if os.path.exists('loss_history.pkl'):
                with open('loss_history.pkl', 'rb') as f:
                    history_dict = pickle.load(f)
                print("Tải history từ loss_history.pkl")
                history = tf.keras.callbacks.History()
                history.history = history_dict
        except Exception as e:
            print(f"Lỗi khi tải mô hình hoặc scaler: {e}")
            model = None
            history = None
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=500, verbose=1, validation_split=0.2, callbacks=[early_stopping])
        model.save(model_path)
        print("Đã lưu mô hình vào weather_model.h5")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print("Đã lưu scaler vào scaler.pkl")
        with open('loss_history.pkl', 'wb') as f:
            pickle.dump(history.history, f)
        print("Đã lưu history vào loss_history.pkl")

    if model:
        loss = model.evaluate(X_test, y_test, verbose=0)
        print(f"MSE trên tập kiểm tra: {loss:.4f}")
    else:
        loss = None

    return model, scaler, history

# === Dự đoán và lưu biểu đồ ===
def predict_and_visualize(model, scaler, df, history):
    current = fetch_weather()
    if not current:
        return None, None, None, False

    X_input = [[current['temp'], current['humidity'], current['pressure']]]
    try:
        X_input_scaled = scaler.transform(X_input)
        print("X_input:", X_input)
        print("X_input_scaled:", X_input_scaled)
        if (X_input_scaled < 0).any() or (X_input_scaled > 1).any():
            print("X_input_scaled ngoài khoảng [0, 1]")
    except Exception as e:
        print(f"Lỗi khi chuẩn hóa dữ liệu đầu vào: {e}")
        return None, None, None, False

    prediction = model.predict(X_input_scaled, verbose=0)[0][0]
    prediction = max(min(prediction, 30), 20)  # Giới hạn 20-30°C
    if abs(prediction - current['temp']) > 1:  # Giới hạn thay đổi tối đa 1°C
        prediction = current['temp'] + (1 if prediction > current['temp'] else -1)
    print("Prediction:", prediction)

    plt.figure(figsize=(10, 6))
    timestamps = df['timestamp'].iloc[-10:].tolist() + [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    temps = df['temp'].iloc[-10:].tolist() + [current['temp']]
    predicted_temps = df['future_temp'].iloc[-10:].tolist() + [prediction]

    plt.plot(timestamps, temps, marker='o', label='Nhiệt độ thực tế (°C)', color='#1f77b4')
    plt.plot(timestamps, predicted_temps, marker='x', linestyle='--', label='Nhiệt độ dự đoán (°C)', color='#ff7f0e')
    plt.xticks(rotation=45)
    plt.xlabel('Thời gian')
    plt.ylabel('Nhiệt độ (°C)')
    plt.title('So sánh Nhiệt độ Thực tế và Dự đoán tại Hà Nội')
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/temperature_plot.png')
    plt.close()

    loss_plot_exists = False
    if history is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(history.history['loss'], label='Loss huấn luyện', color='#1f77b4')
        plt.plot(history.history['val_loss'], label='Loss xác thực', color='#ff7f0e')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title('Loss trong Quá trình Huấn luyện')
        plt.legend()
        plt.savefig('static/loss_plot.png')
        plt.close()
        loss_plot_exists = True

    return current, prediction, history.history['loss'][-1] if history else None, loss_plot_exists

# === Route chính cho web ===
@app.route('/')
def index():
    if os.path.exists('weather_dataset.csv'):
        df = pd.read_csv('weather_dataset.csv')
        print("Tải dữ liệu từ weather_dataset.csv")
        print(df.describe())
        print("Mẫu dữ liệu đầu tiên:")
        print(df.head())
        # Kiểm tra dữ liệu bất thường
        if (df['temp'] < 20).any() or (df['future_temp'] < 20).any():
            print("Dữ liệu CSV chứa nhiệt độ bất thường (<20°C). Xóa và tạo lại...")
            os.remove('weather_dataset.csv')
            df = collect_data(n=200)
        else:
            print("Dữ liệu CSV hợp lệ")
    else:
        df = collect_data(n=200)

    if df.empty:
        error = "Không thu thập được dữ liệu. Kiểm tra API key hoặc kết nối mạng."
        return render_template('index.html', error=error)
    df.to_csv("weather_dataset.csv", index=False)
    print("Đã lưu dữ liệu vào weather_dataset.csv")

    model, scaler, history = train_model(df)
    current, prediction, mse, loss_plot_exists = predict_and_visualize(model, scaler, df, history)

    if current is None:
        error = "Không thể lấy dữ liệu thời tiết hiện tại để dự đoán."
        return render_template('index.html', error=error)

    return render_template('index.html',
                           temp=current['temp'],
                           humidity=current['humidity'],
                           pressure=current['pressure'],
                           feels_like=current['feels_like'],
                           prediction=prediction,
                           mse=mse,
                           timestamp=current['timestamp'],
                           loss_plot_exists=loss_plot_exists)

# === Main ===
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)