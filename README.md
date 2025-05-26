# khntck

README :
Thực hiện trên ubuntu 20.04

Cài python3 và pip:

sudo apt update

sudo apt install python3 python3-pip

cài thư viện :

pip3 install requests pandas numpy tensorflow scikit-learn matplotlib flask

chạy:

python3 weather_forecast_app.py

Để tạo dữ liệu mới và huấn luyện lại: 

rm weather_dataset.csv temp_model.h5 rain_model.h5 scaler.pkl loss_history.pkl
rm -rf static/*

*khi huấn luyện lại thời gian khoảng 15p để lấy dữ liệu mới

api key từ openweather :d2d021ceb18c28747d865fd2b33d71ea

key dự phòng : b19d206dbc49beb17d642bcba1d7ed5b

