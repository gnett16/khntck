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

rm weather_model.h5 scaler.pkl loss_history.pkl weather_dataset.csv

*khi huấn luyện lại thời gian khoảng 15p để lấy dữ liệu mới
