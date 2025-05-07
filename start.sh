pip3 install -r requirements.txt
pip3 uninstall opencv-python opencv-contrib-python opencv-python-headless -y
python3 src/preprocessing/train_scaler.py
python3 src/anomaly_detector/train_ae.py
python3 src/rl_agent/train_agent.py
python3 src/backend/main.py

