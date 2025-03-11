# Handwritten-Digit-Recognition/
# │── data/                         # MNIST 數據集
# │── model/                        # 儲存模型架構 & 預測程式
# │   ├── __init__.py               # 讓 model 成為 Python 模組
# │   ├── mnist_cnn.py              # CNN 模型
# │   ├── train.py                  # 訓練模型
# │   ├── predict.py                # 載入模型並做預測
# │── saved_models/                 # 訓練後的模型權重
# │   ├── best_cnn_model.pth        # 儲存的 PyTorch 模型
# │── app/                          # Flask Web 應用
# │   ├── app.py                    # Flask 伺服器
# │   ├── static/                   # 前端靜態檔案 (CSS, JS)
# │   ├── templates/                # 存放前端 HTML 頁面
# │       ├── index.html            # HTML 頁面本體
# │── requirements.txt              # 需要安裝的套件
# │── README.md                     # 專案說明

---

📌 1️⃣ model/mnist_cnn.py（CNN 模型架構）
>> 負責定義 CNN 模型，並提供 save_model() 和 load_model() 來存取模型權重。
📌 2️⃣ model/train.py（訓練模型）
>> 負責載入 MNIST 數據集、訓練模型，並儲存權重。
📌 3️⃣ model/predict.py（模型預測）
>> 負責載入已訓練的模型，並對單張圖片做預測。
📌 4️⃣ app/app.py（Web 介面）（選配）
>> 讓使用者上傳手寫數字圖片，並透過模型進行預測。
📌 5️⃣ requirements.txt
>> 列出部署專案所需要安裝的 Python 套件。

---

🛠️使用方法
📢 1️⃣ 執行mnist_cnn.py，生成PyTorch模型(mnist_cnn.pth)。
📢 2️⃣ 執行train.py，載入MNIST數據集、訓練模型。
📢 3️⃣ 執行app.py，之後便可進入網站手寫數字圖片，並利用模型進行預測。