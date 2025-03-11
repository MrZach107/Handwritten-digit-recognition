# 手寫數字識別

這個專案展示了一個簡單的手寫數字識別系統，使用卷積神經網路（CNN）和 MNIST 數據集。它包含了模型訓練、預測以及一個 Flask Web 應用來與模型互動。

---

## 主要功能

📌 **1️⃣ model/mnist_cnn.py**  
定義 CNN 模型，並提供 `save_model()` 和 `load_model()` 函數來存取和載入模型權重。

📌 **2️⃣ model/train.py**  
載入 MNIST 數據集，訓練模型並儲存訓練後的模型權重。

📌 **3️⃣ model/predict.py**  
載入訓練好的模型並對單張圖片進行預測。

📌 **4️⃣ app/app.py（選配）**  
提供 Web 介面，讓使用者上傳手寫數字圖片並使用訓練好的模型進行預測。

📌 **5️⃣ requirements.txt**  
列出專案運行所需的 Python 套件。

---

## 使用方法

📢 **1️⃣** 執行 `mnist_cnn.py` 來生成 PyTorch 模型（`mnist_cnn.pth`）。

📢 **2️⃣** 執行 `train.py` 來載入 MNIST 數據集、訓練模型並儲存模型權重。

📢 **3️⃣** 執行 `app.py` 來啟動 Flask 伺服器。然後您可以上傳手寫數字圖片，並使用訓練好的模型進行預測。

---

在運行專案之前，請先安裝 `requirements.txt` 中列出的所有必要套件。