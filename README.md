# 手寫數字識別

簡單的手寫數字識別系統(阿拉伯數字)，使用 CNN 、 MNIST數據集。包含模型訓練、及 Flask Web 來操作辨識系統。

---

## 主要功能

**1️⃣ model/mnist_cnn.py**  
定義 CNN 模型，並提供 `save_model()` 和 `load_model()` 函數來存取和載入模型權重。

**2️⃣ model/train.py**  
載入 MNIST 數據集，訓練模型並儲存訓練後的模型。

**3️⃣ model/predict.py**  
載入訓練好的模型並對圖片判讀。

**4️⃣ app/app.py（選配）**  
提供 Web 介面，可以上傳手寫數字圖片並進行數字判別。

**5️⃣ requirements.txt**  
所需之 Python 套件。

---

## 使用方法

**1️⃣** 執行 `mnist_cnn.py`　來生成 PyTorch 模型（`mnist_cnn.pth`）。

**2️⃣** 執行 `train.py` 以載入 MNIST 數據集、訓練模型並儲存。

**3️⃣** 執行 `app.py` 以啟動 Flask 伺服器。可以上傳手寫數字圖片，並進行判別是什麼數字。

---

提醒：在運行之前，請先安裝 `requirements.txt` 中的所有套件。
