from flask import Flask, request, render_template, jsonify
import os
import sys
import base64
import io
from PIL import Image

# 修正相對路徑導入問題
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.predict import predict

app = Flask(__name__)

# 確保目錄存在
os.makedirs("app/static", exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_digit():
    try:
        # 檢查是否有上傳文件
        if 'file' in request.files:
            file = request.files["file"]
            file_path = os.path.join("app/static", "uploaded.png")
            file.save(file_path)
        # 或者檢查是否有 canvas 數據
        elif 'image_data' in request.form:
            image_data = request.form['image_data']
            # 從 base64 數據中解析圖片
            image_data = image_data.split(',')[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
            file_path = os.path.join("app/static", "canvas.png")
            image.save(file_path)
        else:
            return jsonify({"error": "沒有提供圖片"}), 400
        
        # 進行預測
        prediction, confidence = predict(file_path)
        
        return jsonify({"prediction": prediction, "confidence": f"{confidence:.2f}%"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)