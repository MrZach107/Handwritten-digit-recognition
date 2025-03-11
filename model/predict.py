import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import numpy as np
import cv2

# 修正相對路徑導入問題
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.mnist_cnn import load_model

def preprocess_image(image_path):
    # 讀取圖片
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # 如果 OpenCV 無法讀取，嘗試用 PIL
        img = np.array(Image.open(image_path).convert('L'))
    
    # 二值化處理 (將灰度圖轉為黑白圖)
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    
    # 找到數字的輪廓
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 獲取數字的邊界框
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        
        # 裁剪數字區域
        digit = img[y:y+h, x:x+w]
        
        # 增加填充以保持寬高比
        aspect_ratio = float(w) / h
        if aspect_ratio > 1:
            new_w = 20
            new_h = int(new_w / aspect_ratio)
        else:
            new_h = 20
            new_w = int(new_h * aspect_ratio)
            
        digit = cv2.resize(digit, (new_w, new_h))
        
        # 在28x28畫布上居中
        result = np.zeros((28, 28), dtype=np.uint8)
        offset_x = (28 - new_w) // 2
        offset_y = (28 - new_h) // 2
        result[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = digit
        
        return result
    
    return img

def predict(image_path):
    try:
        # 1. 載入模型
        model = load_model()
        
        # 檢查是否有 CUDA
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # 2. 預處理圖片
        # 使用 OpenCV 進行預處理
        preprocessed_img = preprocess_image(image_path)
        
        # 將 OpenCV 格式轉換為 PIL 格式，然後進行 PyTorch 的標準化處理
        pil_img = Image.fromarray(preprocessed_img)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        image = transform(pil_img).unsqueeze(0).to(device)

        # 3. 進行預測
        with torch.no_grad():
            output = model(image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            prediction = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][prediction].item() * 100
        
        return prediction, confidence
    except Exception as e:
        print(f"預測時發生錯誤: {e}")
        return None, 0

if __name__ == "__main__":
    # 測試預測
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        prediction, confidence = predict(image_path)
        if prediction is not None:
            print(f"預測結果: {prediction}, 信心度: {confidence:.2f}%")
    else:
        print("請提供圖片路徑進行預測")