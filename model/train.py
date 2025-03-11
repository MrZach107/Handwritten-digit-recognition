import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, ConcatDataset
import os
import sys
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

# 修正相對路徑導入問題
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.mnist_cnn import CNN, save_model

# 自定義資料增強轉換
class CustomTransform:
    def __call__(self, img):
        # 套用多種變形來增強對數字9和7的辨識能力
        transforms_list = [
            lambda x: x,  # 原始圖像
            lambda x: x.transform(x.size, Image.AFFINE, (1, 0, 0, 0.176, 1, 0)),  # shear=10
            lambda x: x.transform(x.size, Image.AFFINE, (1, 0, 0, -0.176, 1, 0)),  # shear=-10
            lambda x: ImageEnhance.Contrast(x).enhance(0.8),
            lambda x: ImageEnhance.Contrast(x).enhance(1.2)
        ]
        
        # 隨機選擇一種變換
        transform = np.random.choice(transforms_list)
        return transform(img)

def train_model():
    # 檢查是否有 CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware/Device: {device}")

    # 1. 載入 MNIST 數據集，添加自定義轉換
    custom_transform = transforms.Compose([
        CustomTransform(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 基本轉換
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 確保數據目錄存在
    os.makedirs("./data", exist_ok=True)
    
    # 載入基本數據集
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=basic_transform)
    
    # 增強版數據集
    train_dataset_augmented = datasets.MNIST(root="./data", train=True, download=True, transform=custom_transform)
    
    # 特別處理9和7的樣本 - 創建額外的增強樣本
    indices_9 = [i for i, (_, label) in enumerate(train_dataset) if label == 9]
    indices_7 = [i for i, (_, label) in enumerate(train_dataset) if label == 7]
    
    # 提取這些樣本並進行多次增強
    extra_9_dataset = torch.utils.data.Subset(train_dataset_augmented, indices_9 * 3)  # 複製3次
    extra_7_dataset = torch.utils.data.Subset(train_dataset_augmented, indices_7 * 3)  # 複製3次
    
    # 合併數據集
    combined_dataset = ConcatDataset([train_dataset, extra_9_dataset, extra_7_dataset])
    
    train_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)
    
    # 測試集用於評估
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=basic_transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 2. 初始化 CNN 模型
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 添加學習率調度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    # 3. 訓練模型
    epochs = 10  # 增加訓練輪數
    best_accuracy = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} [{batch_idx * len(images)}/{len(train_loader.dataset)}] - Loss: {loss.item():.6f}")
        
        # 評估模型
        model.eval()
        test_loss = 0
        correct = 0
        confusion_matrix = torch.zeros(10, 10)  # 創建混淆矩陣
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                test_loss += criterion(output, labels).item()
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == labels).sum().item()
                
                # 更新混淆矩陣
                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        
        test_loss /= len(test_loader)
        accuracy = 100 * correct / len(test_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {running_loss/len(train_loader):.6f}, Test Loss: {test_loss:.6f}, Accuracy: {accuracy:.2f}%")
        
        # 輸出9和7的預測結果
        print(f"Accuracy of digit 9: {confusion_matrix[9, 9]/confusion_matrix[9, :].sum()*100:.2f}%")
        print(f"Misclassification of digit 9 as 7: {confusion_matrix[9, 7]/confusion_matrix[9, :].sum()*100:.2f}%")
        print(f"Accuracy of digit 7: {confusion_matrix[7, 7]/confusion_matrix[7, :].sum()*100:.2f}%")
        print(f"Misclassification of digit 7 as 9: {confusion_matrix[7, 9]/confusion_matrix[7, :].sum()*100:.2f}%")

        # 使用學習率調度器
        scheduler.step(test_loss)
        
        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_model(model, path="saved_models/best_cnn_model.pth")
            print(f"New best model saved, accuracy: {accuracy:.2f}%")

    # 4. 儲存最終模型
    save_model(model)
    print(f"Model training completed and saved! Best accuracy: {best_accuracy:.2f}%")

if __name__ == "__main__":
    train_model()

