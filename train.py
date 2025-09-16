# train.py (이미지 전용 평발 판별용)

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import os

# ==========================
# 1. 설정
# ==========================
data_dir = r"C:\Users\박시환\OneDrive - Testworks\바탕 화면\Flat_foot_examination\data"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
num_epochs = 15
img_size = 224
num_workers = 0  # Windows 환경 안정화
save_path = os.path.join(data_dir, "best_flat_foot_model.pth")

# ==========================
# 2. 데이터셋 & 전처리
# ==========================
transform_train = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

# ==========================
# 3. 모델 설정
# ==========================
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))  # 클래스 수 자동 설정
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ==========================
# 4. 학습 루프
# ==========================
best_val_acc = 0.0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    running_corrects = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        running_corrects += torch.sum(torch.argmax(outputs, 1) == labels).item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects / len(train_dataset)

    # 검증
    model.eval()
    val_corrects = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            val_corrects += torch.sum(torch.argmax(outputs, 1) == labels).item()

    val_acc = val_corrects / len(val_dataset)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}")

print(f"\n최고 검증 정확도: {best_val_acc:.4f}")
print(f"모델 저장 완료: {save_path}")

# ==========================
# 5. 새 이미지 평발 판별 함수
# ==========================
def predict_flat_foot(model, image_path):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    inp = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(inp)
        pred = torch.argmax(output, dim=1).item()
    return train_dataset.classes[pred]

# ==========================
# 6. 테스트 예시
# ==========================
test_image = os.path.join(test_dir, "sample.jpg")
model.load_state_dict(torch.load(save_path, map_location=device))
result = predict_flat_foot(model, test_image)
print(f"이미지 '{os.path.basename(test_image)}' 판별 결과: {result}")
