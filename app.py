#C:\Users\박시환\AppData\Local\Programs\Python\Python313\python.exe -m streamlit run app.py

import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import os

# ==========================
# 1. 설정
# ==========================
data_dir = r"C:\Users\박시환\OneDrive - Testworks\바탕 화면\Flat_foot_examination\data"
model_path = os.path.join(data_dir, "best_flat_foot_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 224

# ==========================
# 2. 모델 로드
# ==========================
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 클래스 수
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ==========================
# 3. Streamlit UI
# ==========================
st.set_page_config(page_title="평발 판별기", layout="centered")
st.title("👣 이미지 기반 평발 판별기")
st.write("이미지를 업로드하면 평발 여부를 판별합니다.")

uploaded_file = st.file_uploader("이미지 선택", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
   
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="업로드된 이미지", use_column_width=True)

   
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    inp = transform(image).unsqueeze(0).to(device)

  
    with torch.no_grad():
        output = model(inp)
        pred = torch.argmax(output, dim=1).item()

  
    classes = ["normal", "flat_foot"]
    st.success(f"✅ 판별 결과: {classes[pred]}")

