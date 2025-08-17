import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt

Base_dir = os.path.dirname(__file__)
model_dir = os.path.join(Base_dir, "Model")
model_path = os.path.join(model_dir, "fashion_cnn.pth")

classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def _prep_variant(pil_img, force_invert=None):
    img = pil_img.convert("L")
    arr = np.array(img).astype(np.float32)
    if force_invert is None:
        if arr.mean() > 127:
            arr = 255.0 - arr
    elif force_invert:
        arr = 255.0 - arr
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) * (255.0 / (mx - mn))
    thresh = 20
    coords = np.column_stack(np.where(arr > thresh))
    if coords.size != 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        arr = arr[y0:y1, x0:x1]
    h, w = arr.shape
    target_inner = 24
    scale = target_inner / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    arr_resized = np.array(Image.fromarray(arr.astype(np.uint8)).resize((new_w, new_h), Image.BILINEAR))
    pad_top = (28 - new_h) // 2
    pad_bottom = 28 - new_h - pad_top
    pad_left = (28 - new_w) // 2
    pad_right = 28 - new_w - pad_left
    arr_padded = np.pad(arr_resized, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)
    tensor = torch.tensor(arr_padded / 255.0, dtype=torch.float32).unsqueeze(0)
    tensor = (tensor - 0.5) / 0.5
    return tensor

def predict_image(image):
    t1 = _prep_variant(image, force_invert=None).unsqueeze(0).to(device)
    t2 = _prep_variant(image, force_invert=True).unsqueeze(0).to(device)
    with torch.no_grad():
        o1 = model(t1); p1 = F.softmax(o1, dim=1)
        o2 = model(t2); p2 = F.softmax(o2, dim=1)
    if p2.max() > p1.max():
        probs = p2
        tensor_for_preview = t2.squeeze(0).squeeze(0)
    else:
        probs = p1
        tensor_for_preview = t1.squeeze(0).squeeze(0)
    top_probs, top_classes = torch.topk(probs, 3)
    preds = [(classes[i], float(top_probs[0][j])) for j, i in enumerate(top_classes[0])]
    return preds, probs.cpu().numpy(), tensor_for_preview

def show_predictions_chart(predictions):
    df = pd.DataFrame(predictions, columns=["Class", "Confidence"])
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("Confidence", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("Class", sort="-x"),
        color="Class"
    )
    st.altair_chart(chart, use_container_width=True)

def show_tensor_image(tensor):
    fig, ax = plt.subplots()
    img = tensor.cpu().numpy()
    ax.imshow(img, cmap="gray")
    ax.axis("off")
    st.pyplot(fig)

st.title("Fashion-MNIST Classifier")
st.write("Upload an image of clothing (any size, JPG/PNG) to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    if st.button("Predict"):
        predictions, raw_probs, tensor_image = predict_image(image)
        st.subheader("Top Predictions")
        for label, prob in predictions:
            st.write(f"**{label}**: {prob:.2%}")