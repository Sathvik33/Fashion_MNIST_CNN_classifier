# Fashion MNIST CNN Classifier 🎽👞👜

This project is a **Convolutional Neural Network (CNN)** based deep learning model built to classify images from the **Fashion MNIST dataset** into 10 categories:

- T-shirt/top 👕  
- Trouser 👖  
- Pullover 🧥  
- Dress 👗  
- Coat 🧥  
- Sandal 👡  
- Shirt 👔  
- Sneaker 👟  
- Bag 👜  
- Ankle boot 👢  

We also created a **Streamlit UI** for real-time testing of the trained model.

---

## 📂 Project Structure

FashionMNIST-CNN-Classifier/
│── Model/ # Saved model file (fashion_cnn.pth)
│── Data/ # Dataset (downloaded automatically by PyTorch)
│── train.py # Training & evaluation script
│── app.py # Streamlit UI
│── requirements.txt # Dependencies
│── README.md # Project documentation


---

## 🧠 Model Architecture

The CNN model used here:

1. **Conv2D layer (32 filters, 3x3 kernel, ReLU)**
2. **MaxPooling (2x2)**
3. **Conv2D layer (64 filters, 3x3 kernel, ReLU)**
4. **MaxPooling (2x2)**
5. **Flatten layer**
6. **Fully Connected (128 neurons, ReLU + Dropout)**
7. **Output Layer (10 classes with Softmax)**

This design helps the model extract hierarchical features from images and classify them effectively.

---

## 🔄 Preprocessing

- All Fashion MNIST images are **28x28 grayscale**.  
- Transformations applied:
  - Convert to **Tensor**
  - Normalize pixel values to **[-1, 1]**

---
Training

Dataset: Fashion MNIST

Optimizer: Adam (lr = 0.001)

Loss Function: CrossEntropyLoss

Batch Size: 64

Epochs: 10 (can be increased for higher accuracy)

Accuracy

The model achieved around 92% test accuracy after training.

---

Testing

For evaluation:

Model predictions compared against test labels.

Accuracy calculated as:
accuracy = (correct_predictions / total_samples) * 100

---

Streamlit UI

We created an interactive UI (app.py) using Streamlit:

Upload any 28x28 grayscale image or a resized image.

The model predicts one of the 10 fashion categories.

The UI displays the predicted class.

---

How It Works

train.py → Train the CNN model and save it as Model/fashion_cnn.pth

app.py → Load the saved model and test it on uploaded images.

requirements.txt → Contains required dependencies.

---

🔮 Future Improvements

Add data augmentation for more robustness.

Try transfer learning (ResNet, EfficientNet) for better accuracy.

Expand dataset with real-world images for generalization.
