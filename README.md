# 🐱🐶 Cat Dog CNN Recognition

Ứng dụng AI nhận diện chó và mèo bằng Convolutional Neural Network (CNN) sử dụng TensorFlow và Flask.

## 🚀 Demo Features

- Upload ảnh chó hoặc mèo
- AI tự động phân loại
- CNN Deep Learning
- Flask Web Interface
- TensorFlow AI Model

---

# 🛠️ Công nghệ sử dụng

- Python 3.10
- TensorFlow
- Keras
- Flask
- HTML/CSS
- NumPy
- Pillow

---

# 📂 Cấu trúc project

```bash
cat-dog-cnn/
│
├── app.py
├── train.py
├── cat_dog_model.h5
├── requirements.txt
│
├── dataset/
│   └── train/
│       ├── cat/
│       └── dogs/
│
├── static/
│   ├── style.css
│   └── uploads/
│
├── templates/
│   └── index.html
│
└── README.md
```

---

# ⚙️ Cài đặt project

## 1. Clone repository

```bash
git clone https://github.com/thiendeptrai29/cat-dog-cnn.git
```

## 2. Di chuyển vào project

```bash
cd cat-dog-cnn
```

---

# 🐍 Tạo môi trường ảo

## Windows

```bash
python -m venv venv
```

## Kích hoạt venv

```bash
venv\Scripts\activate
```

---

# 📦 Cài thư viện

```bash
pip install -r requirements.txt
```

---

# 🧠 Train AI Model

## Chuẩn bị dataset

Tạo cấu trúc:

```bash
dataset/
   train/
      cat/
      dogs/
```

- Folder `cat` chứa ảnh mèo
- Folder `dogs` chứa ảnh chó

---

## Chạy train CNN

```bash
python train.py
```

Sau khi train xong sẽ tạo file:

```bash
cat_dog_model.h5
```

---

# 🌐 Chạy website Flask

```bash
python app.py
```

Sau đó mở trình duyệt:

```bash
http://127.0.0.1:5000
```

---

# 📸 Cách sử dụng

1. Upload ảnh chó hoặc mèo
2. Nhấn nút Predict
3. AI sẽ dự đoán kết quả:
   - 🐱 Cat
   - 🐶 Dog

---

# 🧠 CNN Model

Mô hình sử dụng:

- Convolutional Layer
- MaxPooling Layer
- Flatten Layer
- Dense Layer

CNN được train trên dataset Cats vs Dogs để học đặc trưng:

- tai
- mắt
- texture lông
- khuôn mặt
- hình dạng cơ thể

---

# 📊 Accuracy

Accuracy sau khi train đạt khoảng:

```bash
80% - 95%
```

(phụ thuộc dataset và số epoch)

---

# 📌 Dataset

Dataset sử dụng:

Cats vs Dogs Dataset

https://www.kaggle.com/datasets/tongpython/cat-and-dog

---

# 👨‍💻 Author

Thiên đẹp trai 😎

GitHub:
https://github.com/thiendeptrai29

---

# ⭐ Future Improvements

- Dark mode UI
- Webcam realtime prediction
- MobileNetV2 / Transfer Learning
- Deploy online
- React frontend
- Accuracy chart

---

# 📜 License

This project is for educational purposes.