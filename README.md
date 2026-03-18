# 🎭 Deepfake Detector

An AI-powered deepfake detection system that analyses videos and images in real-time to identify manipulated or synthetically generated media — helping the general public fight misinformation and digital fraud.

---

## 🚀 Features

- 🎥 Detect deepfake **videos** frame by frame
- 🖼️ Detect deepfake **images** instantly
- ⚡ **Real-time detection** with live feedback
- 📊 Generate detailed **detection reports**
- 🧠 Powered by deep learning models for high accuracy
- 🌐 Clean and intuitive React-based frontend

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python / Flask | Backend web framework & API |
| OpenCV | Video & image processing |
| TensorFlow / PyTorch | Deep learning model inference |
| React | Frontend user interface |
| HTML / CSS / JS | UI styling & interactions |

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/muthareddycharankumar-ship-it/Deepfake_Detector.git
cd Deepfake_Detector
```

### 2. Create a virtual environment
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Install frontend dependencies
```bash
cd frontend
npm install
```

---

## ▶️ Usage

### 1. Start the Flask backend
```bash
python app.py
```

### 2. Start the React frontend
```bash
cd frontend
npm start
```

### 3. Open your browser
```
http://localhost:3000
```

Upload a video or image and get instant deepfake detection results with a detailed report!

---

## 📁 Project Structure
```
Deepfake_Detector/
├── app.py                  # Main Flask application
├── model_loader.py         # AI model loading & inference
├── video_processor.py      # Video & image processing logic
├── requirements.txt        # Python dependencies
├── frontend/               # React frontend
└── README.md               # Project documentation
```

---

## 🧠 How It Works

1. **Upload** a video or image through the web interface
2. **OpenCV** extracts and preprocesses frames
3. **Deep learning model** analyses each frame for manipulation artifacts
4. **Detection report** is generated with confidence scores
5. **Results** are displayed instantly on the frontend

---

## 📊 Detection Report Includes

- ✅ Real or Fake classification
- 📈 Confidence score percentage
- 🎞️ Frame-by-frame analysis (for videos)
- 📝 Summary report

---

## 📌 Requirements

- Python 3.8+
- Node.js & npm (for frontend)
- GPU recommended for faster inference
- pip & virtual environment

---

## ⚠️ Disclaimer

This tool is intended for educational and awareness purposes. Detection accuracy may vary depending on the quality and type of deepfake. Always verify with multiple sources.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Mutharedy Charan Kumar**
- GitHub: [@muthareddycharankumar-ship-it](https://github.com/muthareddycharankumar-ship-it)
