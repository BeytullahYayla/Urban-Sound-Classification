# Urban Sound Classification using 2D CNN with SE Blocks

This project focuses on classifying urban environmental sounds using deep learning techniques based on 2D convolutional neural networks (CNNs) and mel-spectrogram image representations. A Squeeze-and-Excitation (SE) attention mechanism is integrated to enhance feature selection across channels.

## 🔍 Project Overview
This project aims to develop a high-accuracy classification system for 10 different types of urban sounds (e.g., dog bark, air conditioner, car horn) using the publicly available UrbanSound8K dataset. The input data is transformed into mel-spectrogram images, which are then processed by a CNN model.

---

## 🗃 Dataset
- **Name:** UrbanSound8K
- **Source:** [UrbanSound8K Dataset Website](https://urbansounddataset.weebly.com/urbansound8k.html)
- **Format:** `.wav` files
- **Number of Samples:** 8,732
- **Number of Classes:** 10

---

## 🧼 Data Preprocessing & Feature Extraction
- Audio files are processed using the **Librosa** library.
- Each sound clip is converted into a **mel-spectrogram**, representing the intensity of sound frequencies over time.
- Mel-spectrograms are converted to **RGB images**.
- Images are resized to **256x256** pixels.
- These images are used as input to the neural network model.

---

## 🧠 Model Architecture & Training
The core model is a **2D Convolutional Neural Network** with the following structure:

### 🔷 Convolutional Blocks (4x):
- 3x3 Convolution
- Batch Normalization
- ReLU Activation
- **Squeeze-and-Excitation Block**
- 2x2 Max Pooling

### 🔶 Classifier:
- Flatten
- Dense(512) + ReLU
- Dense(256) + ReLU
- Output layer: Softmax with 10 neurons (one per class)

### 🛠 Training Configuration:
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam
- **Learning Rate Scheduler:** ReduceLROnPlateau
- **Early Stopping:** Enabled
- **Epochs:** 25

---

## 📊 Results
- **Validation Accuracy:** ~94.16%
- **Validation Loss:** ~0.25
- **Evaluation Metrics:**
  - Accuracy / Loss Curves
  - Confusion Matrix
  - Grad-CAM Visualizations

---

## 🔥 Explainability with Grad-CAM
Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize the decision-making process of the CNN. It highlights important regions in the spectrogram image that influenced the classification decision.

---

## 💻 Web Application (Streamlit Interface)
An interactive **Streamlit** app allows users to:
- Upload audio or spectrogram files
- Receive class predictions
- Visualize the Grad-CAM explanation

To run:
```bash
streamlit run streamlit_app.py
```

---

## 🧰 Installation & Running on Local Machine

### 1. Clone the Repository
```bash
git clone https://github.com/BeytullahYayla/Urban-Sound-Classification.git
cd Urban-Sound-Classification
```

### 2. Set Up Environment
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python train.py
```

### 4. Evaluate the Model
```bash
python evaluate.py
```

### 5. Launch the Web App
```bash
streamlit run streamlit_app.py
```

---

## 📂 Project Structure
```
Urban-Sound-Classification/
├── dataset/                  # Contains UrbanSound8K files
├── models/                   # Trained models
├── utils/                    # Utility functions
├── train.py                  # Model training
├── evaluate.py               # Evaluation scripts
├── streamlit_app.py          # Web interface
├── requirements.txt
└── README.md
```

---

## 👤 Author
**Beytullah Yayla**  
Sakarya University, Computer Engineering  
📧 beytullah.yayla1@ogr.sakarya.edu.tr

---

## 📜 License
This project is released under the MIT License.
