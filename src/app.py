import streamlit as st
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from CnnModel import CustomCNN
from Predictor import Predictor, find_best_model
from GradcamVisualizer import GradcamVisualizer
import librosa
import librosa.display
import io

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Urban Sound Classification",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize model and predictor
@st.cache_resource
def load_model():
    model_path = find_best_model()
    model = CustomCNN()
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    label_to_class = {
        0: 'Air Conditioner', 
        1: 'Car Horn', 
        2: 'Children Playing', 
        3: 'Dog Bark', 
        4: 'Drilling', 
        5: 'Engine Idling', 
        6: 'Gun Shot', 
        7: 'Jackhammer', 
        8: 'Siren', 
        9: 'Street Music'
    }
    return model, device, label_to_class

def display_prediction(predicted_class):
    st.markdown(f"""
        <div class="prediction-box">
            <h3 style="margin: 0;">Predicted Sound Class</h3>
            <h2 style="margin: 0; color: #1f77b4;">{predicted_class}</h2>
        </div>
    """, unsafe_allow_html=True)

def audio_to_spectrogram(audio_bytes, sr=22050):
    import io
    import tempfile
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)  # sr=None ile orijinal sample rate
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
        plt.figure(figsize=(12, 4))
        plt.imshow(S_db)  # default colormap (viridis)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(tmpfile.name, bbox_inches='tight', pad_inches=0)
        plt.close()
        img = cv2.imread(tmpfile.name)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    os.remove(tmpfile.name)
    img = cv2.resize(img, (256, 256))  # Modelin beklediği boyut
    return img, S_db

def main():
    # Sidebar
    with st.sidebar:
        st.title("🎵 Urban Sound Classification")
        st.markdown("---")
        st.markdown("""
        ### About
        This application uses a deep learning model to classify urban sounds from spectrogram images or audio files.
        
        ### How to use
        1. Upload a spectrogram image or audio file
        2. View the prediction
        3. Explore the GradCAM or spectrogram visualization
        """)
        st.markdown("---")
        st.markdown("Made with ❤️ using Streamlit")

    st.title("Urban Sound Classification")
    model, device, label_to_class = load_model()
    predictor = Predictor(model, device, label_to_class)
    gradcam_visualizer = GradcamVisualizer(model, "layer4.2")

    main_tab = st.tabs(["📊 Image-based Prediction", "🔊 Audio-based Prediction"])

    # --- IMAGE TAB ---
    with main_tab[0]:
        uploaded_file = st.file_uploader("Choose a spectrogram image...", type=["png", "jpg", "jpeg"], key="img_upload")
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            predicted_class = predictor.predict_class_name(image)
            predicted_class_num = predictor.predict_class(image)
            display_prediction(predicted_class)
            temp_path = "temp_image.png"
            cv2.imwrite(temp_path, image)
            plt.style.use('dark_background')
            gradcam_visualizer.visualize_gradcam(temp_path, predicted_class_num, device, label_to_class)
            st.pyplot(plt.gcf())
            plt.close()
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # --- AUDIO TAB ---
    with main_tab[1]:
        audio_file = st.file_uploader("Upload a .wav audio file...", type=["wav"], key="audio_upload")
        if audio_file is not None:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')
            img, S_db = audio_to_spectrogram(audio_bytes)
            predicted_class = predictor.predict_class_name(img)
            predicted_class_num = predictor.predict_class(img)
            display_prediction(predicted_class)
            y, sr = librosa.load(io.BytesIO(audio_bytes))
            st.write("### Waveform:")
            fig, ax = plt.subplots(figsize=(5, 1.5))
            librosa.display.waveshow(y, sr=sr, ax=ax, color='#1976d2')
            ax.set_axis_off()
            st.pyplot(fig)
            plt.close(fig)
            st.write("### GradCAM Visualization:")
            temp_path = "temp_audio_spec.png"
            cv2.imwrite(temp_path, img)
            plt.style.use('dark_background')
            gradcam_visualizer.visualize_gradcam(temp_path, predicted_class_num, device, label_to_class)
            st.pyplot(plt.gcf())
            plt.close()
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    main() 