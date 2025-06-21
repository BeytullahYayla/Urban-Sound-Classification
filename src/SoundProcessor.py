import librosa
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import re
from dotenv import load_dotenv
import os
import tqdm
load_dotenv()

class SoundProcessor:
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.audio = None
        self.sr = None

    def load_audio(self):
        self.audio, self.sr = librosa.load(self.audio_path, sr=None)
        return self.audio, self.sr
    
    def visualize_audio(self, visualization_type='both'):
        if self.audio is None or self.sr is None:
            self.load_audio()
            
        if visualization_type in ['wave', 'both']:
            plt.figure(figsize=(12, 4))
            librosa.display.waveshow(self.audio, sr=self.sr)
            plt.title(f"Waveform (Shape: {self.audio.shape})")
            plt.show()
        
        if visualization_type in ['mel', 'both']:
            S = librosa.feature.melspectrogram(y=self.audio, sr=self.sr, n_mels=128)
            S_db = librosa.power_to_db(S, ref=np.max)
            plt.figure(figsize=(12, 4))
            img = librosa.display.specshow(S_db, sr=self.sr, x_axis='time', y_axis='mel', fmax=8000)
            plt.colorbar(img, format='%+2.0f dB')
            plt.title(f"Mel Spectrogram (Shape: {S_db.shape})")
            plt.show()
        
        return self.audio, self.sr
    
    def get_audio_duration(self):
        return librosa.get_duration(y=self.audio, sr=self.sr)
    
    def get_audio_sampling_rate(self):
        return self.sr
    
    def audio_as_mel_image(self,n_mels=128):
        audio, sr = librosa.load(self.audio_path, sr=None)
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(12, 4))
        plt.imshow(S_db)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig('temp.png',bbox_inches='tight',pad_inches=0)
        plt.close()
        img = cv2.imread('temp.png')
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        os.remove('temp.png')
        return img
    
    def audio_as_mfcc(self):
        mfccs = librosa.feature.mfcc(y=self.audio, sr=self.sr, n_mfcc=13)
        return mfccs
    
if __name__ == "__main__":
    audio_path = os.getenv('DATA_PATH')
    audio_path = os.path.join(audio_path, 'fold1', '46669-4-0-45.wav')
    sound_processor = SoundProcessor(audio_path)
    sound_processor.visualize_audio()
    print(sound_processor.get_audio_duration())
    print(sound_processor.get_audio_sampling_rate())
    mel_image = sound_processor.audio_as_mel_image()
    plt.imshow(mel_image)
    plt.show()
  