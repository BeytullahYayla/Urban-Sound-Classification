import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
from dotenv import load_dotenv
import os
import glob
import re
from CnnModel import CustomCNN

class Predictor:
    def __init__(self, model, device, label_to_class):
        self.model = model
        self.device = device
        self.label_to_class = label_to_class

    def _predict(self, image):
        self.model.eval()
        # First read the image if it's a path
        if isinstance(image, str):
            image = cv2.imread(image)
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize to 256x256
        image = cv2.resize(image, (256, 256))
        # Convert to tensor and normalize
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)  # Change from HWC to CHW format
        image = image / 255.0
        # Add batch dimension
        image = image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs.data, 1)
            return predicted
        
    def predict(self, image):
        predicted = self._predict(image)
        return predicted.item()
    
    def predict_class(self, image):
        predicted = self._predict(image)
        return predicted.item()
    
    def predict_class_name(self, image):
        predicted = self._predict(image)
        return self.label_to_class[predicted.item()]

def find_best_model():
    model_files = glob.glob('models/model_epoch_*_loss_*.pth')
    if not model_files:
        raise FileNotFoundError("No model files found in models directory")
    
    best_loss = float('inf')
    best_model_path = None
    
    for model_file in model_files:
        loss = float(re.search(r'loss_([0-9.]+)\.pth', model_file).group(1))
        if loss < best_loss:
            best_loss = loss
            best_model_path = model_file
            
    return best_model_path
    
if __name__ == "__main__":
    load_dotenv()
    model_path = find_best_model()
    model = CustomCNN()
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    label_to_class = {0: 'Air Conditioner', 1: 'Car Horn', 2: 'Children Playing', 3: 'Dog Bark', 4: 'Drilling', 5: 'Engine Idling', 6: 'Gun Shot', 7: 'Jackhammer', 8: 'Siren', 9: 'Street Music'}
    predictor = Predictor(model, device, label_to_class)
    image_path = os.getenv('IMAGE_128_PATH')
    image_paths = glob.glob(os.path.join(image_path,'*.png'))
    example_image_path = image_paths[5000]
    example_image = cv2.imread(example_image_path)
    
    # Get actual class from filename
    actual_class = example_image_path.split('\\')[-1].split('_')[0].replace('_', ' ').title()
    
    print(f"Actual class: {actual_class}")
    print(f"Predicted class: {predictor.predict_class_name(example_image)}")
    print(f"Predicted class number: {predictor.predict_class(example_image)}")