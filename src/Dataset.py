import torch 
import cv2
from torch.utils.data import Dataset
from dotenv import load_dotenv
import os
from sklearn.preprocessing import LabelEncoder

class UrbanDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ## resize image to 256x256
        image = cv2.resize(image,(256,256))
        
        # Convert to PyTorch tensor and normalize to [0,1]
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1) # Change from HWC to CHW format
        image = image / 255.0
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)
    
    def get_image_paths(self):
        return self.image_paths
    
    def get_labels(self):
        return self.labels
    
    def get_image_path(self, idx):
        return self.image_paths[idx]
    
    def get_label(self, idx):
        return self.labels[idx]
    
    def get_class_name(self, idx):
        image_path = self.image_paths[idx]
        class_name = image_path.split('\\')[-1].split('_')[0].replace('_', ' ').title()
        return class_name

if __name__ == "__main__":
    load_dotenv()
    image_128_path = os.getenv("IMAGE_128_PATH")
    image_paths_128 = [os.path.join(image_128_path,path) for path in os.listdir(image_128_path)]
    labels_128 = [path.split('\\')[-1].split('_')[0].replace('_', ' ').title() for path in image_paths_128]

    label_encoder = LabelEncoder()
    labels_128 = label_encoder.fit_transform(labels_128)

    label_to_class = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))

    dataset = UrbanDataset(image_paths_128, labels_128)
    print(dataset[0])
    print(dataset.get_image_path(0))
    print(dataset.get_label(0))
    print(dataset.get_class_name(0))
    print(label_to_class)