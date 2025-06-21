import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from CnnModel import SE_CustomCNN
import torch.optim as optim
import torch.nn as nn
from Dataset import UrbanDataset
from dotenv import load_dotenv
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class ModelTrainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, device, scheduler=None, epochs=100):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.epochs = epochs
    
    def train_model(self):
        # Lists to store metrics for plotting
        train_losses = []
        test_losses = []
        train_accs = []
        test_accs = []
        
        # Early stopping parameters
        best_loss = float('inf')
        patience = 10
        counter = 0
        
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images,labels in tqdm(self.train_loader):
                images,labels = images.to(self.device),labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs,labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                _,predicted = torch.max(outputs.data,1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_loss = running_loss/len(self.train_loader)
            train_acc = correct/total
            
            # Testing
            self.model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images,labels in self.test_loader:
                    images,labels = images.to(self.device),labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs,labels)
                    test_loss += loss.item()
                    _,predicted = torch.max(outputs.data,1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            test_loss = test_loss/len(self.test_loader)
            test_acc = correct/total
            
            # Store metrics
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            # Save model if it has better test loss
            if test_loss < best_loss:
                best_loss = test_loss
                counter = 0
                # Save model with epoch number and test loss
                model_path = f'models/model_epoch_{epoch+1}_loss_{test_loss:.4f}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'test_loss': test_loss,
                    'test_acc': test_acc
                }, model_path)
                print(f'Saved better model to {model_path}')
            else:
                counter += 1
                
            if counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break
                
            self.scheduler.step(test_loss)
            
            print(f'Epoch [{epoch+1}/{self.epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
            print('--------------------')
        
        # Plot training curves
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1,2,2)
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(test_accs, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
        
        return train_losses, train_accs, test_losses, test_accs
    
    def evaluate_model(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images,labels in tqdm(self.test_loader):
                images,labels = images.to(self.device),labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs,labels)
                running_loss += loss.item()
                _,predicted = torch.max(outputs.data,1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return running_loss/len(self.test_loader),correct/total


if __name__ == "__main__":
    load_dotenv()
    
    # Check if CUDA is available and set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model and move to GPU
    model = SE_CustomCNN()
    model = model.to(device)
    
    # Initialize optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)

    # Load and prepare data
    image_128_path = os.getenv("IMAGE_128_PATH")
    image_paths_128 = [os.path.join(image_128_path,path) for path in os.listdir(image_128_path)]
    labels_128 = [path.split('\\')[-1].split('_')[0].replace('_', ' ').title() for path in image_paths_128]
    label_encoder = LabelEncoder()
    labels_128 = label_encoder.fit_transform(labels_128)
    label_to_class = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))

    # Split data and create data loaders
    train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(
        image_paths_128, labels_128, 
        stratify=labels_128, 
        test_size=0.2, 
        random_state=42
    )
    
    train_dataset = UrbanDataset(train_image_paths, train_labels)
    test_dataset = UrbanDataset(test_image_paths, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize trainer and start training
    trainer = ModelTrainer(model, train_loader, test_loader, criterion, optimizer, device, scheduler=scheduler)
    trainer.train_model()