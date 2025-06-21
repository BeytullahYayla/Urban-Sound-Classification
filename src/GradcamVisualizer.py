import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from CnnModel import CustomCNN
from Dataset import UrbanDataset
import glob
from sklearn.preprocessing import LabelEncoder

class GradcamVisualizer:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer  # örn: 'layer4.2'
        self.activations = None
        self.gradients = None
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def get_conv_layer(self):
        for name,module in self.model.named_modules():
            if name == self.target_layer:
                return module
        raise ValueError(f"Layer {self.target_layer} not found in model")
    
    # Function to generate Grad-CAM heatmap
    def compute_gradcam(self, img_tensor, class_index):
        # Hook'ları kaydet
        conv_layer = dict([*self.model.named_modules()])[self.target_layer]
        forward_handle = conv_layer.register_forward_hook(self.save_activation)
        backward_handle = conv_layer.register_backward_hook(self.save_gradient)

        # Forward
        output = self.model(img_tensor)
        pred_class = output[:, class_index]
        self.model.zero_grad()
        pred_class.backward()

        # GradCAM hesaplama
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, ...] *= pooled_gradients[i]
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)

        # Hook'ları kaldır
        forward_handle.remove()
        backward_handle.remove()
        return heatmap
    
    def overlay_heatmap(self, image_path, heatmap):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        heatmap = cv2.resize(heatmap, (256, 256))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        return superimposed_img

    def visualize_gradcam(self, image_path, label, device, label_to_class):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)
        image_tensor.requires_grad = True

        heatmap = self.compute_gradcam(image_tensor, label)
        overlayed = self.overlay_heatmap(image_path, heatmap)
        original = image

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(original)
        plt.title(f'Original Image\n Predicted Label: {label_to_class[int(label)]}')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(overlayed)
        plt.title(f'GradCAM Heatmap\nPredicted Label: {label_to_class[int(label)]}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    import os
    model = CustomCNN()
    print(model)
    model.load_state_dict(torch.load("D:/Urban Sound Classification/src/models/model_epoch_25_loss_0.2866.pth")['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    label_to_class = {0: 'Air Conditioner', 1: 'Car Horn', 2: 'Children Playing', 3: 'Dog Bark', 4: 'Drilling', 5: 'Engine Idling', 6: 'Gun Shot', 7: 'Jackhammer', 8: 'Siren', 9: 'Street Music'}
    image_path = os.getenv('IMAGE_128_PATH')
    image_paths = glob.glob(os.path.join(image_path,'*.png'))
    example_image_path = image_paths[0]
    labels_128 = [path.split('\\')[-1].split('_')[0].replace('_', ' ').title() for path in image_paths]
    label_encoder = LabelEncoder()
    labels_128 = label_encoder.fit_transform(labels_128)
    example_label = labels_128[0]
    gradcam_visualizer = GradcamVisualizer(model, "layer4.2")
    gradcam_visualizer.visualize_gradcam(example_image_path,example_label,device,label_to_class)