import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GrapeDiseaseClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def test_model(model_path='best_model.pth', test_path='./grape_dataset1/test'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test directory not found: {test_path}")

    temp_dataset = datasets.ImageFolder(test_path, transform=transforms.Resize((256, 256)))
    num_classes = len(temp_dataset.classes)
    print(f"Found classes: {num_classes} ({temp_dataset.classes})")

    model = GrapeDiseaseClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(test_path, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []
    class_counters = {i: 0 for i in range(num_classes)}
    sample_images = []
    sample_preds = []
    sample_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(inputs.size(0)):
                true_class = labels[i].item()
                if class_counters[true_class] < 3:
                    img = inputs[i].cpu().numpy().transpose((1, 2, 0))
                    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    sample_images.append(np.clip(img, 0, 1))
                    sample_preds.append(preds[i].item())
                    sample_labels.append(true_class)
                    class_counters[true_class] += 1

            if all(count >= 3 for count in class_counters.values()):
                break

    # Визуализация с адаптивной сеткой
    rows = num_classes
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    for class_idx in range(num_classes):
        class_indices = [idx for idx, lbl in enumerate(sample_labels) if lbl == class_idx]
        for col in range(cols):
            if col >= len(class_indices):
                break
            idx = class_indices[col]
            ax = axes[class_idx, col] if num_classes > 1 else axes[col]
            ax.imshow(sample_images[idx])
            true_label = test_dataset.classes[class_idx]
            pred_label = test_dataset.classes[sample_preds[idx]]
            color = 'green' if true_label == pred_label else 'red'
            ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
            ax.axis('off')

    # Скрыть пустые subplots
    for class_idx in range(num_classes):
        class_indices = [idx for idx, lbl in enumerate(sample_labels) if lbl == class_idx]
        for col in range(len(class_indices), cols):
            if num_classes > 1:
                axes[class_idx, col].axis('off')
            else:
                axes[col].axis('off')

    plt.tight_layout()
    plt.show()

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes, digits=4))

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()

    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
    print(f"\nTest Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    test_model()