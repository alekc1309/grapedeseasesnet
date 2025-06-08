import os
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms, datasets
from collections import Counter

# Включим отладочный режим для CUDA
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Устройство для обучения
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Функция трансформаций с учетом специфики листьев винограда
def get_transforms(augment=True):
    base = [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    if augment:
        base = [
            transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    return transforms.Compose(base)

# 2. Усовершенствованная загрузка данных
def load_data():
    def is_valid_file(path):
        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        return path.lower().endswith(valid_ext) and not os.path.basename(path).startswith('.')

    dataset_path = "./grape_dataset1"
    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "test")

    # Проверка и создание понятных сообщений об ошибках
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Директория тренировочных данных не найдена: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Директория тестовых данных не найдена: {val_path}")

    # Загрузка данных с обработкой русских названий классов
    train_dataset = datasets.ImageFolder(
        train_path,
        transform=get_transforms(augment=True),
        is_valid_file=is_valid_file
    )

    val_dataset = datasets.ImageFolder(
        val_path,
        transform=get_transforms(augment=False),
        is_valid_file=is_valid_file
    )

    # Проверка структуры данных
    print("\nОбнаруженные классы:")
    print(f"Тренировочные: {train_dataset.classes}")
    print(f"Тестовые: {val_dataset.classes}")

    if len(train_dataset.classes) != 4:
        raise ValueError(f"Ожидается 4 класса, но найдено {len(train_dataset.classes)}")

    # Вывод статистики
    print("\nСтатистика датасета:")
    print(f"Всего тренировочных образцов: {len(train_dataset)}")
    print(f"Всего тестовых образцов: {len(val_dataset)}")
    print("Распределение по классам (train):", Counter([label for _, label in train_dataset]))

    return train_dataset, val_dataset

# 3. Оптимизированная модель для классификации заболеваний листьев
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

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# 4. Инициализация обучения с проверкой
def initialize_training(train_dataset):
    model = GrapeDiseaseClassifier(num_classes=len(train_dataset.classes))
    model = model.to(device)

    # Проверка работы модели
    test_input = torch.randn(2, 3, 256, 256).to(device)
    try:
        model(test_input)
        print("\nМодель успешно инициализирована на устройстве:", device)
    except Exception as e:
        print("\nОшибка инициализации модели:", e)
        raise

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    return model, optimizer, scheduler

# 5. Главный цикл обучения с улучшенным логированием
def main():
    try:
        train_dataset, val_dataset = load_data()

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        model, optimizer, scheduler = initialize_training(train_dataset)
        criterion = nn.CrossEntropyLoss()

        print("\nНачало обучения...")
        best_val_acc = 0.0

        for epoch in range(30):
            # Training phase
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item()
                    val_correct += outputs.argmax(1).eq(labels).sum().item()

            # Calculate metrics
            train_loss /= len(train_loader)
            train_acc = 100. * correct / total
            val_loss /= len(val_loader)
            val_acc = 100. * val_correct / len(val_dataset)

            # Update scheduler
            scheduler.step()

            # Print statistics
            print(f"Эпоха [{epoch+1}/30]: "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"Новая лучшая модель сохранена с точностью {val_acc:.2f}%")

        print(f"\nОбучение завершено! Лучшая точность: {best_val_acc:.2f}%")

    except Exception as e:
        print("\nПроизошла ошибка:", e)
        print("Проверьте следующее:")
        print("1. Структуру папок (должны быть train/test с подпапками классов)")
        print("2. Наличие изображений в форматах .jpg/.png в подпапках")
        print("3. Совпадение названий классов в train и test")
        print("4. Отсутствие скрытых файлов в папках с данными")

if __name__ == "__main__":
    main()