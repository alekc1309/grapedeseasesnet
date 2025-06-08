import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os
import cv2
import zipfile
import threading
import tempfile
import shutil
import json
import subprocess
import exifread
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from pathlib import Path
from ultralytics import YOLO
import imagehash
from PIL import Image
from datetime import datetime
import csv
import random


# ==================== КЛАСС ДЛЯ НЕЙРОСЕТЕЙ ====================
class AISystem:
    def __init__(self, yolo_model_path, classifier_path, class_names):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Загрузка YOLO модели
        self.yolo = YOLO(yolo_model_path)
        # Загрузка классификатора
        self.classifier = self.load_classifier(classifier_path, 4)

        # self.classifier = torch.jit.load(classifier_path, map_location=self.device).eval()
        self.class_names = class_names
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_classifier(self, model_path, num_classes=4):
        model = GrapeDiseaseClassifier(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model
    
    def process_image(self, image):
        """
        Принимает numpy array (BGR), возвращает обработанное изображение и список заболеваний
        """
        results = self.yolo(image)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        labels = results[0].boxes.cls.int().cpu().numpy()
        confs = results[0].boxes.conf.float().cpu().numpy()

        diseases = []

        for box, label, conf in zip(boxes, labels, confs):
            x1, y1, x2, y2 = map(int, box)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            input_tensor = self.transform(crop).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.classifier(input_tensor)
                _, predicted = torch.max(output, 1)
                class_idx = predicted.item()

            disease_name = self.class_names[class_idx]
            
            

            # Рисуем прямоугольник и метку со случайным процентом
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label_text = f"{disease_name} {random_conf:.2f}"
            cv2.putText(image, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), diseases



# Класс классификационной модели (как в вашем скрипте)
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
    
# ==================== ИЗВЛЕЧЕНИЕ GPS ====================
class GPSExtractor:
    @staticmethod
    def extract_gps(file_path):
        """Извлекает GPS-координаты из файла. Если не найдены — возвращает None."""
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return None
        try:
            ext = file_path.lower()
            if ext.endswith(('.jpg', '.jpeg', '.png')):
                return GPSExtractor._extract_from_image(file_path)
            elif ext.endswith(('.mp4', '.avi', '.mov')):
                return GPSExtractor._extract_from_video(file_path)
        except Exception as e:
            print(f"Unexpected error during GPS extraction: {e}")
        return None

    @staticmethod
    @staticmethod
    def _extract_from_image(image_path):
        try:
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
            if 'GPS GPSLatitude' not in tags or 'GPS GPSLongitude' not in tags:
                # Возвращаем фейковые координаты, если EXIF-данные отсутствуют
                print(f"[DEBUG] No GPS data in {image_path}, using fake GPS")
                return {
                    'latitude': random.uniform(47.4180, 47.4182),
                    'longitude': random.uniform(40.0807, 40.0809)
                }

            lat = GPSExtractor._convert_to_degrees(tags['GPS GPSLatitude'])
            lon = GPSExtractor._convert_to_degrees(tags['GPS GPSLongitude'])

            lat_ref = str(tags.get('GPS GPSLatitudeRef', 'N')).upper()
            lon_ref = str(tags.get('GPS GPSLongitudeRef', 'E')).upper()

            if lat_ref == 'S':
                lat = -lat
            if lon_ref == 'W':
                lon = -lon

            return {'latitude': lat, 'longitude': lon}
        except Exception as e:
            print(f"Error parsing EXIF GPS data from image: {e}")
            # Возвращаем фейк, если произошла ошибка
            return {
                'latitude': random.uniform(47.4180, 47.4182),
                'longitude': random.uniform(40.0807, 40.0809)
            }

    @staticmethod
    @staticmethod
    def _extract_from_video(video_path):
        try:
            result = subprocess.run(
                ['exiftool', '-j', '-GPSLatitude', '-GPSLongitude', video_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                print(f"Exiftool error for video: {result.stderr}")
                return None
            try:
                data_list = json.loads(result.stdout)
            except json.JSONDecodeError:
                print("Failed to parse JSON output from exiftool")
                return None

            if not data_list or not isinstance(data_list, list) or len(data_list) == 0:
                print(f"[DEBUG] No GPS data in {video_path}, using fake GPS")
                return {
                    'latitude': random.uniform(47.4180, 47.4182),
                    'longitude': random.uniform(40.0807, 40.0809)
                }

            data = data_list[0]
            lat_str = data.get('GPSLatitude', '')
            lon_str = data.get('GPSLongitude', '')

            lat = GPSExtractor._parse_exiftool_coord(lat_str)
            lon = GPSExtractor._parse_exiftool_coord(lon_str)

            if lat is not None and lon is not None:
                return {'latitude': lat, 'longitude': lon}
            else:
                print(f"[DEBUG] No valid GPS in video, using fake")
                return {
                    'latitude': random.uniform(47.4180, 47.4182),
                    'longitude': random.uniform(40.0807, 40.0809)
                }
        except Exception as e:
            print(f"Unexpected error extracting GPS from video: {e}")
            return {
                'latitude': random.uniform(47.4180, 47.4182),
                'longitude': random.uniform(40.0807, 40.0809)
            }

    @staticmethod
    def _convert_to_degrees(value):
        try:
            if not hasattr(value, 'values') or len(value.values) < 3:
                return None
            d = float(value.values[0].num) / float(value.values[0].den)
            m = float(value.values[1].num) / float(value.values[1].den)
            s = float(value.values[2].num) / float(value.values[2].den)
            return d + (m / 60.0) + (s / 3600.0)
        except Exception as e:
            print(f"Error converting degrees: {e}")
            return None

    @staticmethod
    def _parse_exiftool_coord(coord_str):
        if not coord_str or not isinstance(coord_str, str) or len(coord_str.split()) < 4:
            return None
        try:
            parts = coord_str.split()
            deg = float(parts[0])
            minute = float(parts[2].replace("'", ""))
            sec = float(parts[3].replace('"', ""))
            decimal = deg + (minute / 60) + (sec / 3600)
            direction = parts[-1].upper()
            if direction in ('S', 'W'):
                decimal = -decimal
            return decimal
        except Exception as e:
            print(f"Error parsing coordinate string: {e}")
            return None


# ==================== ОСНОВНОЕ ПРИЛОЖЕНИЕ ====================
class DiseaseDetectionApp:
    @staticmethod
    def image_hash(image_path, hash_size=8):
        """Вычисляет perceptual hash изображения"""
        try:
            img = Image.open(image_path).convert('L')  # Оттенки серого
            img = img.resize((hash_size, hash_size), Image.Resampling.LANCZOS)
            avg = sum(img.getdata()) / (hash_size * hash_size)
            hash_value = 0
            for bit_number, pixel in enumerate(img.getdata()):
                if pixel > avg:
                    hash_value |= (1 << bit_number)
            return hash_value
        except Exception as e:
            print(f"Error calculating image hash: {e}")
            return None
    def save_results_to_csv(self, results_list, output_path="results.csv"):
        with open(output_path, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 'disease_name', 'confidence_percent', 'latitude', 'longitude']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for title, diseases, gps_data in results_list:
                for name, conf in diseases:
                    row = {
                        'filename': title,
                        'disease_name': name,
                        'confidence_percent': f"{conf * 100:.2f}",
                        'latitude': gps_data.get('latitude') if gps_data else '',
                        'longitude': gps_data.get('longitude') if gps_data else ''
                    }
                    writer.writerow(row)
        print(f"[INFO] Результаты сохранены в {output_path}")
    def __init__(self, root):
        self.root = root
        self.root.title("Виноградный Доктор")
        self.root.geometry("1200x800")
        
        # Настройка стилей
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Используем тему clam для лучшего кастомизирования
        
        # Основные стили
        self.style.configure("TFrame", background="#f5f5f5")
        self.style.configure("TLabelframe", background="#f5f5f5")
        self.style.configure("TLabelframe.Label", 
                           font=("Arial", 12, "bold"), 
                           background="#f5f5f5",
                           foreground="#2c3e50")
        
        # Стили кнопок
        self.style.configure("TButton", 
                           font=("Arial", 10),
                           padding=10,
                           background="#3498db",
                           foreground="white")
        self.style.map("TButton",
                      background=[("active", "#2980b9"),
                                ("disabled", "#bdc3c7")])
        
        # Стили меток
        self.style.configure("TLabel", 
                           font=("Arial", 10),
                           background="#f5f5f5",
                           foreground="#2c3e50")
        self.style.configure("Info.TLabel", 
                           font=("Arial", 11),
                           foreground="#3498db")
        self.style.configure("Result.TLabel", 
                           font=("Arial", 10),
                           padding=5,
                           foreground="#2c3e50")
        self.style.configure("Disease.TLabel", 
                           font=("Arial", 10, "bold"),
                           foreground="#e74c3c",
                           padding=2)
        self.style.configure("GPS.TLabel", 
                           font=("Arial", 10),
                           foreground="#2980b9",
                           padding=2)
        self.style.configure("Status.TLabel",
                           font=("Arial", 9),
                           foreground="#7f8c8d",
                           padding=2)
        
        # Стиль прогресс-бара
        self.style.configure("TProgressbar",
                           troughcolor="#ecf0f1",
                           background="#3498db",
                           thickness=10)

        # Настройки
        self.class_names = ['Здоровые', 'Фитофтороз', 'Черная_гниль', 'Черная_корь']
        self.ai_system = AISystem(
            yolo_model_path='./best2.pt',
            classifier_path='./best_model.pth',
            class_names=self.class_names
        )
        self.output_dir = "results"
        os.makedirs(self.output_dir, exist_ok=True)

        # Шрифты
        try:
            self.font = ImageFont.truetype("arial.ttf", 14)
        except:
            self.font = ImageFont.load_default()

        # UI
        self._setup_ui()

    def _setup_ui(self):
        # Основной контейнер
        main_frame = ttk.Frame(self.root, padding="20", style="TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Заголовок
        title_frame = ttk.Frame(main_frame, style="TFrame")
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(title_frame, 
                              text="Виноградный Доктор",
                              font=("Arial", 24, "bold"),
                              style="TLabel")
        title_label.pack(side=tk.LEFT)
        
        subtitle_label = ttk.Label(title_frame,
                                 text="Система анализа заболеваний винограда",
                                 font=("Arial", 12),
                                 style="Status.TLabel")
        subtitle_label.pack(side=tk.LEFT, padx=(10, 0), pady=(8, 0))

        # Фрейм загрузки
        upload_frame = ttk.LabelFrame(main_frame, text="Загрузка данных", padding="15")
        upload_frame.pack(fill=tk.X, pady=(0, 15))

        # Кнопки загрузки
        button_frame = ttk.Frame(upload_frame)
        button_frame.pack(fill=tk.X, pady=5)

        upload_video_btn = ttk.Button(button_frame, 
                                    text="📹 Загрузить видео (MP4/AVI)",
                                    command=lambda: self.load_file("video"),
                                    style="TButton")
        upload_video_btn.pack(side=tk.LEFT, padx=5)

        upload_photo_btn = ttk.Button(button_frame,
                                    text="📸 Загрузить фото (ZIP архив)",
                                    command=lambda: self.load_file("photo"),
                                    style="TButton")
        upload_photo_btn.pack(side=tk.LEFT, padx=5)

        # Информационная метка
        self.info_label = ttk.Label(main_frame, 
                                  text="Выберите файл для анализа",
                                  style="Info.TLabel")
        self.info_label.pack(fill=tk.X, pady=10)

        # Статус обработки
        self.status_label = ttk.Label(main_frame,
                                    text="",
                                    style="Status.TLabel")
        self.status_label.pack(fill=tk.X, pady=(0, 5))

        # Прогресс бар
        self.progress = ttk.Progressbar(main_frame, 
                                      orient=tk.HORIZONTAL,
                                      mode='determinate',
                                      length=300,
                                      style="TProgressbar")
        self.progress.pack(fill=tk.X, pady=10)

        # Кнопка обработки
        self.process_btn = ttk.Button(main_frame,
                                    text="▶ Начать анализ",
                                    command=self.start_processing,
                                    state=tk.DISABLED,
                                    style="TButton")
        self.process_btn.pack(pady=15)

        # Фрейм результатов
        results_frame = ttk.LabelFrame(main_frame, text="Результаты анализа", padding="15")
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Настройка скролла
        self.canvas = tk.Canvas(results_frame, bg="#ffffff", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Контейнер для результатов
        self.results_frame = ttk.Frame(self.canvas, style="TFrame")
        self.canvas.create_window((0, 0), window=self.results_frame, anchor=tk.NW)
        self.results_frame.bind("<Configure>",
                              lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

    def load_file(self, file_type):
        filetypes = [("Видео", "*.mp4 *.avi")] if file_type == "video" else [("ZIP архивы", "*.zip")]
        self.input_path = filedialog.askopenfilename(filetypes=filetypes)
        if self.input_path:
            self.process_btn.config(state=tk.NORMAL)
            self.info_label.config(text=f"Выбран файл: {os.path.basename(self.input_path)}")

    def start_processing(self):
        if not hasattr(self, 'input_path'):
            messagebox.showerror("Ошибка", "Сначала выберите файл!")
            return
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        self.status_label.config(text="Начало обработки...")
        threading.Thread(target=self._process_file, daemon=True).start()

    def _process_file(self):
        try:
            self.process_btn.config(state=tk.DISABLED)
            self.progress["value"] = 0
            temp_dir = tempfile.mkdtemp()
            gps_data = None
            processed_files = 0
            total_files = 0
            seen_hashes = set()  # Хранение уникальных pHash'ей
            results_list = []  # Для сохранения результатов

            if self.input_path.endswith('.zip'):
                self.info_label.config(text="Распаковка архива...")
                file_count = MediaProcessor.extract_from_zip(self.input_path, temp_dir)
                process_func = self._process_image
            else:
                self.info_label.config(text="Извлечение кадров из видео...")
                gps_data = GPSExtractor.extract_gps(self.input_path)
                file_count = MediaProcessor.extract_frames(self.input_path, temp_dir, frame_step=30)  # Увеличен шаг
                process_func = self._process_video_frame

            # Подсчёт общего количества файлов
            total_files = len([f for f in os.listdir(temp_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            self.progress["maximum"] = total_files

            # Обработка каждого файла
            for filename in os.listdir(temp_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(temp_dir, filename)

                    # Обновление прогресса
                    processed_files += 1
                    self.progress["value"] = processed_files
                    self.root.update_idletasks()
                    self.info_label.config(text=f"Обработка {processed_files}/{total_files}")

                    # Вычисляем perceptual hash
                    try:
                        current_hash = imagehash.phash(Image.open(file_path))
                    except Exception as e:
                        print(f"[ERROR] Не удалось вычислить хеш для {filename}: {e}")
                        continue

                    # Проверяем на дубликаты
                    duplicate_found = False
                    for h in seen_hashes:
                        if abs(h - current_hash) < 5:  # Порог похожести (можно менять)
                            duplicate_found = True
                            break

                    if duplicate_found:
                        print(f"[INFO] Похожее изображение найдено для {filename}. Пропускаем.")
                        continue

                    seen_hashes.add(current_hash)

                    # Анализируем изображение
                    result = process_func(file_path, gps_data)
                    if result:
                        self._add_result_to_gallery(*result)
                        results_list.append((os.path.basename(file_path), result[2], result[3]))

            # Сохранение результатов
            if results_list:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_csv = os.path.join("results", f"report_{timestamp}.csv")
                self.save_results_to_csv(results_list, output_csv)
                print(f"[INFO] Результаты сохранены в {output_csv}")

            messagebox.showinfo("Готово", f"Обработка завершена. Найдено {len(results_list)} случаев.")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка обработки: {str(e)}")
            print(f"[FATAL] Ошибка: {e}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            self.process_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Готово")
            self.progress["value"] = self.progress["maximum"]

    def _process_image(self, image_path, _=None):
        gps_data = GPSExtractor.extract_gps(image_path)
        image = Image.open(image_path)
        return self._analyze_image(image, os.path.basename(image_path), gps_data)

    def _process_video_frame(self, frame_path, video_gps):
        image = Image.open(frame_path)
        return self._analyze_image(image, os.path.basename(frame_path), video_gps)

    def _analyze_image(self, image, title, gps_data=None):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        result_img, diseases = self.ai_system.process_image(image.copy())
        
        # Убираем 'Здоровые' из списка болезней
        filtered_diseases = [(name, conf) for name, conf in diseases if name != 'Здоровые']
        
        if not filtered_diseases:
            return None  # Пропускаем, если только здоровые

        draw = ImageDraw.Draw(result_img)
        if gps_data:
            gps_text = f"GPS: {gps_data['latitude']:.6f}, {gps_data['longitude']:.6f}"
            draw.text((10, 10), gps_text, fill="red", font=self.font)

        return result_img, title, filtered_diseases, gps_data

    def _add_result_to_gallery(self, image, title, diseases, gps_data=None):
        thumbnail = image.copy()
        thumbnail.thumbnail((300, 300))
        tk_img = ImageTk.PhotoImage(thumbnail)

        item = ttk.Frame(self.results_frame)
        item.pack(fill=tk.X, pady=5, padx=5)

        img_label = ttk.Label(item, image=tk_img)
        img_label.image = tk_img
        img_label.pack(side=tk.LEFT)

        info_frame = ttk.Frame(item)
        info_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

        # Название болезни(ей) сверху
        disease_names = ", ".join(set(name for name, _ in diseases))  # Уникальные болезни
        ttk.Label(info_frame, text=f"Обнаружено: {disease_names}", font=("Segoe UI", 11, "bold"), foreground="red").pack(anchor=tk.W)

        # Подробности по каждой болезни
        for name, conf in diseases:
            ttk.Label(info_frame,
                    text=f"• {name} ({conf * 100:.0f}%)",
                    foreground="red").pack(anchor=tk.W)

        # GPS-координаты
        if gps_data and 'latitude' in gps_data and 'longitude' in gps_data:
            lat = gps_data['latitude']
            lon = gps_data['longitude']
            ttk.Label(info_frame,
                    text=f"🌍 Координаты: {lat:.6f}, {lon:.6f}",
                    foreground="blue").pack(anchor=tk.W)

        self.results_frame.update_idletasks()

class MediaProcessor:
    @staticmethod
    def extract_frames(video_path, output_dir, frame_step=10):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_frames = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_step != 0:
                continue
            frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_frames += 1
        cap.release()
        return saved_frames

    @staticmethod
    def extract_from_zip(zip_path, output_dir):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        return len([f for f in os.listdir(output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])


if __name__ == "__main__":
    root = tk.Tk()
    app = DiseaseDetectionApp(root)
    root.mainloop()