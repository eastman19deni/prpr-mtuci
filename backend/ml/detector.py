from ultralytics import YOLO
import cv2
import os

# РЕАЛЬНАЯ ФУНКЦИЯ С ML
# Путь к модели: ищем папку ml/ в корне проекта
# Модель должна лежать в ml/yolov8n.pt (в корне проекта, отдельно от backend)
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Корень проекта
_MODEL_DIR = os.path.join(_BACKEND_DIR, "ml")
_MODEL_PATH = os.path.join(_MODEL_DIR, "yolov8n.pt")
model = YOLO(_MODEL_PATH)

def count_people(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    max_people = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            results = model(frame)[0]
            persons = [obj for obj in results.boxes if int(obj.cls) == 0]
            max_people = max(max_people, len(persons))
        except Exception:
            raise RuntimeError("YOLO inference failed")

    cap.release()
    return max_people

# ВРЕМЕННЫЙ МОК ДЛЯ ТЕСТИРОВАНИЯ БЕЗ ML (закомментирован)
# Раскомментируй код ниже и закомментируй реальную функцию для теста
# def count_people(video_path: str) -> int:
#     """Мок-версия: возвращает случайное число людей (1-10) для тестирования без ML"""
#     # Проверяем, что файл существует (базовая валидация)
#     if not os.path.exists(video_path):
#         raise RuntimeError("Video file not found")
#     
#     # Возвращаем случайное число людей для теста
#     return random.randint(1, 10)