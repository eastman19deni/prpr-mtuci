from ultralytics import YOLO
import cv2
import os
import numpy as np

# РЕАЛЬНАЯ ФУНКЦИЯ С ML
# Путь к модели: ищем папку ml/ в корне проекта
# Модель должна лежать в ml/yolov8n.pt (в корне проекта, отдельно от backend)
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Корень проекта
_MODEL_DIR = os.path.join(_BACKEND_DIR, "ml")
_MODEL_PATH = os.path.join(_MODEL_DIR, "yolov8n.pt")

# Проверяем существование модели перед загрузкой
if not os.path.exists(_MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {_MODEL_PATH}")

try:
    model = YOLO(_MODEL_PATH)
    print(f"✅ Model loaded successfully from {_MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"Failed to load YOLO model: {e}")

def count_people(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    # Собираем количество людей с каждого кадра для более точного результата
    people_counts = []
    frame_count = 0
    processed_frames = 0
    # Обрабатываем каждый 3-й кадр для лучшей точности
    frame_skip = 3
    
    # Получаем размеры кадра для фильтрации
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Cannot read first frame")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Возвращаемся к началу
    
    frame_height, frame_width = first_frame.shape[:2]
    # Более мягкие фильтры по размеру для лучшего обнаружения
    min_height = 30   # Минимальная высота детекции (пиксели) - уменьшено для сидячих людей
    max_height = 600  # Максимальная высота детекции (пиксели) - увеличено для стоячих

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Пропускаем кадры для ускорения
        if frame_count % frame_skip != 0:
            continue

        processed_frames += 1
        try:
            # Пониженный confidence для лучшего обнаружения (особенно сидячих людей)
            # NMS для дедупликации перекрывающихся детекций
            results = model(frame, verbose=False, conf=0.35, iou=0.45)[0]
            
            persons = []
            for obj in results.boxes:
                # Фильтруем только людей (класс 0)
                if int(obj.cls) != 0:
                    continue
                
                # Проверяем confidence
                conf = float(obj.conf)
                if conf < 0.35:
                    continue
                
                # Получаем координаты bbox
                x1, y1, x2, y2 = map(int, obj.xyxy[0])
                height = y2 - y1
                width = x2 - x1
                
                # Фильтруем по размеру - убираем только явно неправильные детекции
                if height < min_height or height > max_height:
                    continue
                
                # Более мягкие фильтры по ширине - убираем только явно неправильные
                # Слишком широкие (вероятно, несколько людей вместе или группа)
                if width > height * 3.0:
                    continue
                
                # Слишком узкие (вероятно, столб или ложное срабатывание)
                if width < height * 0.2:
                    continue
                
                persons.append({
                    'bbox': (x1, y1, x2, y2),
                    'conf': conf,
                    'height': height
                })
            
            people_count = len(persons)
            people_counts.append(people_count)
            
            # Логирование для отладки (первые 10 кадров)
            if processed_frames <= 10:
                print(f"Frame {frame_count}: detected {people_count} people (conf>=0.35, height {min_height}-{max_height}px)")
            
        except Exception as e:
            # Логируем ошибку, но продолжаем обработку
            print(f"Error processing frame {frame_count}: {e}")
            import traceback
            traceback.print_exc()
            continue

    cap.release()
    
    if not people_counts:
        print("Warning: No frames processed successfully")
        return 0
    
    # Используем среднее значение, округленное вверх, для более точного подсчета
    # Это лучше учитывает все кадры, а не только медиану
    median_count = int(np.median(people_counts))
    mean_count = np.mean(people_counts)
    max_count = max(people_counts)
    
    # Используем среднее, округленное до ближайшего целого
    # Если среднее близко к максимуму, используем максимум (для случаев, когда люди появляются не на всех кадрах)
    final_count = int(round(mean_count))
    if max_count > mean_count * 1.3:  # Если максимум значительно больше среднего
        # Возможно, люди появляются не на всех кадрах, используем среднее между средним и максимумом
        final_count = int(round((mean_count + max_count) / 2))
    
    print(f"Processed {processed_frames} frames")
    print(f"People counts: median={median_count}, mean={mean_count:.1f}, max={max_count}, final={final_count}")
    print(f"All counts: {people_counts[:30]}..." if len(people_counts) > 30 else f"All counts: {people_counts}")
    
    return final_count

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