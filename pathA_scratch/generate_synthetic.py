"""
Путь А, шаг 1 — генерируем СИНТЕТИЧЕСКИЙ датасет.

Пока нет реальных фото, учимся на нарисованных "иглах":
на шумном фоне — тонкая, еле заметная линия в случайном месте.
Для каждой картинки сохраняем МЕТКУ = координаты центра иглы (x, y).

Так мы полностью повторяем реальную задачу ("найди тонкий объект"),
но данные получаем мгновенно и в любом количестве.

Запуск:  python pathA_scratch/generate_synthetic.py
Результат: data/synthetic/images/*.png  и  data/synthetic/labels.csv
"""
import os
import csv
import numpy as np
from PIL import Image

# --- настройки ---
N_IMAGES = 2000        # сколько картинок сгенерировать
SIZE = 64              # размер картинки в пикселях (64x64), маленький = учится быстро
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
IMG_DIR = os.path.join(OUT_DIR, "images")

# генератор случайных чисел с фиксированным seed -> результат воспроизводим
rng = np.random.default_rng(42)


def make_one():
    """Создаёт одну картинку с иглой и возвращает (массив_картинки, x_центра, y_центра)."""
    # 1) фон: серый шум (имитация текстуры стола/ковра)
    img = rng.normal(loc=128, scale=18, size=(SIZE, SIZE)).astype(np.float32)

    # 2) параметры иглы: центр, угол, длина
    cx = rng.uniform(12, SIZE - 12)          # центр по X (не у самого края)
    cy = rng.uniform(12, SIZE - 12)          # центр по Y
    angle = rng.uniform(0, np.pi)            # наклон
    length = rng.uniform(10, 20)             # длина иглы в пикселях
    brightness = rng.uniform(25, 45)         # НАСКОЛЬКО игла ярче фона (маленькое = еле заметно)

    # 3) рисуем тонкую линию, проходя вдоль иглы маленькими шагами
    dx, dy = np.cos(angle), np.sin(angle)
    for t in np.linspace(-length / 2, length / 2, num=int(length * 3)):
        px = int(round(cx + t * dx))
        py = int(round(cy + t * dy))
        if 0 <= px < SIZE and 0 <= py < SIZE:
            img[py, px] += brightness      # чуть светлее фона = наша игла

    # 4) немного размываем и добавляем шум, чтобы игла была "еле заметной"
    img += rng.normal(0, 6, size=img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, cx, cy


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    labels_path = os.path.join(OUT_DIR, "labels.csv")

    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "x", "y"])   # заголовок: имя файла и координаты иглы
        for i in range(N_IMAGES):
            img, cx, cy = make_one()
            name = f"{i:05d}.png"
            Image.fromarray(img, mode="L").save(os.path.join(IMG_DIR, name))
            # МЕТКИ нормируем в диапазон [0,1] (делим на размер) — так сети удобнее учиться
            writer.writerow([name, round(cx / SIZE, 5), round(cy / SIZE, 5)])
            if (i + 1) % 500 == 0:
                print(f"  сгенерировано {i + 1}/{N_IMAGES}")

    print(f"Готово. Картинки: {IMG_DIR}")
    print(f"Метки:   {labels_path}")


if __name__ == "__main__":
    main()
