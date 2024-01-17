import cv2
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np

# Захват видео с веб-камеры
cam = cv2.Videocamture(0)

# Инициализация распознавания рук с помощью Mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands()

# Инициализация инструментов рисования на изображении
mpDraw = mp.solutions.drawing_utils

# Получение информации о громкости аудиоустройства
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Получение диапазона значений громкости
volumeMin, volumeMax = volume.GetVolumeRange()[:2]

while True:
    # Чтение изображения с веб-камеры
    success, img = cam.read()

    # Преобразование изображения из формата BGR в RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Обработка изображения для распознавания рук
    results = hands.process(img)

    # Список, содержащий координаты ключевых точек рук
    l = []

    # Если обнаружены руки на изображении
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmark.landmark):
                # Получение координаты точки в пикселях
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                l.append([id, cx, cy])

            # Визуализация ключевых точек рук и связей между ними на изображении
            mpDraw.draw_landmarks(img, hand_landmark, mpHands.HAND_CONNECTIONS)

    # Если список координат ключевых точек не пуст
    if l != []:
        # Координаты указательного пальца
        x1, y1 = l[4][1], l[4][2]
        # Координаты большого пальца
        x2, y2 = l[8][1], l[8][2]

        # Отображение окружности на пальцах
        cv2.circle(img, (x1, y1), 4, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 4, (255, 0, 0), cv2.FILLED)

        # Отображение линии между пальцами
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Вычисление длины отрезка между пальцами
        length = hypot(x2 - x1, y2 - y1)

        # Интерполяция длины отрезка в диапазон значений громкости
        vol = np.interp(length, [15, 220], [volumeMin, volumeMax])

        # Установка уровня громкости аудиоустройства
        volume.SetMasterVolumeLevel(vol, None)

    # Отображение изображения с визуализацией
    cv2.imshow("Image", img)

    # Выход из цикла при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
