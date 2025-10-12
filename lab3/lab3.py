import cv2
import sys

tracker_types = {
    'CSRT': cv2.TrackerCSRT_create,
    'KCF': cv2.TrackerKCF_create,
    'MOSSE': cv2.legacy.TrackerMOSSE_create
}
selected_tracker = 'CSRT'
tracker = tracker_types[selected_tracker]()

# Відкриття відео з камери
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Не вдалося відкрити відео джерело.")
    sys.exit()

# Читання першого кадру
ok, frame = video.read()
if not ok:
    print("Не вдалося прочитати відео.")
    sys.exit()

# Вибір об'єкта для відстеження
bbox = cv2.selectROI("Select ROI", frame, False)
if not bbox[2] or not bbox[3]: sys.exit()

# Ініціалізація трекера вибраним об'єктом
ok = tracker.init(frame, bbox)
cv2.destroyWindow("Select ROI")

while True:
    ok, frame = video.read()
    if not ok: break

    timer = cv2.getTickCount() # Старт таймера для вимірювання FPS
    ok, bbox = tracker.update(frame) # Оновлення трекера
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer) # Розрахунок FPS

    if ok:
        # Об'єкт успішно відстежується
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
        status_color = (0, 255, 0)
    else:
        status_color = (0, 0, 255)

    # display result
    info_text = f"{selected_tracker} | FPS: {int(fps)}"
    cv2.putText(frame, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    cv2.imshow("Відстеження об'єкта", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()