import cv2
from ultralytics import YOLO
import tkinter as tk

# YOLOv8s modell betöltése (az ultralytics csomagból)
model = YOLO("yolov8s.pt")

# Kép beolvasása OpenCV-vel
image = cv2.imread(r'D:\NJE\mestinal\gyak08\2.jpg')
if image is None:
    print("Nem sikerült betölteni a képet.")
    exit()

# Képernyőméret lekérése tkinter-rel az ablakméretezéshez
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

# Kép méret és méretezési arány számítása
img_height, img_width = image.shape[:2]
max_width, max_height = int(screen_width * 0.8), int(screen_height * 0.8)
scale = min(max_width / img_width, max_height / img_height, 1)

if scale < 1:
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    image = cv2.resize(image, (new_width, new_height))

# YOLO detektálás
results = model(image)[0]

# Detektált objektumok kirajzolása
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls_id = int(box.cls[0])
    conf = box.conf[0]
    label = f"{model.names[cls_id]} {conf:.2f}"

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Ablak megnyitása arányos mérettel
cv2.namedWindow("YOLO Objektumdetektálás", cv2.WINDOW_NORMAL)
cv2.imshow("YOLO Objektumdetektálás", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
