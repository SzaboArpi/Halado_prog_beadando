# Videó elemzési kód dokumentációja

Ez a kód egy YOLOv8 (You Only Look Once) modellt használ arra, hogy személyeket detektáljon és kövessen egy videóban, majd egy annotált kimeneti videót generáljon, amely jelöli a detektált személyeket és megszámolja az egyedi személyeket a teljes videó során.

## Előfeltételek

* **ultralytics** könyvtár a YOLOv8 modellhez.
* **opencv-python** könyvtár a videó beolvasásához és írásához.
* **yolov8s.pt** modellfájl, amelyet a program automatikusan letölt, ha nincs jelen.

## Használat

1. **Telepítse a szükséges könyvtárakat:** A kód elején található `!pip install ultralytics` parancs biztosítja az `ultralytics` telepítését.
2. **Futtassa a kódot:** A kód futtatásakor megkéri a felhasználót, hogy adjon meg egy bemeneti videófájl nevet. Például: `video1.mp4`.
    * Ha üresen hagyja és Entert nyom, az alapértelmezett `video1.mp4` fájlt használja.
3. **Bemeneti videó:** Győződjön meg róla, hogy a megadott videófájl (pl. `video1.mp4`) jelen van a Colab környezetben.

## Működés

* **YOLO modell betöltése:** Betölti a `yolov8s.pt` YOLO modellt.
* **'person' osztály azonosítása:** Meghatározza a 'person' (személy) osztály indexét a modell által detektálható objektumok közül.
* **Videó beolvasása:** Megnyitja a felhasználó által megadott videót.
* **Kimeneti videó írása:** Inicializál egy videóírót `output_video_with_detections_v2.mp4` néven, ahova az annotált képkockákat menti.

<hr style="page-break-after: always;">

* **Képkocka feldolgozás:** Végighalad a videó összes képkockáján:
  * Minden képkockán személyeket detektál és követ `persist=True` beállítással, hogy megőrizze a követési azonosítókat a képkockák között.
  * Rajzol egy zöld határolókeretet a detektált személyek köré, és hozzáadja a követési azonosítójukat (`ID`) és megbízhatósági pontszámukat (`Conf`).
  * Nyomon követi az **egyedi személyek számát** a videóban (minden detektált személyt csak egyszer számol).
  * Megjeleníti a **feldolgozott képkockák számát** és a **kihagyott képkockák számát**.
  * Különbséget tesz a valódi stream vége és a kihagyott képkockák között a `cv2.CAP_PROP_POS_FRAMES` és `cv2.CAP_PROP_FRAME_COUNT` segítségével.
* **Eredmények kiírása:** A feldolgozás befejeztével a program kiírja a feldolgozott képkockák számát, a kihagyott képkockák számát, az összes egyedi detektált személy számát és a kimeneti videó elérési útját.

## Források

* **ultralytics könyvtár és YOLOv8 modell:** [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
* **opencv-python könyvtár:** [OpenCV Documentation](https://docs.opencv.org/)
* **Google Colab AI ügynök:** [Google Colab](https://colab.research.google.com/)
