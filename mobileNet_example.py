import cv2
import numpy as np
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
import tensorflow as tf

# GPU kontrolü
print("TensorFlow GPU available:", tf.config.list_physical_devices('GPU'))

# MobileNet modelini yükle
model = MobileNet(weights="imagenet")

# Kamerayı başlat
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Görüntüyü yeniden boyutlandır ve ön işleme uygula
    resized_frame = cv2.resize(frame, (224, 224))  # MobileNet giriş boyutu
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)  # RGB formatına çevir
    x = np.expand_dims(rgb_frame, axis=0)  # Batch boyutunu ekle
    x = preprocess_input(x)  # MobileNet için ön işleme

    # Tahmin yap
    predictions = model.predict(x)
    label = decode_predictions(predictions, top=1)[0][0]

    # Etiket bilgilerini çöz
    label_name, label_confidence = label[1], label[2]

    # Dikdörtgen ve metin için koordinatlar
    text = f"{label_name} ({label_confidence*100:.2f}%)"
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x_start, y_start = 10, 30  # Metnin başlangıç pozisyonu
    x_end, y_end = x_start + text_width + 10, y_start + text_height + 10  # Dikdörtgenin boyutu

    # Çerçeve üzerine dikdörtgen çiz
    cv2.rectangle(frame, (x_start, y_start - text_height - 10), (x_end, y_end), (0, 255, 0), -1)  # Dolgu rengi
    cv2.rectangle(frame, (x_start, y_start - text_height - 10), (x_end, y_end), (0, 0, 0), 2)  # Dış çerçeve

    # Metni dikdörtgenin içine yerleştir
    cv2.putText(
        frame,
        text,
        (x_start + 5, y_start),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),  # Beyaz yazı rengi
        2,
        cv2.LINE_AA
    )

    # Çerçeveyi göster
    cv2.imshow("MobileNet Prediction", frame)

    # 'q' tuşu ile çıkış yap
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()