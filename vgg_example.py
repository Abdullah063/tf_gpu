import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# GPU kullanımı kontrol
print("TensorFlow GPU available:", tf.config.list_physical_devices('GPU'))

# VGG16 modelini yükle
model = VGG16(weights="imagenet")

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
    resized_frame = cv2.resize(frame, (224, 224))  # 224x224 boyutunda olmalı
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)  # RGB formatına çevir
    x = np.expand_dims(rgb_frame, axis=0)  # Batch boyutunu ekle
    x = preprocess_input(x)  # VGG16 için ön işleme uygula

    # Tahmin yap
    predictions = model.predict(x)
    label = decode_predictions(predictions, top=1)[0][0]

    # Etiket bilgilerini çöz
    label_name, label_confidence = label[1], label[2]

    # Çerçeve üzerine tahmini yaz
    cv2.putText(
        frame,
        f'{label_name} ({label_confidence*100:.2f}%)',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
        cv2.LINE_AA
    )

    # Çerçeveyi göster
    cv2.imshow("VGG16 Prediction (GPU)", frame)

    # 'q' tuşu ile çıkış yap
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()