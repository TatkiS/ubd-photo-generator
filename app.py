import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

# Налаштування сторінки сайту
st.set_page_config(page_title="Генератор фото для УБД", layout="centered")

st.title("📸 Автоматична підготовка фото на УБД (3х4)")
st.write("Завантажте оригінальне фото. Сайт автоматично прибере фон, зробить його білим та відкадрує за стандартом.")

# Кнопка для завантаження файлу на сайті
uploaded_file = st.file_uploader("Оберіть фотографію (JPG або PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Конвертуємо завантажений файл у формат для OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    h_img, w_img, _ = image.shape

    # Ініціалізація ШІ MediaPipe
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    mp_face_detection = mp.solutions.face_detection

    with st.spinner('Обробка фотографії... зачекайте кілька секунд'):
        with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_seg, \
             mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_det:

            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 1. Видалення фону
            seg_results = selfie_seg.process(rgb_img)
            mask = seg_results.segmentation_mask > 0.5
            condition = np.stack((mask,) * 3, axis=-1)
            white_background = np.ones(image.shape, dtype=np.uint8) * 255
            img_white_bg = np.where(condition, image, white_background)

            # 2. Пошук обличчя та кадрування
            det_results = face_det.process(rgb_img)
            
            if not det_results.detections:
                st.error("❌ Помилка: Обличчя не знайдено. Будь ласка, завантажте інше фото, де добре видно обличчя в анфас.")
            else:
                bbox = det_results.detections[0].location_data.relative_bounding_box
                x = int(bbox.xmin * w_img)
                y = int(bbox.ymin * h_img)
                w = int(bbox.width * w_img)
                h = int(bbox.height * h_img)

                # Розрахунок пропорцій 3:4 з правильними відступами
                face_height = int(h * 1.35)
                face_center_x = x + w // 2
                face_top_y = y - int(h * 0.2)

                target_h = int(face_height / 0.75)
                target_w = int(target_h * 3 / 4)

                crop_y1 = face_top_y - int(target_h * 0.10)
                crop_y2 = crop_y1 + target_h
                crop_x1 = face_center_x - target_w // 2
                crop_x2 = crop_x1 + target_w

                # Додавання білих полів, якщо рамка виходить за межі фото
                pad_top = max(0, -crop_y1)
                pad_bottom = max(0, crop_y2 - h_img)
                pad_left = max(0, -crop_x1)
                pad_right = max(0, crop_x2 - w_img)

                if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                    img_padded = cv2.copyMakeBorder(img_white_bg, pad_top, pad_bottom, pad_left, pad_right, 
                                                    cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    crop_y1 += pad_top
                    crop_y2 += pad_top
                    crop_x1 += pad_left
                    crop_x2 += pad_left
                    cropped_photo = img_padded[crop_y1:crop_y2, crop_x1:crop_x2]
                else:
                    cropped_photo = img_white_bg[crop_y1:crop_y2, crop_x1:crop_x2]

                # Фінальний розмір високої якості
                final_photo = cv2.resize(cropped_photo, (600, 800), interpolation=cv2.INTER_AREA)

                # Відображення результату на сайті
                st.success("✅ Фото успішно підготовлено!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Оригінал", use_container_width=True)
                with col2:
                    st.image(cv2.cvtColor(final_photo, cv2.COLOR_BGR2RGB), caption="Результат 3х4", use_container_width=True)

                # Кнопка для скачування результату
                is_success, buffer = cv2.imencode(".jpg", final_photo)
                if is_success:
                    st.download_button(
                        label="📥 Скачати готове фото для друку",
                        data=buffer.tobytes(),
                        file_name="ubd_photo_3x4.jpg",
                        mime="image/jpeg"
                    )