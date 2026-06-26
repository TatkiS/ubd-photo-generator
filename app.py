import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Генератор фото на УБД", layout="centered")

st.title("📸 Автоматичний генератор фото 3х4 на УБД")
st.write("Програма автоматично знайде обличчя, обріже фото під стандарт 3х4 та підготує аркуш для друку.")

uploaded_file = st.file_uploader("Виберіть та завантажте фото військового:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Читання зображення
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        st.error("Не вдалося завантажити зображення. Спробуйте інший файл.")
    else:
        h_orig, w_orig, _ = img.shape
        
        # Конвертуємо в сірий колір для розпізнавання
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Завантажуємо вбудований надійний класифікатор облич
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Шукаємо обличчя
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        
        if len(faces) == 0:
            st.warning("⚠️ Обличчя не виявлено автоматично. Переконайтеся, що на фото чітко видно обличчя, або спробуйте інше якісне фото.")
        else:
            # Беремо перше знайдене обличчя
            x, y, w, h = faces[0]
            
            # Центр обличчя
            cx = x + w // 2
            cy = y + h // 2
            
            # Для УБД (3х4) потрібні пропорції: ширина = W, висота = W * 1.333
            # Розширюємо зону навколо обличчя, щоб захопити плечі та верх голови
            crop_w = int(w * 2.2)
            crop_h = int(crop_w * 4 / 3)
            
            # Визначаємо координати обрізки зі зміщенням трохи вгору, щоб не зрізати зачіску
            start_x = cx - crop_w // 2
            start_y = cy - int(crop_h * 0.45)
            
            # Коригуємо, якщо межі виходять за рамки оригінального фото
            if start_x < 0: start_x = 0
            if start_y < 0: start_y = 0
            
            end_x = start_x + crop_w
            end_y = start_y + crop_h
            
            if end_x > w_orig: end_x = w_orig
            if end_y > h_orig: end_y = h_orig
            
            # Обрізаємо фото
            cropped = img[start_y:end_y, start_x:end_x]
            
            # Приводимо до чіткого стандарту (600х800 пікселів)
            final_photo = cv2.resize(cropped, (600, 800))
            
            # Показуємо результат користувачу
            st.image(cv2.cvtColor(final_photo, cv2.COLOR_BGR2RGB), caption="Готове фото 3х4", width=250)
            
            # Створення аркуша для друку (6 штук)
            st.subheader("🖨️ Підготовка до друку")
            
            # Створюємо біле полотно під формат друку
            padding = 40
            canvas_w = 600 * 2 + padding * 3
            canvas_h = 800 * 3 + padding * 4
            
            canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
            
            # Координати для 6 фото (2 стовпчики, 3 рядки)
            positions = [
                (padding, padding), (padding, 600 + padding * 2),
                (800 + padding * 2, padding), (800 + padding * 2, 600 + padding * 2),
                (1600 + padding * 3, padding), (1600 + padding * 3, 600 + padding * 2)
            ]
            
            for pos_y, pos_x in positions:
                canvas[pos_y:pos_y+800, pos_x:pos_x+600] = final_photo
                # Малюємо тонку сіру лінію навколо кожного фото для зручності вирізання
                cv2.rectangle(canvas, (pos_x, pos_y), (pos_x + 600, pos_y + 800), (210, 210, 210), 2)
            
            st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), caption="Зразок готового аркуша (6 шт.)", width=350)
            
            # Кнопка скачування готового результату
            _, buffer = cv2.imencode('.jpg', canvas)
            st.download_button(
                label="📥 Завантажити готовий аркуш для друку",
                data=buffer.tobytes(),
                file_name="foto_ubd_3x4_x6.jpg",
                mime="image/jpeg"
            )
