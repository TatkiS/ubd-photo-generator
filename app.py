import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Генератор фото на УБД", layout="centered")

st.title("📸 Професійний генератор фото 3х4 на УБД")
st.write("Програма готує стандартний аркуш **10х15 см** для фотодруку. Всі пропорції захищені від спотворень.")

uploaded_file = st.file_uploader("Завантажте фото військового:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        st.error("Не вдалося прочитати файл. Спробуйте інше зображення.")
    else:
        h_orig, w_orig, _ = img.shape
        
        # Створюємо бічне меню керування
        st.sidebar.header("⚙️ Налаштування кадру")
        mode = st.sidebar.radio(
            "Режим обрізки фото:",
            ["🤖 Автоматичний (пошук обличчя)", "📐 Ручний (кадрування по центру)"]
        )
        
        final_photo = None
        
        if mode == "🤖 Автоматичний (пошук обличчя)":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(60, 60))
            
            if len(faces) == 0:
                st.sidebar.warning("⚠️ Обличчя не знайдено автоматично. Перемкнено в ручний режим.")
                mode = "📐 Ручний (кадрування по центру)"
            else:
                # Беремо найбільше обличчя
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                
                # Інтерактивні слайдери для користувача
                zoom = st.sidebar.slider("🔍 Масштаб обрізки (зум кадру)", 1.4, 3.0, 2.0, 0.1)
                shift_y = st.sidebar.slider("↕️ Позиція голови (вище/нижче)", 0.0, 0.5, 0.22, 0.02)
                
                crop_h = int(h * zoom)
                crop_w = int(crop_h * 3 / 4)
                
                cx = x + w // 2
                start_x = cx - crop_w // 2
                start_y = y - int(crop_h * shift_y)
                
                # Захист від виходу за межі оригінального фото
                if start_x < 0: start_x = 0
                if start_y < 0: start_y = 0
                if start_x + crop_w > w_orig: start_x = w_orig - crop_w
                if start_y + crop_h > h_orig: start_y = h_orig - crop_h
                
                if crop_w > w_orig or crop_h > h_orig:
                    scale = min(w_orig / crop_w, h_orig / crop_h)
                    crop_w = int(crop_w * scale)
                    crop_h = int(crop_h * scale)
                    start_x = max(0, (x + w // 2) - crop_w // 2)
                    start_y = max(0, y - int(crop_h * shift_y))
                    start_x = min(start_x, w_orig - crop_w)
                    start_y = min(start_y, h_orig - crop_h)
                
                cropped = img[start_y:start_y+crop_h, start_x:start_x+crop_w]
                final_photo = cv2.resize(cropped, (600, 800))
                
        if mode == "📐 Ручний (кадрування по центру)" or final_photo is None:
            st.sidebar.info("💡 Фото кадрується чітко по центру під 3:4 без розтягування обличчя.")
            target_ratio = 3 / 4
            current_ratio = w_orig / h_orig
            
            if current_ratio > target_ratio:
                new_w = int(h_orig * target_ratio)
                start_x = (w_orig - new_w) // 2
                cropped = img[0:h_orig, start_x:start_x+new_w]
            else:
                new_h = int(w_orig / target_ratio)
                start_y = (h_orig - new_h) // 2
                cropped = img[start_y:start_y+new_h, 0:w_orig]
                
            final_photo = cv2.resize(cropped, (600, 800))

        # Відображення результатів на екрані (у два стовпчики)
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📸 Одиночне фото")
            st.image(cv2.cvtColor(final_photo, cv2.COLOR_BGR2RGB), caption="Розмір 3х4 см", use_container_width=True)
            
        # Створення преміум-аркуша 10х15 см (2000 x 3000 пікселів)
        canvas = np.ones((3000, 2000, 3), dtype=np.uint8) * 255
        
        # Точні математичні координати для симетричного розміщення 6 фото (2х3)
        positions = [
            (260, 380), (260, 1020),
            (1100, 380), (1100, 1020),
            (1940, 380), (1940, 1020)
        ]
        
        for pos_y, pos_x in positions:
            canvas[pos_y:pos_y+800, pos_x:pos_x+600] = final_photo
            # Тонка світло-сіра рамка, яка полегшить розрізання ножицями
            cv2.rectangle(canvas, (pos_x, pos_y), (pos_x + 600, pos_y + 800), (220, 220, 220), 2)
            
        with col2:
            st.subheader("🖨️ Аркуш для друку 10х15 см")
            st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), caption="Готовий блок (6 шт.)", use_container_width=True)
            
        # Кнопка скачування стандартизованого файлу
        _, buffer = cv2.imencode('.jpg', canvas)
        st.download_button(
            label="📥 Завантажити готовий аркуш 10х15 см",
            data=buffer.tobytes(),
            file_name="ubd_photo_10x15.jpg",
            mime="image/jpeg"
        )
