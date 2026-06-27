import streamlit as st
import cv2
import numpy as np
from rembg import remove, new_session

st.set_page_config(page_title="Генератор фото на УБД", layout="wide")

st.title("📸 Професійний Студійний Генератор Фото 3х4")
st.write("Використовуйте повзунки в меню ліворуч, щоб ідеально відцентрувати фотографію.")

uploaded_file = st.file_uploader("Завантажте оригінальне фото військового:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        st.error("Не вдалося прочитати файл. Спробуйте інше зображення.")
    else:
        h_orig, w_orig, _ = img.shape
        
        st.sidebar.header("🧠 1. Алгоритм фону")
        model_choice = st.sidebar.radio(
            "Оберіть нейромережу:",
            [
                "🟢 ISNet (Найкращий для форми)", 
                "🟡 U2Net (Стандартний)", 
                "⚪ Без видалення фону"
            ]
        )
        
        model_map = {
            "🟢 ISNet (Найкращий для форми)": "isnet-general-use",
            "🟡 U2Net (Стандартний)": "u2net"
        }

        img_no_bg = img.copy()

        if model_choice != "⚪ Без видалення фону":
            session_name = model_map[model_choice]
            with st.spinner(f'🤖 Завантаження {session_name}...'):
                session = new_session(session_name)
                rgba_output = remove(img, session=session, post_process_mask=True)
                
                white_bg = np.ones_like(img) * 255
                alpha = rgba_output[:, :, 3] / 255.0
                alpha = np.expand_dims(alpha, axis=2)
                img_no_bg = (rgba_output[:, :, :3] * alpha + white_bg * (1 - alpha)).astype(np.uint8)

        st.sidebar.markdown("---")
        st.sidebar.header("⚙️ 2. Ручне редагування кадру")
        
        mode = st.sidebar.radio(
            "Початкова точка:",
            ["🤖 Відштовхуватись від обличчя", "📐 Відштовхуватись від центру фото"]
        )
        
        # Повзунки для ручного редагування
        zoom = st.sidebar.slider("🔍 Масштаб (Ближче / Далі)", 1.0, 5.0, 2.1, 0.1)
        shift_y = st.sidebar.slider("↕️ Зсув (Вгору / Вниз)", -1.0, 1.0, 0.25, 0.05)
        shift_x = st.sidebar.slider("↔️ Зсув (Вліво / Вправо)", -1.0, 1.0, 0.0, 0.05)
        
        # Визначення базових координат залежно від режиму
        if mode == "🤖 Відштовхуватись від обличчя":
            gray = cv2.cvtColor(img_no_bg, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(60, 60))
            
            if len(faces) == 0:
                st.sidebar.warning("Обличчя не знайдено. Використовується центр фото.")
                base_cx, base_cy = w_orig // 2, h_orig // 2
                base_h = h_orig // 2
            else:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                base_cx, base_cy = x + w // 2, y + h // 2
                base_h = h
        else:
            base_cx, base_cy = w_orig // 2, h_orig // 2
            base_h = h_orig // 2

        # Розрахунок рамки кадрування (завжди чітко 3:4)
        crop_h = int(base_h * zoom)
        crop_w = int(crop_h * 3 / 4)
        
        # Застосування ручних зсувів користувача
        start_x = base_cx - crop_w // 2 + int(crop_w * shift_x)
        start_y = base_cy - crop_h // 2 - int(crop_h * shift_y)
        
        # АВТОМАТИЧНИЙ ЗАХИСТ ПРОПОРЦІЙ ТА ДОДАВАННЯ ФОНУ
        pad_top = max(0, -start_y)
        pad_bottom = max(0, (start_y + crop_h) - h_orig)
        pad_left = max(0, -start_x)
        pad_right = max(0, (start_x + crop_w) - w_orig)
        
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            img_padded = cv2.copyMakeBorder(img_no_bg, pad_top, pad_bottom, pad_left, pad_right, 
                                            cv2.BORDER_CONSTANT, value=[255, 255, 255])
            start_x += pad_left
            start_y += pad_top
            cropped = img_padded[start_y:start_y+crop_h, start_x:start_x+crop_w]
        else:
            cropped = img_no_bg[start_y:start_y+crop_h, start_x:start_x+crop_w]
            
        final_photo = cv2.resize(cropped, (600, 800))

        # --- Виведення на екран ---
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📸 Результат 3х4")
            st.image(cv2.cvtColor(final_photo, cv2.COLOR_BGR2RGB), caption="Ваше фото", use_container_width=True)
            
        canvas = np.ones((3000, 2000, 3), dtype=np.uint8) * 255
        
        positions = [
            (260, 380), (260, 1020),
            (1100, 380), (1100, 1020),
            (1940, 380), (1940, 1020)
        ]
        
        for pos_y, pos_x in positions:
            canvas[pos_y:pos_y+800, pos_x:pos_x+600] = final_photo
            cv2.rectangle(canvas, (pos_x, pos_y), (pos_x + 600, pos_y + 800), (230, 230, 230), 2)
            
        with col2:
            st.subheader("🖨️ Макет 10х15 см (6 шт)")
            st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), caption="Готовий блок для друку", use_container_width=True)
            
        _, buffer = cv2.imencode('.jpg', canvas)
        st.download_button(
            label="📥 Завантажити макет 10х15 см",
            data=buffer.tobytes(),
            file_name="ubd_photo_10x15_ready.jpg",
            mime="image/jpeg"
        )
