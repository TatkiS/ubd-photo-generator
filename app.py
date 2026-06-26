import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Генератор фото на УБД", layout="centered")

st.title("📸 Автоматичний генератор фото 3х4 на УБД")
st.write("Програма автоматично знайде обличчя, обріже фото під суворий стандарт 3х4 та підготує аркуш для друку.")

uploaded_file = st.file_uploader("Виберіть та завантажте фото військового:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Читання зображення
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        st.error("Не вдалося завантажити зображення. Спробуйте інший файл.")
    else:
        h_orig, w_orig, _ = img.shape
        
        # Конвертуємо в сірий колір для розпізнавання обличчя
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(80, 80))
        
        if len(faces) == 0:
            st.warning("⚠️ Обличчя не виявлено автоматично. Переконайтеся, що на фото чітко видно обличчя без головного убору.")
        else:
            # Беремо найбільше знайдене обличчя
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            
            # Розрахунок ідеального розміру кадру під документ (пропорція 3:4)
            # Щоб обличчя займало ~70-75% висоти, кадр має бути приблизно в 1.8 рази більшим за висоту самого обличчя
            crop_h = int(h * 1.85)
            crop_w = int(crop_h * 3 / 4)
            
            # Центрування по горизонталі та вертикалі (зміщення вгору, щоб захопити плечі)
            cx = x + w // 2
            start_x = cx - crop_w // 2
            start_y = y - int(crop_h * 0.18)  # Робить ідеальний відступ над головою
            
            # --- РОЗУМНИЙ ЗАХИСТ ВІД ДЕФОРМАЦІЇ ТА ВИХОДУ ЗА МЕЖІ ---
            if start_x < 0: start_x = 0
            if start_y < 0: start_y = 0
            if start_x + crop_w > w_orig: start_x = w_orig - crop_w
            if start_y + crop_h > h_orig: start_y = h_orig - crop_h
            
            # Якщо оригінальне фото занадто мале для правильних пропорцій, масштабуємо рамку
            if crop_w > w_orig or crop_h > h_orig:
                scale = min(w_orig / crop_w, h_orig / crop_h)
                crop_w = int(crop_w * scale)
                crop_h = int(crop_h * scale)
                start_x = max(0, (x + w // 2) - crop_w // 2)
                start_y = max(0, y - int(crop_h * 0.18))
                start_x = min(start_x, w_orig - crop_w)
                start_y = min(start_y, h_orig - crop_h)
            
            end_x = start_x + crop_w
            end_y = start_y + crop_h
            
            # Обрізання без жодного спотворення пропорцій
            cropped = img[start_y:end_y, start_x:end_x]
            
            # Фінальне приведення до чіткого цифрового стандарту
            final_photo = cv2.resize(cropped, (600, 800))
            
            st.image(cv2.cvtColor(final_photo, cv2.COLOR_BGR2RGB), caption="Правильне фото 3х4 (пропорційне)", width=220)
            
            # Створення ідеально відцентрованого аркуша (6 штук)
            st.subheader("🖨️ Готовий аркуш до друку")
            
            padding = 50
            canvas_w = 600 * 2 + padding * 3  # Створює симетричні поля
            canvas_h = 800 * 3 + padding * 4
            
            canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
            
            # Точні координати для розміщення 2х3
            positions = [
                (padding, padding), 
                (padding, 600 + padding * 2),
                (800 + padding * 2, padding), 
                (800 + padding * 2, 600 + padding * 2),
                (1600 + padding * 3, padding), 
                (1600 + padding * 3, 600 + padding * 2)
            ]
            
            for pos_y, pos_x in positions:
                canvas[pos_y:pos_y+800, pos_x:pos_x+600] = final_photo
                # Легкі маркери для зручного розрізання ножицями
                cv2.rectangle(canvas, (pos_x, pos_y), (pos_x + 600, pos_y + 800), (230, 230, 230), 2)
            
            st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), caption="Зразок фінального бланка", width=320)
            
            # Кнопка збереження
            _, buffer = cv2.imencode('.jpg', canvas)
            st.download_button(
                label="📥 Завантажити ідеальний аркуш для друку",
                data=buffer.tobytes(),
                file_name="foto_ubd_correct_3x4.jpg",
                mime="image/jpeg"
            )
