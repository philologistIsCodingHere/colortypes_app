from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
import numpy as np
import joblib
import os
from werkzeug.utils import secure_filename
from PIL import Image
import base64
import io

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def load_model():
    try:
        model = joblib.load('color_type_model.pkl')
        return model
    except:
        return None
    
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(image_path):
    try:
        if not os.path.exists(image_path):
            print(f"Файл не существует: {image_path}")
            return None, None
            
        image = cv2.imread(image_path)
        if image is None:
            print(f"OpenCV не смог загрузить: {image_path}")
            return None, None
        
        print(f"Изображение загружено.")
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        if len(faces) == 0:
            print("Лица не обнаружены")
            return None, None
        x, y, w, h = faces[0]
        x, y, w, h = int(x), int(y), int(w), int(h)

        image_with_face = image.copy()
        cv2.rectangle(image_with_face, (x, y), (x + w, y + h), (0, 255, 0), 3)

        processed_filename = 'processed_' + os.path.basename(image_path)
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
        cv2.imwrite(processed_path, image_with_face)
        print(f"Обработанное изображение сохранено: {processed_path}")

        face_roi = image[y:y+h, x:x+w]

        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        
        l, a, b = cv2.split(lab)
        h, s, v = cv2.split(hsv)
        
        features = {
            'warmth': float(np.mean(a)),
            'brightness': float(np.mean(l)),
            'saturation': float(np.mean(s))
        }
        
        with open(processed_path, 'rb') as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        return features, image_base64
        
    except Exception as e:
        print(f"Ошибка в extract_features: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'GET':
        return redirect('/')
    
    if 'photo' not in request.files:
        flash('Пожалуйста, выберите файл')
        return redirect('/')
    
    file = request.files['photo']
    
    if file.filename == '':
        return redirect('/')
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        model = load_model()
        if model is None:
            flash('Модель не найдена! Сначала обучите модель.')
            return redirect('/')
        
        features, image_with_face = extract_features(filepath)
        
        if features is None:
            flash('Не удалось найти лицо на фото. Попробуйте другое изображение.')
            return redirect('/')
        
        X = [[features['warmth'], features['brightness'], features['saturation']]]
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        color_names = {
            'winter': 'ЗИМА ❄️',
            'spring': 'ВЕСНА 🌼', 
            'summer': 'ЛЕТО 🌸',
            'autumn': 'ОСЕНЬ 🍁'
        }
        
        color_colors = {
            'winter': '#4A90E2',
            'spring': '#FFD700', 
            'summer': '#FF69B4',
            'autumn': '#FF8C00'
        }
        
        result_data = {
            'prediction': prediction,
            'prediction_text': color_names[prediction],
            'color': color_colors[prediction],
            'probabilities': [],
            'features': features,
            'image_with_face': image_with_face,
            'original_filename': filename
        }
        
        for i, color_type in enumerate(model.classes_):
            result_data['probabilities'].append({
                'name': color_names[color_type],
                'value': round(probabilities[i] * 100, 1),
                'color': color_colors[color_type]
            })
        
        recommendations = {
            'winter': {
                'title': '❄️ ЗИМА: Холодные, яркие, контрастные тона',
                'makeup': 'Яркие помады, холодные тени, четкие контуры',
                'colors': 'Чистый белый, черный, ярко-синий, фуксия',
                'accessories': 'Серебро, белое золото, яркие контрастные аксессуары'
            },
            'spring': {
                'title': '🌼 ВЕСНА: Теплые, яркие, свежие тона',
                'makeup': 'Теплые персиковые тона, золотистые хайлайтеры',
                'colors': 'Персиковый, коралловый, теплый бежевый, золотистый',
                'accessories': 'Золото, дерево, теплые тона'
            },
            'summer': {
                'title': '🌸 ЛЕТО: Холодные, приглушенные, мягкие тона',
                'makeup': 'Холодные розовые тона, натуральный макияж',
                'colors': 'Серый, голубой, розовый, лавандовый, мятный',
                'accessories': 'Серебро, платина, нежные пастельные аксессуары'
            },
            'autumn': {
                'title': '🍁 ОСЕНЬ: Теплые, приглушенные, насыщенные тона',
                'makeup': 'Землистые тени, теплые румяна, коричневые подводки',
                'colors': 'Терракотовый, оливковый, горчичный, коричневый',
                'accessories': 'Золото, бронза, дерево, теплые камни'
            }
        }
        
        result_data['recommendations'] = recommendations[prediction]
        
        return render_template('result.html', **result_data)
    
    else:
        flash('Разрешены только файлы с расширениями: png, jpg, jpeg, gif')
        return redirect('/')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    port = 5001
    while True:
        try:
            print(f"Пробуем запустить на порту {port}...")
            app.run(debug=True, host='0.0.0.0', port=port)
            break
        except OSError:
            print(f"Порт {port} занят, пробуем следующий...")
            port += 1
            if port > 5010:
                print("Не удалось найти свободный порт")
                break