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
        print(f"🔍 Пытаемся загрузить: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"❌ Файл не существует: {image_path}")
            return None, None
            
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ OpenCV не смог загрузить: {image_path}")
            return None, None
        
        print(f"✅ Изображение загружено, размер: {image.shape}")
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        print(f"🔍 Найдено лиц: {len(faces)}")
        
        if len(faces) == 0:
            print("❌ Лица не обнаружены")
            return None, None
        
        # Берем первое найденное лицо
        x, y, w, h = faces[0]
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # Создаем изображение с выделенным лицом ДО извлечения признаков
        image_with_face = image.copy()
        cv2.rectangle(image_with_face, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Сохраняем обработанное изображение
        processed_filename = 'processed_' + os.path.basename(image_path)
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
        cv2.imwrite(processed_path, image_with_face)
        print(f"💾 Обработанное изображение сохранено: {processed_path}")
        
        # Теперь извлекаем признаки из оригинального лица
        face_roi = image[y:y+h, x:x+w]
        
        # Цветовой анализ
        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        
        l, a, b = cv2.split(lab)
        h, s, v = cv2.split(hsv)
        
        features = {
            'warmth': float(np.mean(a)),
            'brightness': float(np.mean(l)),
            'saturation': float(np.mean(s))
        }
        
        print(f"📊 Извлеченные признаки: {features}")
        
        return features, processed_filename
        
    except Exception as e:
        print(f"❌ Ошибка в extract_features: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    print("=" * 50)
    print("🎯 ANALYZE CALLED!")
    print(f"📦 Method: {request.method}")
    print("=" * 50)
    
    if request.method == 'GET':
        return redirect('/')
    
    if 'photo' not in request.files:
        print("❌ Файл не найден в request.files")
        flash('❌ Пожалуйста, выберите файл')
        return redirect('/')
    
    file = request.files['photo']
    print(f"📁 Получен файл: {file.filename}")
    
    if file.filename == '':
        print("❌ Имя файла пустое")
        flash('❌ Файл не выбран')
        return redirect('/')
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"💾 Файл сохранен: {filepath}")
        
        model = load_model()
        if model is None:
            print("❌ Модель не найдена!")
            flash('❌ Модель не найдена! Сначала обучите модель.')
            return redirect('/')
        
        features, image_with_face = extract_features(filepath)
        
        if features is None:
            print("❌ Не удалось проанализировать фото")
            flash('❌ Не удалось найти лицо на фото. Попробуйте другое изображение.')
            return redirect('/')
        
        # Предсказываем цветотип
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
        
        # Добавляем вероятности
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
        
        print("✅ Анализ завершен успешно!")
        return render_template('result.html', **result_data)
    
    else:
        print("❌ Неподдерживаемый формат файла")
        flash('❌ Разрешены только файлы с расширениями: png, jpg, jpeg, gif')
        return redirect('/')

if __name__ == '__main__':
    # Создаем папку для загрузок если её нет
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
        print(f"📁 Создана папка: {app.config['UPLOAD_FOLDER']}")
    
    # Пробуем разные порты
    port = 5001
    while True:
        try:
            print(f"🔄 Пробуем запустить на порту {port}...")
            app.run(debug=True, host='0.0.0.0', port=port)
            break
        except OSError:
            print(f"❌ Порт {port} занят, пробуем следующий...")
            port += 1
            if port > 5010:
                print("😞 Не удалось найти свободный порт")
                break