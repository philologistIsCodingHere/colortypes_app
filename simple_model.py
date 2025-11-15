import cv2
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
import joblib

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_simple_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    if len(faces) == 0:
        return None
    
    x, y, w, h = faces[0]
    face_roi = image[y:y+h, x:x+w]
    
    lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    
    l, a, b = cv2.split(lab)
    h, s, v = cv2.split(hsv)
    
    features = {
        'warmth': np.mean(a),       
        'brightness': np.mean(l),  
        'saturation': np.mean(s)    
    }
    
    return features

def create_color_type_dataset():
    data = []
    
    color_types = {
        'winter': 'winter',    
        'spring': 'spring',    
        'summer': 'summer',  
        'autumn': 'autumn'     
    }
    
    for folder_name, color_label in color_types.items():
        folder_path = os.path.join('train', folder_name)
        
        if not os.path.exists(folder_path):
            print(f"Папка {folder_path} не найдена")
            continue
            
        image_files = [f for f in os.listdir(folder_path) 
                      if f.endswith(('.jpg', '.png'))]
        
        print(f"{folder_name}: {len(image_files)} фото")
        
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            features = extract_simple_features(img_path)
            
            if features:
                features['color_type'] = color_label
                features['filename'] = img_file
                data.append(features)
                print(f"{img_file}")
    
    df = pd.DataFrame(data)
    
    if len(df) > 0:
        df.to_csv('color_type_dataset.csv', index=False)
        print(f"\nДатасет создан! Всего: {len(df)} фото")
        
        print("Распределение по цветотипам:")
        for color_type in color_types.values():
            count = sum(df['color_type'] == color_type)
            print(f"{color_type}: {count} фото")
    return df

def train_color_type_model(df):
    print("\nОБУЧАЕМ МОДЕЛЬ...")
    
    X = df[['warmth', 'brightness', 'saturation']]
    y = df['color_type']
    
    print(f"Обучаем на {len(X)} примерах")
    print(f"Цветотипы: {list(y.unique())}")
    
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        random_state=42
    )
    
    model.fit(X, y)
    
    accuracy = model.score(X, y)
    print(f"Модель обучена! Точность: {accuracy:.1%}")

    joblib.dump(model, 'color_type_model.pkl')
    print("Модель сохранена!")
    
    return model

def predict_color_type(model, image_path):
    print(f"\nАНАЛИЗ: {os.path.basename(image_path)}")
    
    features = extract_simple_features(image_path)
    if not features:
        print("Не удалось проанализировать фото")
        return
    
    X_test = [[features['warmth'], features['brightness'], features['saturation']]]
    prediction = model.predict(X_test)[0]
    probabilities = model.predict_proba(X_test)[0]

    color_names = {
        'winter': 'ЗИМА',
        'spring': 'ВЕСНА', 
        'summer': 'ЛЕТО',
        'autumn': 'ОСЕНЬ'
    }
    
    print(f"Параметры:")
    print(f"Теплота: {features['warmth']:.1f}")
    print(f"Яркость: {features['brightness']:.1f}")
    print(f"Насыщенность: {features['saturation']:.1f}")
    
    print(f"\nРЕЗУЛЬТАТ: {color_names[prediction]}")
    
    print(f"\nВЕРОЯТНОСТИ:")
    for i, color_type in enumerate(model.classes_):
        prob = probabilities[i]
        print(f"{color_names[color_type]}: {prob:.1%}")

print("=" * 50)

dataset = create_color_type_dataset()

if dataset is not None:
    model = train_color_type_model(dataset)
    print("\nТЕСТИРОВАНИЕ:")
    print("=" * 40)
    
    if os.path.exists('test'):
        for test_file in os.listdir('test'):
            if test_file.endswith(('.jpg', '.png')):
                test_path = os.path.join('test', test_file)
                predict_color_type(model, test_path)
                print("-" * 40)
    else:
        print("Создай папку 'test' и положи туда фото для теста!")
        
    print("\nГОТОВО! Модель обучена определять все цветотипы!")
else:
    print("Не удалось создать датасет. Проверь папки с фото!")
'''
def explain_color_rules(warmth, brightness, saturation):
    print("\nПРАВИЛА ОПРЕДЕЛЕНИЯ:")
    
    if warmth < 125:
        if brightness > 155 and saturation > 75:
            print("ЗИМА: холодный + яркий + насыщенный")
        else:
            print("ЛЕТО: холодный + приглушенный + мягкий")
    else:
        if brightness > 160 and saturation > 65:
            print("ВЕСНА: теплый + яркий + свежий")
        else:
            print("ОСЕНЬ: теплый + приглушенный + насыщенный")
'''