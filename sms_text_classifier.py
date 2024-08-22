# Название файла: sms_text_classifier.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Загрузка данных
url = 'https://path/to/your/sms_spam_collection.csv'  # Замените на URL вашего файла или загрузите локально
data = pd.read_csv(url, delimiter='\t', header=None, names=['label', 'message'])

# Преобразование меток
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Разделение данных на обучающий и тестовый наборы
X = data['message']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Токенизация и паддинг
max_words = 10000
max_len = 150

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_len)

X_test_sequences = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_len)

# Создание модели
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(X_train_padded, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Оценка модели
loss, accuracy = model.evaluate(X_test_padded, y_test)
print(f'Accuracy: {accuracy:.2f}')

# Функция для предсказания
def predict_message(message):
    seq = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded)
    likelihood = pred[0][0]
    label = 'ham' if likelihood < 0.5 else 'spam'
    return [likelihood, label]

# Тестирование функции
test_message = "Congratulations! You've won a $1000 gift card. Call now to claim your prize."
result = predict_message(test_message)
print(f'Prediction: {result}')
