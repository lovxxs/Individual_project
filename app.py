import streamlit as st
import joblib
import scipy.sparse
import pandas as pd
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

sentiment_pipeline = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")
def calculate_sentiment(text):
    result = sentiment_pipeline(text)[0]
    label = result['label']
    score = result['score']
    
    if label == "POSITIVE":
        return score
    elif label == "NEGATIVE":
        return -score
    else:
        return 0.0

nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

def calculate_quality_score(text):
    words = text.split()
    filtered_words = [word for word in words if len(word) > 2 and word.lower() not in stop_words]
    word_count = len(filtered_words)
    unique_words = len(set(filtered_words))
    diversity = unique_words / word_count if word_count > 0 else 0
    
    quality_score = min(5.0, max(1.0, 2 + diversity * 3)) 
    return quality_score

corpus = [
    "В этой статье рассматриваются финансы и подчеркивается, что необходимо учитывать различные компромиссы.",
    "Недавно я начал заниматься спортом в своей повседневной жизни и обнаружил, что забота о личной жизни остается главной заботой...",
    "Следующая краткая информация о технологиях показывает, что вопросы конфиденциальности остаются главной проблемой. Отзывы сообщества...",
    "Как человек, следящий за развитием технологий, я считаю, что реакция сообщества была исключительно положительной..."
]

plagiarism_vectorizer = TfidfVectorizer()
corpus_vectors = plagiarism_vectorizer.fit_transform(corpus)

def calculate_plagiarism_score(text):
    text_vector = plagiarism_vectorizer.transform([text])
    similarities = cosine_similarity(text_vector, corpus_vectors).flatten()
    max_similarity = max(similarities)
    return max_similarity

@st.cache_resource
def load_model_components():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, vectorizer, scaler

def predict_author(text, model, vectorizer, scaler):
    text_vectorized = vectorizer.transform([text])

    length_chars = len(text)
    length_words = len(text.split())
    quality_score = calculate_quality_score(text)
    sentiment = calculate_sentiment(text)
    plagiarism_score = calculate_plagiarism_score(text)
    
    numeric_data = {
        'length_chars': [length_chars],
        'length_words': [length_words],
        'quality_score': [quality_score],
        'sentiment': [sentiment],
        'plagiarism_score': [plagiarism_score]
    }
    numeric_df = pd.DataFrame(numeric_data)
    
    numeric_scaled = scaler.transform(numeric_df)
    
    combined_features = scipy.sparse.hstack([text_vectorized, numeric_scaled])
    
    prediction = model.predict(combined_features)[0]
    return "Человек" if prediction == 0 else "ИИ"

st.title("Кем написан текст? ИИ или человек?")
st.markdown("---")

model, vectorizer, scaler = load_model_components()

user_input = st.text_area("Введите текст:", "")

if st.button("Определить автора"):
    if user_input.strip() == "":
        st.warning("Пожалуйста, введите текст.")
    else:
        sentiment = calculate_sentiment(user_input)
        quality_score = calculate_quality_score(user_input)
        plagiarism_score = calculate_plagiarism_score(user_input)
        
        st.write(f"**Эмоциональная окраска:** {sentiment:.2f}")
        st.write(f"**Качество текста:** {quality_score:.2f}")
        st.write(f"**Уровень плагиата:** {plagiarism_score:.2f}")
        
        result = predict_author(user_input, model, vectorizer, scaler)
        st.success(f"**Текст написан: {result}**", icon="✅")

menu = ["Главная", "О проекте", "Тестовые примеры", "Настройки"]
choice = st.sidebar.selectbox("Меню", menu)

if choice == "Главная":
    st.title("Автор проекта.")
    st.write("Световой Артем, 3391 группа.")
elif choice == "О проекте":
    st.title("О проекте")
    st.write("Это проект создан для определения автора текста: человек или ИИ.")
elif choice == "Тестовые примеры":
    st.title("Тестовые примеры")
    test_texts = [
        "Недавно я познакомился с образованием в своей повседневной жизни и обнаружил, что результаты многообещающие, но...",
        "Искусственный интеллект — это технология будущего."
    ]
    for text in test_texts:
        result = predict_author(text, model, vectorizer, scaler)
        st.write(f"'{text}' → {result}")
