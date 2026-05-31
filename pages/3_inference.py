import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Предсказание типа вина", layout="wide")
st.title("Предсказание типа вина")
st.markdown("Введите параметры химического состава вина или загрузите CSV файл.")

MODEL_PATHS = {
    "ML1: Классическая (LogReg)": "models/ML1_LogReg.pkl",
    "ML2: Бустинг (GradientBoosting)": "models/ML2_GradientBoosting.pkl",
    "ML3: Продвинутый градиентный бустинг (CatBoost)": "models/ML3_CatBoost.cbm",
    "ML4: Бэггинг (BaggingClassifier)": "models/ML4_Bagging.pkl",
    "ML5: Стэкинг (StackingClassifier)": "models/ML5_Stacking.pkl",
    "ML6: Нейросеть": "models/ML6_NeuralNet.keras"
}

@st.cache_resource
def load_models():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    loaded_models = {}
    for name, path in MODEL_PATHS.items():
        try:
            if path.endswith('.keras'):
                import tensorflow as tf
                loaded_models[name] = tf.keras.models.load_model(path)
            elif path.endswith('.cbm'):
                import catboost
                model = catboost.CatBoostClassifier()
                model.load_model(path)
                loaded_models[name] = model
            else:
                with open(path, 'rb') as f:
                    loaded_models[name] = pickle.load(f)
        except Exception as e:
            st.error(f"Ошибка загрузки {name}: {e}")
            st.stop()
            
    return loaded_models, scaler

models, scaler = load_models()

def predict_wine(scaled_array, model, model_name):
    if isinstance(scaled_array, pd.DataFrame):
        scaled_array = scaled_array.values
    
    if model_name == "ML6: Нейросеть":
        prob = model.predict(scaled_array, verbose=0)[0][0]
        pred_class = 1 if prob > 0.5 else 0
        return pred_class, prob
    elif model_name == "ML3: CatBoost":
        pred_class = model.predict(scaled_array)[0]
        prob = model.predict_proba(scaled_array)[0][1]
        return pred_class, prob
    else:
        pred_class = model.predict(scaled_array)[0]
        prob = model.predict_proba(scaled_array)[0][1]
        return pred_class, prob

features = {
    "fixed acidity": (3.0, 16.0, 0.1, "г/дм³"),
    "volatile acidity": (0.0, 2.0, 0.01, "г/дм³"),
    "citric acid": (0.0, 2.0, 0.01, "г/дм³"),
    "residual sugar": (0.0, 70.0, 0.1, "г/дм³"),
    "chlorides": (0.0, 0.6, 0.01, "г/дм³"),
    "free sulfur dioxide": (1.0, 300.0, 1.0, "мг/дм³"),
    "total sulfur dioxide": (6.0, 450.0, 1.0, "мг/дм³"),
    "density": (0.98, 1.05, 0.0001, "г/см³"),
    "pH": (2.7, 4.0, 0.01, "pH"),
    "sulphates": (0.2, 2.0, 0.01, "г/дм³"),
    "alcohol": (8.0, 15.0, 0.1, "%"),
    "quality": (3, 10, 1, "балл"),
}

required_cols = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol", "quality"
]

tabs = st.tabs(["Ручной ввод", "Загрузка CSV"])

with tabs[0]:
    st.subheader("Введите характеристики вина")
    
    user_inputs = {}
    cols = st.columns(3)
    
    for i, (name, (mn, mx, step, unit)) in enumerate(features.items()):
        with cols[i % 3]:
            if name == "density" or name == "chlorides":
                val = st.number_input(
                    f"{name} ({unit})",
                    min_value=float(mn),
                    max_value=float(mx),
                    value=float(mn),
                    step=float(step),
                    format="%.4f"
                )
            else:
                val = st.number_input(
                    f"{name} ({unit})",
                    min_value=float(mn),
                    max_value=float(mx),
                    value=float(mn),
                    step=float(step)
                )
            user_inputs[name] = val

    if st.button("Предсказать", type="primary"):
        input_df = pd.DataFrame([user_inputs])[required_cols]
        scaled_df = scaler.transform(input_df)
        
        results = []
        for name, model in models.items():
            pred_class, prob = predict_wine(scaled_df, model, name)
            
            wine_type = "Красное вино" if pred_class == 1 else "Белое вино"
            conf = prob if pred_class == 1 else (1 - prob)
            
            results.append({
                "Модель": name,
                "Тип": wine_type,
                "Уверенность": f"{conf:.2%}"
            })
        
        st.dataframe(pd.DataFrame(results), use_container_width=True)

with tabs[1]:
    st.subheader("Загрузка файла для пакетного анализа")
    uploaded_file = st.file_uploader("Загрузите CSV файл с данными", type=["csv"])
    
    if uploaded_file is not None:
        try:
            data_to_predict = pd.read_csv(uploaded_file, index_col=0)
            
            missing_cols = [col for col in required_cols if col not in data_to_predict.columns]
            if missing_cols:
                st.error(f"В файле не хватает колонок: {missing_cols}")
                st.write(f"Ожидаемые: {required_cols}")
            else:
                st.info("Файл загружен. Нажмите кнопку для анализа.")
                st.dataframe(data_to_predict.head(), use_container_width=True)
                
                if st.button("Анализировать все записи"):
                    X = data_to_predict[required_cols]
                    X_scaled = scaler.transform(X)
                    
                    all_predictions = []
                    for name, model in models.items():
                        if name == "ML6: Нейросеть":
                            probs = model.predict(X_scaled, verbose=0)
                            preds = (probs > 0.5).astype(int).flatten()
                        elif name == "ML3: CatBoost":
                            preds = model.predict(X_scaled).astype(int)
                        else:
                            preds = model.predict(X_scaled).astype(int)
                        all_predictions.append(preds)
                    
                    all_predictions = np.array(all_predictions)
                    
                    final_predictions = []
                    for i in range(all_predictions.shape[1]):
                        votes = all_predictions[:, i]
                        red_votes = np.sum(votes == 1)
                        white_votes = np.sum(votes == 0)
                        
                        if red_votes > white_votes:
                            final_predictions.append(1)
                        elif white_votes > red_votes:
                            final_predictions.append(0)
                        else:
                            final_predictions.append(1)
                    
                    df_res = data_to_predict.copy()
                    df_res['Prediction'] = np.where(np.array(final_predictions) == 0, 'White', 'Red')
                    
                    red_count = np.sum(np.array(final_predictions) == 1)
                    white_count = np.sum(np.array(final_predictions) == 0)
                    
                    st.success(f"Прогноз выполнен для {len(df_res)} записей (ансамбль из {len(models)} моделей)")
                    st.write(f"**Красных вин:** {red_count}")
                    st.write(f"**Белых вин:** {white_count}")
                    st.dataframe(df_res, use_container_width=True)
                    
        except Exception as e:
            st.error(f"❌ Ошибка обработки файла: {e}")
            import traceback
            st.code(traceback.format_exc())