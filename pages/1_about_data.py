import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="О датасете", layout="wide")

st.title("Описание набора данных")

st.markdown("---")

st.header("1. Предметная область")
st.markdown("""
Датасет содержит результаты **физико-химических анализов** португальских вин 
(красных `red` и белых `white` сортов). 

**Задача:** Бинарная классификация — определение типа вина по его химическому составу.

Данные получены из сертификатов качества и включают 12 признаков + целевую переменную.
""")

try:
    df_red = pd.read_csv('data/winequality-red.csv', sep=";")
    df_white = pd.read_csv('data/winequality-white.csv', sep=";")
    df_red["type"] = 1
    df_white["type"] = 0
    df = pd.concat([df_red, df_white], ignore_index=True)
except:
    st.warning("Файл данных не найден. Убедитесь, что файл `final_data_wine.csv` находится в папке `data/`")
    st.stop()

st.header("2. Общая информация о датасете")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Количество записей", f"{len(df):,}")
with col2:
    st.metric("Количество признаков", len(df.columns))
with col3:
    st.metric("Красные вина", f"{len(df[df['type'] == 1]):,}")
with col4:
    st.metric("Белые вина", f"{len(df[df['type'] == 0]):,}")

st.info(f"**Дисбаланс классов:** Белых вин в {len(df[df['type']==0])//len(df[df['type']==1])} раза больше, чем красных")

# Описание признаков
st.header("3. Описание признаков")

features_df = pd.DataFrame({
    "Признак": [
        "fixed acidity",
        "volatile acidity", 
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "quality"
    ],
    "Описание": [
        "Фиксированная (титруемая) кислотность (г/дм³)",
        "Летучая кислотность (г/дм³)",
        "Лимонная кислота (г/дм³)",
        "Остаточный сахар (г/дм³)",
        "Хлориды (г/дм³)",
        "Свободный диоксид серы SO₂ (мг/дм³)",
        "Общий диоксид серы SO₂ (мг/дм³)",
        "Плотность (г/см³)",
        "Водородный показатель pH",
        "Сульфаты (г/дм³)",
        "Содержание алкоголя (%)",
        "Оценка качества (по шкале 0-10)"
    ],
    "Тип данных": [
        "float32", "float32", "float32", "float32", "float32",
        "int32", "int32", "float32", "float32", "float32",
        "float32", "int32"
    ]
})

st.dataframe(features_df, use_container_width=True)

st.markdown("**Целевая переменная:** `type` — тип вина (0 = white/белое, 1 = red/красное)")

st.header("4. Описательная статистика")
st.dataframe(df.describe().round(3), use_container_width=True)

st.header("5. Предобработка данных и EDA")

st.subheader("5.1. Проверка на пропуски")
missing = df.isnull().sum()
if missing.sum() == 0:
    st.success("Пропущенные значения отсутствуют")
else:
    st.warning(f"Найдено пропусков: {missing.sum()}")
    st.dataframe(missing[missing > 0])

st.subheader("5.2. Проверка на дубликаты")
duplicates = df.duplicated(keep=False).sum()
st.write(f"Количество дубликатов: **{duplicates}** ({duplicates/len(df)*100:.1f}% от датасета)")

if duplicates > 0:
    st.info("""
    **Дубликаты не удалялись** намеренно. 
    Поскольку дублирующиеся записи составляют около трети датасета, 
    было принято решение, что это может быть обусловлено наличием 
    **неучтённых признаков** (например, партия производства, регион, 
    технология ферментации), которые не вошли в исходный набор данных. 
    Удаление таких записей могло бы привести к потере важной информации 
    и искусственному сужению выборки.
    """)

st.subheader("5.3. Распределение целевой переменной")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    df['type'].value_counts().plot(kind='bar', ax=ax, color=['#4CAF50', '#F44336'])
    ax.set_xlabel('Тип вина')
    ax.set_ylabel('Количество')
    ax.set_xticklabels(['Белое (0)', 'Красное (1)'], rotation=0)
    ax.set_title('Распределение классов')
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    percentages = df['type'].value_counts(normalize=True) * 100
    ax.pie(percentages.values, labels=['Белое', 'Красное'], autopct='%1.1f%%', 
           colors=['#4CAF50', '#F44336'])
    ax.set_title('Процентное соотношение')
    st.pyplot(fig)

st.warning("**Наблюдается дисбаланс классов** — белых вин значительно больше. Для решения задачи классификации применялась техника SMOTE (синтетическая передискретизация).")

st.subheader("5.4. Корреляционный анализ")

fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", 
            linewidths=.5, ax=ax)
ax.set_title('Корреляционная матрица признаков', fontsize=14, pad=20)
st.pyplot(fig)

st.markdown("""
**Ключевые выводы по корреляциям:**
- `alcohol` положительно коррелирует с `quality` (0.48)
- `volatile acidity` отрицательно коррелирует с `quality` (-0.20)
- `free sulfur dioxide` и `total sulfur dioxide` сильно коррелируют между собой (0.67)
- `density` отрицательно коррелирует с `alcohol` (-0.78)
""")

st.subheader("5.5. Распределение числовых признаков")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'type' in numeric_cols:
    numeric_cols.remove('type')
if 'quality' in numeric_cols:
    numeric_cols.remove('quality')

fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()

for idx, col in enumerate(numeric_cols[:12]):
    sns.histplot(df[col], kde=True, ax=axes[idx], color='skyblue')
    axes[idx].set_title(f'Распределение: {col}', fontsize=10)
    axes[idx].tick_params(labelsize=8)

plt.tight_layout()
st.pyplot(fig)

st.subheader("5.6. Обработка выбросов")

st.markdown("""
**Метод:** Межквартильный размах (IQR)

**Подход:**
1. Выбросы определялись отдельно для красных и белых вин
2. Для признаков `free sulfur dioxide`, `total sulfur dioxide`, `fixed acidity`, 
   `citric acid`, `residual sugar`, `chlorides` использовался множитель 4.0
3. Для остальных признаков — стандартный множитель 3.0

**Результат:** Удалено **476 записей** (из 6497 до 6021 после очистки)
""")

feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                if col not in ['type', 'quality']]

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for idx, col in enumerate(feature_cols):
    sns.boxplot(data=df, x='type', y=col, ax=axes[idx], 
                palette=['#4CAF50', '#F44336'])
    axes[idx].set_title(f'{col}', fontsize=10)
    axes[idx].set_xlabel('Тип вина (0=белое, 1=красное)')
    axes[idx].tick_params(labelsize=8)

plt.tight_layout()
st.pyplot(fig)

st.subheader("5.7. Распределение числовых признаков после обработки выбросов")

data_final = pd.read_csv('data/final_data_wine.csv', index_col=0)

numeric_cols = data_final.select_dtypes(include=[np.number]).columns.tolist()
if 'type' in numeric_cols:
    numeric_cols.remove('type')
if 'quality' in numeric_cols:
    numeric_cols.remove('quality')

fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()

for idx, col in enumerate(numeric_cols[:12]):
    sns.histplot(data_final[col], kde=True, ax=axes[idx], color='skyblue')
    axes[idx].set_title(f'Распределение: {col}', fontsize=10)
    axes[idx].tick_params(labelsize=8)

plt.tight_layout()
st.pyplot(fig)

st.markdown("""
По графикам видно, что данные стали чище и большинство выбросов убраны
""")