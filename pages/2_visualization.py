import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Визуализация данных", layout="wide")
st.title("Визуализация зависимостей в наборе данных")

try:
    df = pd.read_csv('data/final_data_wine.csv', index_col=0)
    
    if df['type'].dtype == 'int64' or df['type'].dtype == 'int32':
        df['type_name'] = df['type'].map({0: 'White', 1: 'Red'})
    else:
        df['type_name'] = df['type']

except FileNotFoundError:
    st.error("Файл data/final_data_wine.csv не найден. Убедитесь, что папка data существует.")
    st.stop()

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Распределение классов", "Корреляция", "Boxplots", "Scatter Plot"])

with tab1:
    st.subheader("Визуализация распределений признаков")
    
    plot_type = st.selectbox(
        "Выберите тип визуализации:",
        [
            "Гистограмма признака (все данные)",
            "Гистограмма признака (по классам)",
        ]
    )
    
    feature_cols = [col for col in df.columns if col not in ["type", "Unnamed: 0"]]
    
    if "Гистограмма" in plot_type:
        selected_feature = st.selectbox("Выберите признак:", feature_cols)
    else:
        selected_feature = None

    if plot_type == "Гистограмма признака (все данные)":
        st.write(f"Распределение признака **{selected_feature}** по всему датасету.")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[selected_feature], kde=True, color="skyblue", bins=30, ax=ax)
        ax.set_title(f"Распределение: {selected_feature}")
        ax.set_xlabel(selected_feature)
        ax.set_ylabel("Count")
        st.pyplot(fig)

    elif plot_type == "Гистограмма признака (по классам)":
        st.write(f"Сравнение распределения **{selected_feature}** для красного и белого вина.")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(
            data=df,
            x=selected_feature,
            hue="type",
            kde=True,
            palette={0: 'red', 1: 'blue'},
            bins=30,
            alpha=0.6,
            ax=ax
        )
        ax.set_title(f"Распределение {selected_feature} по типам вина")
        ax.set_xlabel(selected_feature)
        ax.set_ylabel("Count")
        ax.legend(["Белое (0)", "Красное (1)"])
        st.pyplot(fig)

with tab2:
    st.subheader("2. Корреляционная матрица признаков")
    st.write("Показывает, как сильно признаки зависят друг от друга.")
    
    cols_to_corr = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns
    if 'type' in cols_to_corr:
        cols_to_corr = cols_to_corr.drop('type')
    
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[cols_to_corr].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
    st.pyplot(fig2)

with tab3:
    st.subheader("3. Распределение признаков по типам вин (Boxplots)")
    st.write("Выберите признак для анализа:")
    
    features = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
    if 'type' in features: features.remove('type')
    
    selected_feature = st.selectbox("Признак", features)
    
    if selected_feature:
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='type_name', y=selected_feature, data=df, palette=['blue', 'red'], ax=ax3)
        ax3.set_title(f"Распределение '{selected_feature}' по типам вин")
        st.pyplot(fig3)

with tab4:
    st.subheader("4. Зависимость между двумя признаками")
    
    col1, col2 = st.columns(2)
    with col1:
        x_feat = st.selectbox("Ось X", features, index=features.index('alcohol') if 'alcohol' in features else 0)
    with col2:
        y_feat = st.selectbox("Ось Y", features, index=features.index('density') if 'density' in features else 1)
        
    fig4, ax4 = plt.subplots(figsize=(8, 5), dpi=100)

    sns.regplot(data=df[df['type_name'] == 'Red'], 
                x=x_feat, y=y_feat, 
                ax=ax4, 
                scatter_kws={'alpha': 0.5, 's': 30},
                line_kws={'color': 'blue', 'linewidth': 2})
    
    sns.regplot(data=df[df['type_name'] == 'White'], 
                x=x_feat, y=y_feat, 
                ax=ax4, 
                scatter_kws={'alpha': 0.5, 's': 30},
                line_kws={'color': 'red', 'linewidth': 2})
    
    sns.regplot(data=df, x=x_feat, y=y_feat, scatter=False, ci=None,
                    line_kws={"color": "black", "linestyle": "--", "linewidth": 2}, label="общий",
                    ax=ax4)

    sns.scatterplot(data=df, x=x_feat, y=y_feat, hue='type_name', palette=['blue', 'red'], ax=ax4)
    ax4.set_title(f"Зависимость {y_feat} от {x_feat}")
    st.pyplot(fig4)