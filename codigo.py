import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuración de la página de Streamlit ---
st.set_page_config(
    page_title="Simulación de ML Supervisado Interactivo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Aplicación Interactiva de Modelos de ML Supervisados 🚀")
st.markdown("""
Esta aplicación te permite **subir tu propio archivo CSV**, realizar un análisis exploratorio,
entrenar varios modelos de clasificación supervisada y comparar sus rendimientos.
""")

# --- Carga de Datos CSV ---
st.sidebar.header("Carga tu Conjunto de Datos")
uploaded_file = st.sidebar.file_uploader("Elige un archivo CSV", type="csv")

# Comprobamos si se ha subido un archivo
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("¡Archivo cargado exitosamente!")

        st.subheader("1. Vista Previa del Conjunto de Datos Cargado")
        st.write(f"Conjunto de datos cargado con **{df.shape[0]}** filas y **{df.shape[1]}** columnas.")
        st.dataframe(df.head())

        # --- Selección de la Variable Objetivo (Clase) ---
        st.sidebar.markdown("---")
        st.sidebar.header("Configuración del Conjunto de Datos")
        target_column = st.sidebar.selectbox(
            "Selecciona la columna que representa la variable objetivo (Clase):",
            df.columns
        )

        # Asignar X (características) y y (variable objetivo)
        # Se excluye la columna objetivo para X
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Validar si hay suficientes clases para clasificación
        if y.nunique() < 2:
            st.error("La columna seleccionada no tiene al menos dos clases únicas. Por favor, elige una columna diferente.")
            st.stop()

    except Exception as e:
        st.error(f"Error al leer el archivo. Asegúrate de que sea un archivo CSV válido. Error: {e}")
        st.stop() # Detiene la ejecución si hay un error
else:
    st.info("Por favor, sube un archivo CSV para comenzar. Los modelos y gráficos se mostrarán aquí.")
    st.stop() # Detiene la ejecución si no hay archivo subido

# --- División de los datos en conjuntos de entrenamiento y prueba ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# --- Análisis Exploratorio de Datos (EDA) ---
st.markdown("---")
st.subheader("2. Análisis Exploratorio de Datos (EDA) 📊")

st.markdown("#### Estadísticas Descriptivas")
st.dataframe(df.describe())

st.markdown("#### Matriz de Correlación")
# Selecciona solo las columnas numéricas para el cálculo de la correlación
numeric_df = X.select_dtypes(include=np.number)
if not numeric_df.empty and len(numeric_df.columns) > 1:
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
    plt.title("Matriz de Correlación entre Características Numéricas")
    st.pyplot(fig_corr)
else:
    st.warning("No hay suficientes columnas numéricas para mostrar la matriz de correlación.")

st.markdown("---")
st.markdown("#### Distribución de Características")
# Obtiene solo las columnas numéricas para la selección
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
selected_features_for_hist = st.multiselect(
    "Selecciona características para ver su distribución:",
    options=numeric_cols,
    default=numeric_cols[0:min(3, len(numeric_cols))] if numeric_cols else []
)
if selected_features_for_hist:
    for feature in selected_features_for_hist:
        fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
        sns.histplot(X[feature], kde=True, ax=ax_hist, color='skyblue')
        plt.title(f"Distribución de {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frecuencia")
        st.pyplot(fig_hist)
else:
    st.info("No hay características numéricas para graficar histogramas.")


st.markdown("#### Gráfico de Dispersión Interactivo (Primeras dos características)")
st.write("Visualización de las dos primeras características numéricas principales por clase.")
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) >= 2:
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        x=df[numeric_cols[0]],
        y=df[numeric_cols[1]],
        hue=df[target_column],
        palette='viridis',
        marker='o',
        s=100,
        edgecolor='k',
        ax=ax_scatter
    )
    plt.title(f"Dispersión de '{numeric_cols[0]}' vs '{numeric_cols[1]}' por Clase")
    plt.xlabel(numeric_cols[0])
    plt.ylabel(numeric_cols[1])
    st.pyplot(fig_scatter)
else:
    st.warning("No hay suficientes características numéricas para el gráfico de dispersión.")


# --- Selección de Modelo y Parámetros ---
st.markdown("---")
st.subheader("3. Selección y Entrenamiento de Modelos 🤖")

st.sidebar.header("Configuración de Modelos ML")
model_choices = st.sidebar.multiselect(
    "Selecciona 1 a 5 Modelos de Clasificación:",
    ["K-Nearest Neighbors (KNN)", "Árbol de Decisión", "Clasificador Bayesiano Gausiano", "Support Vector Machine (SVM)", "Regresión Logística"],
    default=["K-Nearest Neighbors (KNN)", "Árbol de Decisión"]
)

models = {}
metrics_results = []

for model_name in model_choices:
    st.markdown(f"#### Parámetros para: {model_name}")
    model = None

    if model_name == "K-Nearest Neighbors (KNN)":
        n_neighbors = st.sidebar.slider(f"KNN: Número de Vecinos (k) para {model_name}", min_value=1, max_value=20, value=5, key=f"{model_name}_k")
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif model_name == "Árbol de Decisión":
        max_depth = st.sidebar.slider(f"Árbol: Profundidad Máxima para {model_name}", min_value=3, max_value=15, value=7, key=f"{model_name}_depth")
        min_samples_leaf = st.sidebar.slider(f"Árbol: Mínimo de Muestras por Hoja para {model_name}", min_value=1, max_value=10, value=3, key=f"{model_name}_leaf")
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
    elif model_name == "Clasificador Bayesiano Gausiano":
        model = GaussianNB()
    elif model_name == "Support Vector Machine (SVM)":
        C_svm = st.sidebar.slider(f"SVM: Parámetro de Regularización (C) para {model_name}", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key=f"{model_name}_C")
        kernel_svm = st.sidebar.selectbox(f"SVM: Kernel para {model_name}", ("rbf", "linear", "poly", "sigmoid"), key=f"{model_name}_kernel")
        model = SVC(C=C_svm, kernel=kernel_svm, random_state=42)
    elif model_name == "Regresión Logística":
        C_lr = st.sidebar.slider(f"Regresión Logística: Parámetro de Regularización (C) para {model_name}", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key=f"{model_name}_C")
        solver_lr = st.sidebar.selectbox(f"Regresión Logística: Solver para {model_name}", ("liblinear", "lbfgs", "sag", "saga"), key=f"{model_name}_solver")
        model = LogisticRegression(C=C_lr, solver=solver_lr, max_iter=1000, random_state=42)

    if model:
        st.write(f"Entrenando {model_name}...")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            st.write(f"### Resultados para {model_name}")
            st.metric(label="Exactitud (Accuracy)", value=f"{accuracy:.4f}")

            with st.expander(f"Ver Reporte de Clasificación para {model_name}"):
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

            with st.expander(f"Ver Matriz de Confusión para {model_name}"):
                fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                cm_display = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Blues, ax=ax_cm)
                st.pyplot(fig_cm)

            if model_name == "Árbol de Decisión":
                with st.expander("Ver Visualización del Árbol de Decisión"):
                    fig_tree, ax_tree = plt.subplots(figsize=(20, 15))
                    plot_tree(model, filled=True, feature_names=X.columns.tolist(), class_names=[str(c) for c in sorted(y.unique())], ax=ax_tree, fontsize=8)
                    st.pyplot(fig_tree)

            metrics_results.append({
                "Modelo": model_name,
                "Exactitud": accuracy,
                "Precisión (Macro Avg)": report['macro avg']['precision'],
                "Recall (Macro Avg)": report['macro avg']['recall'],
                "F1-Score (Macro Avg)": report['macro avg']['f1-score']
            })
        except Exception as e:
            st.error(f"Error al entrenar {model_name}: {e}")

# --- Comparación de Modelos ---
st.markdown("---")
st.subheader("4. Comparación de Modelos ✨")

if metrics_results:
    comparison_df = pd.DataFrame(metrics_results).set_index("Modelo")
    st.dataframe(comparison_df.style.highlight_max(axis=0, color='lightgreen'))

    st.markdown("#### Gráfico de Comparación de Exactitud")
    fig_comp, ax_comp = plt.subplots(figsize=(10, 6))
    sns.barplot(x=comparison_df.index, y="Exactitud", data=comparison_df, palette="viridis", ax=ax_comp)
    plt.ylim(0, 1)
    plt.title("Comparación de Exactitud de los Modelos")
    plt.ylabel("Exactitud")
    plt.xlabel("Modelo")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_comp)
else:
    st.warning("Selecciona al menos un modelo para ver la comparación.")

st.markdown("---")
st.markdown("¡Gracias por usar esta aplicación interactiva!")
