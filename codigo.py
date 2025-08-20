import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC # Nuevo modelo: Support Vector Classifier
from sklearn.linear_model import LogisticRegression # Nuevo modelo: Regresión Logística
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
Esta aplicación te permite generar un conjunto de datos simulado, realizar un análisis exploratorio,
entrenar varios modelos de clasificación supervisada y comparar sus rendimientos.
""")

# --- Controles de Generación de Datos en el Sidebar ---
st.sidebar.header("Generación de Datos Simulados")
n_samples = st.sidebar.slider("Número de Muestras", min_value=100, max_value=2000, value=300, step=50)
n_features = st.sidebar.slider("Número de Columnas (Características)", min_value=2, max_value=10, value=6, step=1)
n_classes = st.sidebar.slider("Número de Clases", min_value=2, max_value=5, value=2, step=1)
random_state_data = st.sidebar.number_input("Semilla Aleatoria para Datos", value=42, step=1)

# --- Generación del conjunto de datos simulado ---
@st.cache_data
def generate_simulated_data(n_samples, n_features, n_classes, random_state):
    """
    Genera un conjunto de datos simulado para tareas de clasificación.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(n_features, n_features - 1), # Asegura al menos una característica informativa
        n_redundant=max(0, n_features - min(n_features, n_features - 1) - 1),
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=1.0,
        random_state=random_state,
        flip_y=0.01
    )
    feature_names = [f"Característica_{i+1}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['Clase'] = y
    return df, X, y

# Generar los datos
df, X, y = generate_simulated_data(n_samples, n_features, n_classes, random_state_data)

st.subheader("1. Vista Previa del Conjunto de Datos Simulado")
st.write(f"Conjunto de datos generado con **{df.shape[0]}** muestras y **{df.shape[1]-1}** características.")
st.dataframe(df.head()) # Mostrar las primeras filas del DataFrame

# --- División de los datos en conjuntos de entrenamiento y prueba ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# --- Análisis Exploratorio de Datos (EDA) ---
st.markdown("---")
st.subheader("2. Análisis Exploratorio de Datos (EDA) 📊")

st.markdown("#### Estadísticas Descriptivas")
st.dataframe(df.describe())

st.markdown("#### Matriz de Correlación")
# Excluir la columna 'Clase' para la correlación de características
fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
sns.heatmap(df.drop('Clase', axis=1).corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
plt.title("Matriz de Correlación entre Características")
st.pyplot(fig_corr)

st.markdown("#### Distribución de Características")
selected_features_for_hist = st.multiselect(
    "Selecciona características para ver su distribución:",
    options=df.columns[:-1].tolist(),
    default=df.columns[0:min(3, df.shape[1]-1)].tolist() # Selecciona las primeras 3 por defecto
)
if selected_features_for_hist:
    for feature in selected_features_for_hist:
        fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
        sns.histplot(df[feature], kde=True, ax=ax_hist, color='skyblue')
        plt.title(f"Distribución de {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frecuencia")
        st.pyplot(fig_hist)

st.markdown("#### Gráfico de Dispersión Interactivo (Primeras dos características)")
st.write("Visualización de las dos primeras características principales por clase.")
fig_scatter, ax_scatter = plt.subplots(figsize=(10, 8))
sns.scatterplot(
    x=df.iloc[:, 0], # Primera característica
    y=df.iloc[:, 1], # Segunda característica
    hue=df['Clase'],
    palette='viridis',
    marker='o',
    s=100,
    edgecolor='k',
    ax=ax_scatter
)
plt.title(f"Dispersión de '{df.columns[0]}' vs '{df.columns[1]}' por Clase")
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
st.pyplot(fig_scatter)


# --- Selección de Modelo y Parámetros ---
st.markdown("---")
st.subheader("3. Selección y Entrenamiento de Modelos 🤖")

st.sidebar.header("Configuración de Modelos ML")
model_choices = st.sidebar.multiselect(
    "Selecciona 1 a 5 Modelos de Clasificación:",
    ["K-Nearest Neighbors (KNN)", "Árbol de Decisión", "Clasificador Bayesiano Gausiano", "Support Vector Machine (SVM)", "Regresión Logística"],
    default=["K-Nearest Neighbors (KNN)", "Árbol de Decisión"] # Modelos por defecto
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
                    plot_tree(model, filled=True, feature_names=df.columns[:-1].tolist(), class_names=[str(c) for c in sorted(df['Clase'].unique())], ax=ax_tree, fontsize=8)
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
    plt.ylim(0, 1) # La exactitud va de 0 a 1
    plt.title("Comparación de Exactitud de los Modelos")
    plt.ylabel("Exactitud")
    plt.xlabel("Modelo")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_comp)
else:
    st.warning("Selecciona al menos un modelo para ver la comparación.")

st.markdown("---")
st.markdown("¡Gracias por usar esta aplicación interactiva de ML!")
