# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# --- Configuración de la página ---
st.set_page_config(
    page_title="Comparación de Clasificadores",
    layout="wide",
)

# --- Título y descripción ---
st.title("📊 Comparación de Modelos de Clasificación")
st.markdown("Esta aplicación simula un conjunto de datos y evalúa el rendimiento de tres algoritmos de clasificación: *KNN, **Árbol de Decisión* y *Bayesiano Gaussiano*.")
st.markdown("---")

# --- Creación de los datos simulados ---
# El número de muestras es 300 y el número de características es 6.
n_samples = 300
n_features = 6
X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=4,
    n_redundant=1,
    n_classes=2,
    random_state=42
)
data = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(n_features)])
data['target'] = y

st.subheader("1. Conjunto de Datos Simulados")
st.write(f"Se ha creado un conjunto de datos simulado con *{n_samples} muestras* y *{n_features} columnas*.")
st.write("Aquí puedes ver las primeras 5 filas del conjunto de datos:")
st.dataframe(data.head())
st.write("---")

# --- División del conjunto de datos ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

st.subheader("2. Proceso de Entrenamiento y Evaluación")
st.write(f"El conjunto de datos se ha dividido en un conjunto de entrenamiento ({len(X_train)} muestras) y un conjunto de prueba ({len(X_test)} muestras).")
st.write("Ahora entrenaremos y evaluaremos cada modelo de clasificación.")
st.write("---")

# --- Configuración de los modelos ---
# Inicializamos los modelos que vamos a usar.
knn = KNeighborsClassifier(n_neighbors=5)
decision_tree = DecisionTreeClassifier(random_state=42)
naive_bayes = GaussianNB()

# --- Diccionario de modelos para iterar fácilmente ---
models = {
    'KNN': knn,
    'Árbol de Decisión': decision_tree,
    'Clasificador Bayesiano': naive_bayes
}

# --- Entrenamiento y evaluación de los modelos ---
results = {}
for name, model in models.items():
    # Entrenar el modelo
    model.fit(X_train, y_train)
    # Predecir sobre el conjunto de prueba
    y_pred = model.predict(X_test)
    # Calcular la precisión (accuracy)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

# --- Mostrar los resultados ---
st.subheader("3. Resultados de Precisión (Accuracy)")
st.write("A continuación se muestra la precisión de cada modelo en el conjunto de prueba:")

# Crear un DataFrame para mostrar los resultados en una tabla
results_df = pd.DataFrame(results.items(), columns=['Modelo', 'Precisión'])
st.table(results_df.style.highlight_max(axis=0))

st.markdown("---")

# --- Conclusión y sugerencia de mejora ---
st.subheader("¡Hecho!")
st.info("Puedes modificar los parámetros de los modelos en el código (por ejemplo, n_neighbors para KNN) para ver cómo afecta el rendimiento.")
