# -*- coding: utf-8 -*-

#Importar librerías principales

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, roc_auc_score, f1_score
from sklearn.impute import SimpleImputer # Importación crucial para el modelo de Diabetes
import gradio as gr
from collections import defaultdict # Para el cálculo del umbral ideal

"""Subir los datasets"""

import pandas as pd
import numpy as np

# --- URLs de los Datasets ---
# 1. Dataset de Costos de Seguro Médico
url_ins = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
# 2. Dataset de Predicción de Diabetes (Pima Indians)
url_dia = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

# --- Carga de Datos ---
df_ins = pd.read_csv(url_ins)
df_dia = pd.read_csv(url_dia)

# --- Corrección Crucial para el Dataset de Diabetes (Punto de Mejora) ---
# En el dataset de diabetes, los valores '0' en ciertas columnas médicas
# son en realidad datos faltantes (NaN) y deben ser imputados.
cols_dia_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_dia_missing:
    df_dia[col] = df_dia[col].replace(0, np.nan)

# --- Impresión para verificar la carga ---
print("Dataset de Seguro Médico cargado (Primeras 5 filas):")
print(df_ins.head())
print("\nDataset de Diabetes cargado (Primeras 5 filas y valores faltantes):")
print(df_dia.head())
print("\nConteo de NaNs en Diabetes después de la corrección:")
print(df_dia.isnull().sum())

# --- URLs de los Datasets ---
# 1. Dataset de Costos de Seguro Médico
url_ins = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
# 2. Dataset de Predicción de Diabetes (Pima Indians)
url_dia = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

# --- Carga de Datos ---
df_ins = pd.read_csv(url_ins)
df_dia = pd.read_csv(url_dia)

# --- Corrección Crucial para el Dataset de Diabetes (Punto de Mejora) ---
# En el dataset de diabetes, los valores '0' en ciertas columnas médicas
# son en realidad datos faltantes (NaN) y deben ser imputados.
cols_dia_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_dia_missing:
    df_dia[col] = df_dia[col].replace(0, np.nan)

# --- Impresión para verificar la carga ---
print("Dataset de Seguro Médico cargado (Primeras 5 filas):")
print(df_ins.head())
print("\nDataset de Diabetes cargado (Primeras 5 filas y valores faltantes):")
print(df_dia.head())
print("\nConteo de NaNs en Diabetes después de la corrección:")
print(df_dia.isnull().sum())

"""Preprocesamiento y división de datos"""

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

# Suponiendo que df_ins y df_dia ya fueron cargados y limpiados (como en la respuesta anterior)

# --- 1. DEFINICIÓN DE X e Y Y DIVISIÓN DE DATOS ---

# Modelo de Costos de Seguro Médico (Regresión)
X_ins, y_ins = df_ins.drop('charges', axis=1), df_ins['charges']
# División del 80% para entrenamiento, 20% para prueba
X_ins_tr, X_ins_te, y_ins_tr, y_ins_te = train_test_split(X_ins, y_ins, test_size=0.2, random_state=42)

# Modelo de Predicción de Diabetes (Clasificación)
X_dia, y_dia = df_dia.drop('Outcome', axis=1), df_dia['Outcome']
# División con 'stratify=y_dia' para mantener la proporción de la clase 'Diabetes' en ambos conjuntos
X_dia_tr, X_dia_te, y_dia_tr, y_dia_te = train_test_split(X_dia, y_dia, test_size=0.2, stratify=y_dia, random_state=42)

print(f"Tamaño de entrenamiento del Seguro: {X_ins_tr.shape}")
print(f"Tamaño de entrenamiento de Diabetes: {X_dia_tr.shape}")

# --- 2. DEFINICIÓN DE TRANSFORMADORES DE PREPROCESAMIENTO ---

## 🛠️ A. Preprocesamiento para Costos de Seguro (Regresión Lineal)

# Columnas Numéricas: Edad, IMC, Hijos (Necesitan escalado)
num_ins = ['age', 'bmi', 'children']
# Columnas Categóricas: Sexo, Fumador, Región (Necesitan One-Hot Encoding)
cat_ins = ['sex', 'smoker', 'region']

# Definición del ColumnTransformer para el seguro
pre_ins = ColumnTransformer(
    transformers=[
        # Aplica StandardScaler a las columnas numéricas
        ('num', StandardScaler(), num_ins),
        # Aplica OneHotEncoder a las columnas categóricas
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_ins)
    ],
    remainder='passthrough'
)
print("\nColumnTransformer 'pre_ins' definido.")

## 💉 B. Preprocesamiento para Predicción de Diabetes (Regresión Logística)

# Todas las columnas son numéricas (Pregnancies, Glucose, BMI, etc.)
num_dia = X_dia.columns.tolist()

# Definición del ColumnTransformer para diabetes (IMPUTACIÓN CORREGIDA)
# Se usa un Pipeline dentro del ColumnTransformer para Imputar primero y luego Escalar
pre_dia = ColumnTransformer(
    transformers=[
        ('imputer_scale',
         Pipeline([
             # 1. Imputa los valores faltantes (NaN, antes 0) con la mediana.
             ('imputer', SimpleImputer(strategy='median')),
             # 2. Escala los datos para la Regresión Logística.
             ('scaler', StandardScaler())
         ]),
         num_dia)
    ],
    remainder='passthrough'
)
print("ColumnTransformer 'pre_dia' definido (Incluye Imputación y Escalado).")

"""Entrenamiento de modelos base"""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
import joblib

# Asegúrate de que las variables X_ins_tr, y_ins_tr, pre_ins, X_dia_tr, y_dia_tr, pre_dia
# ya estén definidas y listas de los pasos anteriores.

# --- 1. MODELO DE COSTOS DE SEGURO (REGRESIÓN LINEAL) ---

# 1.1. Definición del Pipeline
# Combina el preprocesamiento 'pre_ins' con el modelo de LinearRegression.
pipe_lin = Pipeline([
    ("prep", pre_ins),
    ("lin", LinearRegression())
])

# 1.2. Entrenamiento
pipe_lin.fit(X_ins_tr, y_ins_tr)

# 1.3. Persistencia (Guardar el modelo)
joblib.dump(pipe_lin, 'pipe_lin.joblib')
print("Modelo de Seguro guardado como 'pipe_lin.joblib'.")

# ------------------------------------------------------------------------

# --- 2. MODELO DE PREDICCIÓN DE DIABETES (REGRESIÓN LOGÍSTICA) ---

# 2.1. Definición del Pipeline
# Combina el preprocesamiento CORREGIDO 'pre_dia' (con imputación) con el modelo de LogisticRegression.
# Se usan C=0.01 (regularización) y max_iter=2000 (robustez) como hiperparámetros de partida.
pipe_log = Pipeline([
    ("prep", pre_dia),
    ("log", LogisticRegression(random_state=42, C=0.01, max_iter=2000))
])

# 2.2. Entrenamiento
pipe_log.fit(X_dia_tr, y_dia_tr)

# 2.3. Persistencia (Guardar el modelo)
joblib.dump(pipe_log, 'pipe_log.joblib')
print("Modelo de Diabetes guardado como 'pipe_log.joblib'.")

"""Determinar y visualizar umbral óptimo"""

from sklearn.metrics import recall_score, precision_score, f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# --- 1. FUNCIÓN PARA ENCONTRAR EL UMBRAL ÓPTIMO (Máximo F1-Score) ---

def find_optimal_threshold(model, X_test, y_test):
    # Obtener las probabilidades de la clase positiva (diabetes=1)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Rango de umbrales a evaluar
    thresholds = np.linspace(0.01, 0.99, 100)
    scores = defaultdict(list)

    # Calcular métricas para cada umbral
    for t in thresholds:
        # Clasificar la predicción usando el umbral actual
        y_pred = (y_pred_proba >= t).astype(int)

        # Almacenar el umbral y las métricas
        scores['threshold'].append(t)
        scores['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        scores['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        scores['precision'].append(precision_score(y_test, y_pred, zero_division=0))

    df_scores = pd.DataFrame(scores)

    # F1-Score
    optimal_t_row = df_scores.loc[df_scores['f1'].idxmax()]

    return df_scores, optimal_t_row

# --- 2. CÁLCULO Y RESULTADOS ---

df_thresholds, optimal_t = find_optimal_threshold(pipe_log, X_dia_te, y_dia_te)

umbral_ideal = optimal_t['threshold']

print(f"--- Resultado del Umbral Óptimo ---")
print(f"Umbral Ideal (Max F1-Score): {umbral_ideal:.4f}")
print(f"Métricas en este umbral:")
print(f"  F1-Score: {optimal_t['f1']:.4f}")
print(f"  Sensibilidad (Recall): {optimal_t['recall']:.4f}")
print(f"  Precisión (Precision): {optimal_t['precision']:.4f}")

# --- 3. VISUALIZACIÓN ---

plt.figure(figsize=(10, 6))
plt.plot(df_thresholds['threshold'], df_thresholds['precision'], label='Precisión (Precision)')
plt.plot(df_thresholds['threshold'], df_thresholds['recall'], label='Sensibilidad (Recall)')
plt.plot(df_thresholds['threshold'], df_thresholds['f1'], label='F1-Score', linestyle='--')

# Marcar el punto del umbral óptimo
plt.axvline(x=umbral_ideal, color='r', linestyle=':', label=f'Umbral óptimo (F1): {umbral_ideal:.4f}')

plt.title('Análisis de Umbral para el Modelo de Predicción de Diabetes')
plt.xlabel('Umbral de Probabilidad')
plt.ylabel('Métrica')
plt.legend()
plt.grid(True, alpha=0.5)
plt.show()

# Guardar el umbral ideal en una variable para usarla en la interfaz de Gradio
umbral_ideal_diabetes = umbral_ideal

"""Visualización de errores en regresión"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# --- 1. GENERAR PREDICCIONES Y RESIDUALES ---
# Generar predicciones en el conjunto de prueba
y_pred_ins = pipe_lin.predict(X_ins_te)

# Calcular los residuales (errores)
residuals = y_ins_te - y_pred_ins

# --- 2. EVALUACIÓN DE MÉTRICAS ---
r2 = r2_score(y_ins_te, y_pred_ins)
rmse = np.sqrt(mean_squared_error(y_ins_te, y_pred_ins))

print(f"--- Evaluación del Modelo de Regresión Lineal ---")
print(f"R² Score (Bondad de ajuste): {r2:.4f}")
print(f"RMSE (Error Cuadrático Medio Raíz): ${rmse:,.2f}")


# --- 3. VISUALIZACIÓN DE ERRORES ---

plt.figure(figsize=(15, 6))

# 3.1. Gráfico de Predicciones vs. Valores Reales (Actual vs. Predicted)
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_ins_te, y=y_pred_ins, alpha=0.6)
# Línea de 45 grados (y = x)
plt.plot([y_ins_te.min(), y_ins_te.max()], [y_ins_te.min(), y_ins_te.max()],
         '--r', linewidth=2, label='Predicción Ideal (y=x)')

plt.title('Predicciones vs. Valores Reales')
plt.xlabel('Cargos Reales del Seguro ($)')
plt.ylabel('Cargos Predichos del Seguro ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# 3.2. Gráfico de Residuales vs. Predicciones (Error Analysis)
plt.subplot(1, 2, 2)
sns.scatterplot(x=y_pred_ins, y=residuals, alpha=0.6)
# Línea de error cero (y = 0)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)

plt.title('Análisis de Residuales (Errores)')
plt.xlabel('Cargos Predichos del Seguro ($)')
plt.ylabel('Residuales (Error: Real - Predicho)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Asegúrate de que pipe_lin, X_ins_te, y_ins_te estén disponibles

# --- 1. GENERAR PREDICCIONES Y RESIDUALES ---
# Generar predicciones en el conjunto de prueba
y_pred_ins = pipe_lin.predict(X_ins_te)

# Calcular los residuales (errores)
residuals = y_ins_te - y_pred_ins

# --- 2. EVALUACIÓN DE MÉTRICAS ---
r2 = r2_score(y_ins_te, y_pred_ins)
rmse = np.sqrt(mean_squared_error(y_ins_te, y_pred_ins))

print(f"--- Evaluación del Modelo de Regresión Lineal ---")
print(f"R² Score (Bondad de ajuste): {r2:.4f}")
print(f"RMSE (Error Cuadrático Medio Raíz): ${rmse:,.2f}")


# --- 3. VISUALIZACIÓN DE ERRORES ---

plt.figure(figsize=(15, 6))

# 3.1. Gráfico de Predicciones vs. Valores Reales (Actual vs. Predicted)
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_ins_te, y=y_pred_ins, alpha=0.6)
# Línea de 45 grados (y = x), donde las predicciones perfectas deberían caer
plt.plot([y_ins_te.min(), y_ins_te.max()], [y_ins_te.min(), y_ins_te.max()],
         '--r', linewidth=2, label='Predicción Ideal (y=x)')

plt.title('Predicciones vs. Valores Reales')
plt.xlabel('Cargos Reales del Seguro ($)')
plt.ylabel('Cargos Predichos del Seguro ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# 3.2. Gráfico de Residuales vs. Predicciones (Error Analysis)
plt.subplot(1, 2, 2)
sns.scatterplot(x=y_pred_ins, y=residuals, alpha=0.6)
# Línea de error cero (y = 0)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)

plt.title('Análisis de Residuales (Errores)')
plt.xlabel('Cargos Predichos del Seguro ($)')
plt.ylabel('Residuales (Error: Real - Predicho)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

"""Matriz de confusión y curva ROC del modelo de diabetes"""

from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Asegúrate de que pipe_log, X_dia_te y y_dia_te estén disponibles
# Usaremos el umbral ideal que encontramos para la Matriz de Confusión
umbral_ideal_diabetes = 0.45 # Valor de referencia

# --- 1. GENERAR PREDICCIONES Y PROBABILIDADES ---

# Probabilidades de la clase positiva (diabetes=1)
y_proba_dia = pipe_log.predict_proba(X_dia_te)[:, 1]

# Predicciones binarias usando el umbral ideal
y_pred_dia = (y_proba_dia >= umbral_ideal_diabetes).astype(int)

# --- 2. MATRIZ DE CONFUSIÓN ---

# 2.1. Calcular la Matriz de Confusión
cm = confusion_matrix(y_dia_te, y_pred_dia)

print(f"--- Matriz de Confusión (Umbral: {umbral_ideal_diabetes:.2f}) ---")
print(classification_report(y_dia_te, y_pred_dia, target_names=['No Diabetes', 'Diabetes'], zero_division=0))

# 2.2. Visualización
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicción Negativa', 'Predicción Positiva'],
            yticklabels=['Real Negativa', 'Real Positiva'])
plt.title(f'Matriz de Confusión (Umbral: {umbral_ideal_diabetes:.2f})')
plt.xlabel('Predicción del Modelo')
plt.ylabel('Valor Real')
plt.show()

# Análisis de la Matriz:
tn, fp, fn, tp = cm.ravel()
print(f"\nMétricas Clave:")
print(f"  Verdaderos Positivos (TP - Aciertos de Diabetes): {tp}")
print(f"  Verdaderos Negativos (TN - Aciertos de No Diabetes): {tn}")
print(f"  Falsos Positivos (FP - Error Tipo I): {fp}")
print(f"  Falsos Negativos (FN - Error Tipo II): {fn}")

# --- 3. CURVA ROC Y ÁREA BAJO LA CURVA (AUC) ---

# 3.1. Calcular la tasa de verdaderos positivos (TPR) y la tasa de falsos positivos (FPR)
fpr, tpr, thresholds = roc_curve(y_dia_te, y_proba_dia)
roc_auc = auc(fpr, tpr)

# 3.2. Visualización
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.4f})')
# Línea de 45 grados (clasificador aleatorio)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Clasificador Aleatorio (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR) / Sensibilidad')
plt.title('Curva ROC para Predicción de Diabetes')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

"""Importancia de características (Random Forest visual)"""

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

# Suponiendo que los datos de entrenamiento (X_ins_tr, y_ins_tr, X_dia_tr, y_dia_tr)
# y los preprocesadores (pre_ins, pre_dia) están definidos.

# --- 1. MODELO DE COSTOS DE SEGURO (RANDOM FOREST REGRESSOR) ---

# 1.1. Crear el pipeline de Regresión con Random Forest
pipe_rf_ins = Pipeline([("prep", pre_ins), ("rf", RandomForestRegressor(n_estimators=100, random_state=42))])

# 1.2. Entrenar el modelo
pipe_rf_ins.fit(X_ins_tr, y_ins_tr)

# 1.3. Extraer la importancia de las características
# Obtener los nombres de las características después de la transformación
feature_names_ins = list(pipe_rf_ins['prep'].get_feature_names_out())
importances_ins = pipe_rf_ins['rf'].feature_importances_

# 1.4. Crear DataFrame para visualización
feature_importance_df_ins = pd.DataFrame({
    'Feature': feature_names_ins,
    'Importance': importances_ins
}).sort_values(by='Importance', ascending=False)

# --- 2. MODELO DE PREDICCIÓN DE DIABETES (RANDOM FOREST CLASSIFIER) ---

# 2.1. Crear el pipeline de Clasificación con Random Forest
pipe_rf_dia = Pipeline([("prep", pre_dia), ("rf", RandomForestClassifier(n_estimators=100, random_state=42))])

# 2.2. Entrenar el modelo
pipe_rf_dia.fit(X_dia_tr, y_dia_tr)

# 2.3. Extraer la importancia de las características
# Obtener los nombres de las características (solo las de imputación/escalado)
feature_names_dia = [col.split('__')[1] for col in pipe_rf_dia['prep'].get_feature_names_out()]
importances_dia = pipe_rf_dia['rf'].feature_importances_

# 2.4. Crear DataFrame para visualización
feature_importance_df_dia = pd.DataFrame({
    'Feature': feature_names_dia,
    'Importance': importances_dia
}).sort_values(by='Importance', ascending=False)


# --- 3. VISUALIZACIÓN COMPARATIVA (Punto 3) ---

plt.figure(figsize=(15, 6))

# Subgráfico 1: Importancia de Características del Seguro Médico
plt.subplot(1, 2, 1)
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance_df_ins.head(10), # Mostrar el top 10
    palette='viridis'
)
plt.title('Influencia en Costos de Seguro (Random Forest Regressor)')
plt.xlabel('Importancia (Gini/MSE Reduction)')
plt.ylabel('Característica')

# Subgráfico 2: Importancia de Características de Diabetes
plt.subplot(1, 2, 2)
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance_df_dia.head(10),
    palette='magma'
)
plt.title('Influencia en Predicción de Diabetes (Random Forest Classifier)')
plt.xlabel('Importancia (Gini Importance)')
plt.ylabel('Característica')

plt.tight_layout()
plt.show()

"""Comparativa general"""

from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, f1_score
import numpy as np
import pandas as pd
import joblib # Necesario si necesitas recargar los modelos si no están en memoria

# Cargar los modelos si no están disponibles en la sesión actual
# pipe_lin = joblib.load('pipe_lin.joblib')
# pipe_log = joblib.load('pipe_log.joblib')

# Definición del umbral ideal para diabetes (ajustar según el cálculo anterior)
umbral_ideal_diabetes = 0.45

# --- 1. EVALUACIÓN DEL MODELO DE SEGURO (REGRESIÓN LINEAL) ---

# Generar predicciones en el conjunto de prueba
y_pred_ins = pipe_lin.predict(X_ins_te)

# Calcular métricas de regresión
r2 = r2_score(y_ins_te, y_pred_ins)
rmse = np.sqrt(mean_squared_error(y_ins_te, y_pred_ins))
mae = np.mean(np.abs(y_ins_te - y_pred_ins)) # Error Absoluto Medio (fácil de interpretar)

# --- 2. EVALUACIÓN DEL MODELO DE DIABETES (REGRESIÓN LOGÍSTICA) ---

# Generar probabilidades y predicciones binarias
y_proba_dia = pipe_log.predict_proba(X_dia_te)[:, 1]
y_pred_dia_opt = (y_proba_dia >= umbral_ideal_diabetes).astype(int)

# Calcular métricas de clasificación
auc_roc = roc_auc_score(y_dia_te, y_proba_dia)
f1_opt = f1_score(y_dia_te, y_pred_dia_opt)
accuracy_default = pipe_log.score(X_dia_te, y_dia_te) # Accuracy con umbral por defecto (0.5)

# --- 3. CONSOLIDACIÓN Y VISUALIZACIÓN DE RESULTADOS ---

# Diccionario para almacenar los resultados
resultados = {
    "Modelo": ["Seguro Médico (Regresión Lineal)", "Predicción de Diabetes (Regresión Logística)"],
    "Tipo de Problema": ["Regresión", "Clasificación Binaria"],
    "Métrica Principal": ["R² Score", "AUC-ROC"],
    "Resultado": [f"{r2:.4f}", f"{auc_roc:.4f}"],
    "Métrica Secundaria 1": ["RMSE", "F1-Score (Umbral Ópt.)"],
    "Resultado Secundario 1": [f"${rmse:,.2f}", f"{f1_opt:.4f}"],
    "Métrica Secundaria 2": ["MAE", "Accuracy (Umbral 0.5)"],
    "Resultado Secundario 2": [f"${mae:,.2f}", f"{accuracy_default:.4f}"]
}

df_comparativa = pd.DataFrame(resultados)

print("           ANÁLISIS COMPARATIVO GENERAL DE MODELOS")

# Imprimir los resultados en formato tabular
print(df_comparativa.to_markdown(index=False))

print("\n\n--- INTERPRETACIÓN DE MÉTRICAS ---\n")
print("Modelo de Seguro (Regresión):")
print(f"  R² Score ({r2:.4f}): Indica que el {r2*100:.1f}% de la varianza en los costos es explicada por el modelo.")
print(f"  RMSE (${rmse:,.2f}): En promedio, el modelo se equivoca por ${rmse:,.2f} en la predicción de costos.")
print("\nModelo de Diabetes (Clasificación):")
print(f"  AUC-ROC ({auc_roc:.4f}): El modelo tiene una probabilidad del {auc_roc*100:.1f}% de clasificar un caso positivo por encima de uno negativo.")
print(f"  F1-Score Opt. ({f1_opt:.4f}): Un buen balance entre Precisión y Sensibilidad usando el umbral ajustado ({umbral_ideal_diabetes:.2f}).")

"""Guardar modelos entrenados"""

import joblib

# 1. Guardar el Pipeline de Regresión Lineal (Seguro)
joblib.dump(pipe_lin, 'pipe_lin.joblib')
print("Modelo de Seguro Médico (pipe_lin.joblib) guardado exitosamente.")

# 2. Guardar el Pipeline de Regresión Logística (Diabetes)
joblib.dump(pipe_log, 'pipe_log.joblib')
print("Modelo de Diabetes (pipe_log.joblib) guardado exitosamente.")

"""Interfaz web interactiva (Gradio)"""

import gradio as gr
import joblib
import pandas as pd
import numpy as np

# --- 1. CARGAR MODELOS Y DEFINICIONES ---

try:
    # Cargar los Pipelines completos (incluye preprocesamiento y modelo)
    pipe_lin = joblib.load('pipe_lin.joblib')
    pipe_log = joblib.load('pipe_log.joblib')
except FileNotFoundError:
    print("Error: Asegúrate de haber guardado 'pipe_lin.joblib' y 'pipe_log.joblib'.")
    # En un entorno de producción, aquí se detendría la ejecución.

# Definición de variables clave
umbral_ideal_diabetes = 0.45
cols_dia_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']


# --- 2. FUNCIONES DE PREDICCIÓN PARA GRADIO ---

def pred_ins(age, sex, bmi, children, smoker, region):
    """Predice el costo del seguro médico."""
    df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": int(children), # Asegurar tipo int
        "smoker": smoker,
        "region": region
    }])

    try:
        costo_predicho = pipe_lin.predict(df)[0]
        # Formatear el resultado en un string
        return f"Costo estimado del seguro: ${costo_predicho:,.2f}"
    except Exception as e:
        return f"Error en la predicción del seguro: {e}"


def pred_dia(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, threshold):
    """Predice la probabilidad de diabetes."""
    # Crear DataFrame de entrada
    df = pd.DataFrame([{
        "Pregnancies": Pregnancies,
        "Glucose": Glucose,
        "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness,
        "Insulin": Insulin,
        "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
        "Age": Age
    }])

    # Reemplazar 0s por NaN para que el pipeline de imputación los maneje
    for col in cols_dia_missing:
        if col in df.columns:
            # Reemplazamos si el valor es numérico 0 o None
            df[col] = df[col].replace(0, np.nan)

    try:
        # Obtener la probabilidad de la clase positiva (1: Diabetes)
        prob = pipe_log.predict_proba(df)[:, 1][0]

        # Clasificar según el umbral proporcionado
        cl = int(prob >= threshold)
        clase = "Diabetes" if cl == 1 else "No diabetes"

        resultado = {
            "Predicción": clase,
            "Probabilidad (Positiva)": f"{prob:.4f}",
            "Umbral Usado": f"{threshold:.2f}"
        }

        # Retornar un string formateado para el Textbox
        return "\n".join([f"{k}: {v}" for k, v in resultado.items()])
    except Exception as e:
        return f"Error en la predicción de diabetes: {e}"


# --- 3. INTERFAZ GRÁFICA (GR.BLOCKS) ---

with gr.Blocks() as demo:
    gr.Markdown("## 🌐 Predicción de costos de seguro y diabetes (Servicio Web)")

    # 3.1. Pestaña de Costos de Seguro (Regresión Lineal)
    with gr.Tab("Cálculo de Costos de Seguro"):
        gr.Markdown("### 💰 Modelo de Regresión Lineal de Costos")

        with gr.Row():
            age=gr.Number(label="Edad", value=30, precision=0, minimum=18, maximum=100, scale=1)
            sex=gr.Dropdown(["male","female"], label="Sexo", value="female", scale=1)
            bmi=gr.Number(label="IMC (kg/m²)", value=25, minimum=15, maximum=55, scale=1)

        with gr.Row():
            children=gr.Number(label="Hijos", value=0, precision=0, minimum=0, maximum=5, scale=1)
            smoker=gr.Dropdown(["yes","no"], label="Fumador", value="no", scale=1)
            region=gr.Dropdown(["southwest", "southeast", "northwest", "northeast"], label="Región", value="southeast", scale=1)

        out1=gr.Textbox(label="Costo Estimado del Seguro", scale=2)

        gr.Button("Predecir Costo", variant="primary").click(
            pred_ins,
            [age, sex, bmi, children, smoker, region],
            out1
        )

    # 3.2. Pestaña de Predicción de Diabetes (Regresión Logística)
    with gr.Tab("Predicción de Diabetes"):
        gr.Markdown("### 💉 Modelo de Regresión Logística de Diabetes")

        # Slider para seleccionar el umbral (Punto 1)
        thr=gr.Slider(
            0.05, 0.95, value=umbral_ideal_diabetes, step=0.01,
            label=f"Umbral de Clasificación (Ideal F1-Score: {umbral_ideal_diabetes:.2f})"
        )

        with gr.Row():
            Pregnancies=gr.Number(label="Embarazos", value=1, precision=0, minimum=0)
            Glucose=gr.Number(label="Glucosa (Concentración)", value=120, minimum=0)
            BloodPressure=gr.Number(label="Presión Arterial", value=70, minimum=0)
            SkinThickness=gr.Number(label="Espesor de Piel", value=20, minimum=0)

        with gr.Row():
            Insulin=gr.Number(label="Insulina", value=80, minimum=0)
            BMI=gr.Number(label="IMC", value=25, minimum=0)
            DiabetesPedigreeFunction=gr.Number(label="Función de Pedigrí", value=0.5, minimum=0)
            Age=gr.Number(label="Edad", value=30, precision=0, minimum=18)

        out2=gr.Textbox(label="Resultado de Predicción", lines=4)

        gr.Button("Predecir Diabetes", variant="primary").click(
            pred_dia,
            [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, thr],
            out2
        )

# --- 4. LANZAR LA INTERFAZ ---
# Usa share=True si necesitas un enlace público temporal para producción/demostración
demo.launch(debug=True)