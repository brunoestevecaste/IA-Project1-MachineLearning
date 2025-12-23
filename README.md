# 🫀 Predicción de Enfermedad Cardíaca - Pipeline Completo

Proyecto de Machine Learning para la predicción de enfermedades cardíacas utilizando análisis exploratorio, limpieza de datos avanzada, ingeniería de características y modelos de clasificación optimizados.

## 📊 Estructura del Proyecto

```
├── EDA/                          # Análisis Exploratorio de Datos
│   ├── analisis_univariante.py
│   ├── analisis_multivariante.py
│   └── implementation.ipynb
│
├── Limpieza_IC/                  # Limpieza y Feature Engineering
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── pipeline_function.py
│   └── implementation.ipynb
│
└── Modelos/                      # Modelos de Clasificación
    ├── cascade_logistic_model.py
    ├── pipeline_function.py
    └── implementation.ipynb
```

## 🔍 Componentes Principales

### 1. Análisis Exploratorio de Datos
- Análisis univariante y multivariante de variables
- Visualización de distribuciones y correlaciones
- Identificación de patrones en datos faltantes

### 2. Limpieza de Datos e Ingeniería de Características
- **Gestión de valores faltantes y atípicos**: Tratamiento diferenciado para valores codificados como `-9` y `?`.
- **Ingeniería de Características**: Creación de nuevas variables sintéticas para capturar patrones de riesgo.
- **Codificación**: 
    - *Label encoder* para características categóricas con sentido ordinal.
    - *One-hot encoder* para características categóricas con sentido nominal.
- **Estandarización**: Escalado de características mediante Z-Score `StandardScaler`.
- **Modelado y Validación**:
    - Comparación de estrategias utilizando Regresión Logística como modelo base.
    - Validación cruzada estratificada `Stratified K-Fold` con 5 particiones.
    - Optimización de hiperparámetros mediante Grid Search.
- **Evaluación**: Estimación del rendimiento y estabilidad del modelo.
$$
\text{Media de la precisión} \pm \text{Desviación estándar}
$$

- **Producción**: Generación automática del archivo `submission.csv` utilizando la mejor configuración encontrada.
- **Visualizaciones**: Gráficas de la distribución de los nulos, análisis del bias/varianza y análisis de la importancia de las características.
### 3. Modelado
- **Baseline**: Regresión Logística con validación cruzada estratificada (5-fold)
- **Modelos en cascada**:
  - `CascadedLogisticRegression`: Modelo binario (0 vs >0) + multiclase (1-4)
  - `ThresholdedCascadedLogisticRegression`: Modelo con umbral ajustable para clase 0
- **Optimización**: Grid Search sobre hiperparámetros `C`, `penalty`, `solver`

## 📈 Resultados Destacados

| Estrategia de Imputación `-9` | Estrategia de Imputación `?` | CV Accuracy (Test) |
|:------------------------------|:-----------------------------|:------------------:|
| Mediana y Moda                | Mediana y Moda              | **0.59782**        |

**Variable más importante**: `combined_risk` (peso: -0.4175)

## 🎯 Conclusiones Clave

1. **Imputación simple > Métodos avanzados**: La imputación por mediana/moda demostró mejor generalización que KNN y MICE, evitando sobreajuste en datasets pequeños.

2. **Feature Engineering crítico**: Las variables sintéticas (`combined_risk`, `age_chol_interaction`) superaron en importancia a muchas variables originales.

3. **Alto bias, baja varianza**: El modelo sufre de subajuste más que de sobreajuste. La mejora no vendrá de modelos más complejos, sino de mejores features o recuperar variables eliminadas.

4. **Regularización moderada**: Una regularización fuerte (`C` pequeña) fue clave para controlar el ruido inherente en el dataset.

## 🛠️ Tecnologías Utilizadas

- **Python 3.x**
- **Scikit-learn**: Modelado, validación cruzada, Grid Search
- **Pandas/NumPy**: Manipulación de datos
- **Plotly**: Visualizaciones

## 🚀 Uso

Cada carpeta contiene un notebook `implementation.ipynb` con el flujo completo:

```bash
# Ejemplo: Ejecutar pipeline de limpieza
cd Limpieza_IC
jupyter notebook implementation.ipynb
```

## 📝 Notas Adicionales

- El dataset presenta desbalanceo de clases y alta tasa de valores faltantes en variables clave.
- La eliminación de `ca` y `thal` redujo el poder predictivo pero mejoró la estabilidad.
- Los modelos en cascada se exploraron para mejorar la discriminación de la clase 0.

---

**Autores**: Marta Soler Ebri, Javier Gracia, Bruno Esteve, Ignacio Benlloch 
**Fecha**: Diciembre 2025
