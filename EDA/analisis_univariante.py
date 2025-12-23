import numpy as np
import pandas as pd
import plotly.express as px


def clean_to_float(df, exclude_cols=None):
    """
    Convierte todas las columnas (salvo las excluidas) a float.
    - Strings no numéricos -> NaN
    - Valores negativos    -> NaN
    """
    df_clean = df.copy()
    if exclude_cols is None:
        exclude_cols = []

    for col in df_clean.columns:
        if col in exclude_cols:
            continue

        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean[col] = df_clean[col].astype(float)
        df_clean.loc[df_clean[col] < 0, col] = np.nan

    return df_clean


def dataset_summary(df):
    """
    Extrae una descripción general del dataset:
    - Número de filas y columnas
    - Tipos de datos
    - Nulos por columna
    - Estadísticos básicos (describe)
    """
    summary = {}
    summary["shape"] = df.shape
    summary["dtypes"] = df.dtypes
    summary["missing_values"] = df.isna().sum()
    summary["basic_stats"] = df.describe(include="all").T
    return summary


# Paleta de azules para usar en los gráficos
BLUE_PALETTE = [
    "#D0E1F2",  # azul muy claro
    "#B0D2E7",
    "#84BCDB",
    "#57A0CE",
    "#3383BE",
    "#1764AB"   # azul muy oscuro
]


def plot_univariate(df, max_unique_as_categorical=10):
    """
    Realiza un análisis univariante simple de cada columna usando Plotly:
    - Variables con <= max_unique_as_categorical valores únicos -> gráfico de barras
    - Variables con más valores únicos -> histograma
    """
    for col in df.columns:
        serie = df[col].dropna()

        if serie.empty:
            print(f"La columna '{col}' está vacía (solo NaN). Se omite.")
            continue

        n_unique = serie.nunique()

        # ---------- Categóricas: gráfico de barras ----------
        if n_unique <= max_unique_as_categorical:
            value_counts = serie.value_counts().sort_index()
            x_vals = value_counts.index.astype(str)
            y_vals = value_counts.values

            fig = px.bar(
                x=x_vals,
                y=y_vals,
                title=f"Distribución de {col} (categórica, {n_unique} valores únicos)",
                labels={"x": col, "y": "Frecuencia"},
                color=x_vals,  # para usar la paleta azul
                color_discrete_sequence=BLUE_PALETTE
            )
            fig.update_layout(
                template="simple_white",
                # AÑADIDO: Centrar el título
                title_x=0.5
            )
            fig.show()

        # ---------- Continuas: histograma ----------
        else:
            temp_df = pd.DataFrame({col: serie})

            fig = px.histogram(
                temp_df,
                x=col,
                nbins=30,
                title=f"Distribución de {col} (continua)",
                color_discrete_sequence=[BLUE_PALETTE[3]]
            )
            fig.update_layout(
                yaxis_title="Frecuencia",
                template="simple_white",
                # AÑADIDO: Centrar el título
                title_x=0.5
            )
            fig.show()