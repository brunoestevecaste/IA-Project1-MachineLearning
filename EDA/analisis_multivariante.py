import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go


# ============================
# Variables continuas / categóricas
# ============================

def get_variable_types(df, label_col="label"):
    """
    Devuelve listas de columnas numéricas continuas y categóricas
    basadas en el significado del dataset de heart disease.
    Si alguna no existe en el df, se ignora.
    """
    continuous_candidates = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical_candidates = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

    continuous_cols = [c for c in continuous_candidates if c in df.columns and c != label_col]
    categorical_cols = [c for c in categorical_candidates if c in df.columns and c != label_col]

    already = set(continuous_cols + categorical_cols + [label_col])
    other_num = [c for c in df.columns
                 if c not in already and np.issubdtype(df[c].dtype, np.number)]

    continuous_cols += other_num

    return continuous_cols, categorical_cols


# ============================
# Resumen por label
# ============================

def multivariate_summary_by_label(df, label_col="label"):
    """
    Calcula estadísticas descriptivas de las variables numéricas continuas
    por cada valor de label.
    """
    if label_col not in df.columns:
        raise ValueError(f"La columna '{label_col}' no está en el DataFrame.")

    continuous_cols, _ = get_variable_types(df, label_col=label_col)
    grouped = df.groupby(label_col)

    summary = {
        "counts_by_label": grouped.size(),
        "mean_by_label": grouped[continuous_cols].mean(),
        "median_by_label": grouped[continuous_cols].median(),
        "std_by_label": grouped[continuous_cols].std()
    }
    return summary


# Paleta de azules (discreta) para boxplots y barras
BLUE_PALETTE = [
    "#D0E1F2",  # azul muy claro
    "#B0D2E7",
    "#84BCDB",
    "#57A0CE",
    "#3383BE",
    "#1764AB"   # azul muy oscuro
]


# ============================
# Boxplots continuas vs label (Plotly)
# ============================

def plot_numeric_by_label(df, label_col="label"):
    """
    Genera boxplots interactivos (Plotly) de cada variable numérica continua frente a label.
    """
    if label_col not in df.columns:
        raise ValueError(f"La columna '{label_col}' no está en el DataFrame.")

    continuous_cols, _ = get_variable_types(df, label_col=label_col)

    for col in continuous_cols:
        fig = px.box(
            df,
            x=label_col,
            y=col,
            color=label_col,
            color_discrete_sequence=BLUE_PALETTE,
            title=f"{col} por {label_col}"
        )
        fig.update_layout(
            xaxis_title=label_col,
            yaxis_title=col,
            template="simple_white",
            # AÑADIDO: Centrar el título
            title_x=0.5
        )
        fig.show()


# ============================
# Categóricas vs label (barras apiladas, Plotly)
# ============================

def plot_categorical_by_label(df, label_col="label", categorical_cols=None):
    """
    Genera gráficos de barras apiladas (Plotly) para variables categóricas vs label.
    Muestra la distribución de label dentro de cada categoría.
    """
    if label_col not in df.columns:
        raise ValueError(f"La columna '{label_col}' no está en el DataFrame.")

    if categorical_cols is None:
        _, categorical_cols = get_variable_types(df, label_col=label_col)

    labels = sorted(df[label_col].dropna().unique())

    for col in categorical_cols:
        ct = pd.crosstab(df[col], df[label_col], normalize="index")

        fig = go.Figure()

        for i, lab in enumerate(labels):
            if lab not in ct.columns:
                continue
            fig.add_bar(
                name=str(lab),
                x=ct.index.astype(str),
                y=ct[lab],
                marker_color=BLUE_PALETTE[i % len(BLUE_PALETTE)]
            )

        fig.update_layout(
            barmode="stack",
            title=f"Distribución de {label_col} dentro de cada categoría de {col}",
            xaxis_title=col,
            yaxis_title="Proporción",
            legend_title=label_col,
            template="simple_white",
            # AÑADIDO: Centrar el título
            title_x=0.5
        )
        fig.show()


# ============================
# Matriz de correlación (Plotly, azules + números)
# ============================

def plot_correlation_matrix(df, cols=None):
    """
    Dibuja la matriz de correlaciones con Plotly:
    - mapa de calor en escala azul
    - valores numéricos anotados en cada celda.
    """
    if cols is None:
        corr = df.select_dtypes(include=[np.number]).corr()
    else:
        corr = df[cols].select_dtypes(include=[np.number]).corr()

    # Escala de azules (claro → oscuro)
    # Nota: He cambiado la variable de PURPLE_scale a BLUE_scale y el colorscale a 'Blues'.
    BLUE_scale = [
        [0.0,  "#F3E5F5"], # Podrías cambiar estos colores a tonos de azul para más consistencia
        [0.5,  "#84BCDB"],
        [1.0,  "#1764AB"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='Blues',
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlación")
        )
    )

    # Anotaciones numéricas
    for i, row in enumerate(corr.index):
        for j, col in enumerate(corr.columns):
            val = corr.iloc[i, j]
            text_color = "white" if abs(val) > 0.5 else "black"
            fig.add_annotation(
                x=col,
                y=row,
                text=f"{val:.2f}",
                showarrow=False,
                font=dict(color=text_color, size=10)
            )

    fig.update_layout(
        title="Matriz de correlación",
        xaxis=dict(tickangle=45),
        template="simple_white",
        # AÑADIDO: Centrar el título
        title_x=0.5
    )

    fig.show()
    return corr