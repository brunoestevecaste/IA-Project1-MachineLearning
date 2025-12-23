import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from IPython.display import display
from sklearn.model_selection import learning_curve

def plot_learning_curve_bias_variance(estimator, X, y, cv=5):

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, 
        X, 
        y, 
        cv=cv,  
        train_sizes=np.linspace(0.1, 1.0, 10), 
        scoring='accuracy'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(128, 0, 128, 0.2)', 
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        name='Std Train'
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([test_mean + test_std, (test_mean - test_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.2)', 
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        name='Std Test'
    ))

    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean,
        mode='lines+markers',
        name='Media Entrenamiento',
        line=dict(color='purple', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=train_sizes, y=test_mean,
        mode='lines+markers',
        name='Media Validación Cruzada',
        line=dict(color='blue', width=3)
    ))

    fig.update_layout(
        title="Diagnóstico Bias/Varianza (Learning Curve)",
        title_x=0.5, 
        xaxis_title="Número de ejemplos de entrenamiento",
        yaxis_title="Precisión (Accuracy)", 
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
    )

    fig.write_image("./images/learning_curve.png", scale=3, width=1200, height=700)
    fig.show()

def display_feature_importances(feature_names, coefs, top_n=22):

    feature_importance = pd.DataFrame({
        'Característica': feature_names,
        'Peso': coefs,
        'Peso absoluto': abs(coefs) 
    })

    feature_importance = feature_importance.sort_values(by='Peso absoluto', ascending=False)

    print("Importancia de Características:")
    display(feature_importance[['Característica', 'Peso']].head(top_n))

    plot_data = feature_importance.head(top_n).copy()

    plot_data = plot_data.iloc[::-1]

    colors = ['rgba(0, 0, 255, 0.5)' if x > 0 else 'rgba(128, 0, 128, 0.5)' for x in plot_data['Peso']]

    fig = go.Figure(go.Bar(
        x=plot_data['Peso'],
        y=plot_data['Característica'],
        orientation='h',
        marker_color=colors,
        text=plot_data['Peso'].round(4), 
        textposition='auto'
    ))

    fig.update_layout(
        title=f"Importancia de Características",
        title_x=0.5, 
        xaxis_title="Peso",
        yaxis_title="Característica",
        height=600 + (top_n * 10), 
    )

    fig.add_vline(x=0, line_width=1, line_dash="solid", line_color="black")

    fig.write_image("./images/feature_importance.png", scale=3, width=1200, height=700)
    fig.show()

def plot_missing_plotly(data, title, color_scale, threshold=20, filename=None):

    df_plot = data.reset_index()
    df_plot.columns = ['Característica', 'Porcentaje']

    fig = px.bar(
        df_plot,
        x='Porcentaje',
        y='Característica',
        orientation='h',
        text_auto='.1f',  
        color='Porcentaje',
        color_continuous_scale=color_scale,
        title=title
    )

    fig.update_layout(
        title_x=0.5,  
        xaxis_title="Porcentaje",
        yaxis_title="Característica",
        showlegend=False,
        height=500
    )

    fig.add_vline(
        x=threshold, 
        line_dash="dash", 
        line_color="black"
    )
    fig.write_image(filename, scale=3, width=1200, height=700)
    fig.show()