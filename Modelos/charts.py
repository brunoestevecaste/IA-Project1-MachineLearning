import plotly.graph_objects as go

def plot_accuracy_circle(accuracy, title="Accuracy"):
    value = round(accuracy * 100, 2)

    fig = go.Figure(
        data=[
            go.Pie(
                values=[value, 100 - value],
                hole=0.7,
                sort=False,
                direction="clockwise",
                marker=dict(
                    colors=["blue", "lightgray"]
                ),
                textinfo="none", 
                hoverinfo="skip",
                showlegend=False
            )
        ]
    )

    fig.update_layout(
        title=title,
        title_x=0.5,
        margin=dict(t=60, b=20, l=20, r=20),
        width=400,
        height=400,
        annotations=[
            dict(
                text=f"{value}%",
                x=0.5, y=0.5,
                xanchor="center", yanchor="middle",
                showarrow=False,
                font=dict(size=32, color="black")
            )
        ]
    )

    fig.show()

    fig.write_image("./images/accuracy.png", scale=3, width=1200, height=700)