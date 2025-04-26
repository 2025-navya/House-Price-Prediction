# visualizations.py
import pandas as pd
import plotly.graph_objs as go
import pickle

def create_visuals():
    df = pd.read_csv("Housing.csv")
    model = pickle.load(open("model.pkl", "rb"))
    le_dict = pickle.load(open("label_encoders.pkl", "rb"))

    cat_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                'airconditioning', 'prefarea', 'furnishingstatus']

    for col in cat_cols:
        df[col] = le_dict[col].transform(df[col])

    X = df.drop("price", axis=1)
    y_actual = df["price"]
    y_predicted = model.predict(X)

    # Scatter Plot
    scatter = go.Figure()
    scatter.add_trace(go.Scatter(
        x=list(range(len(y_actual))),
        y=y_actual,
        mode='markers',
        name='Actual',
        marker=dict(color='blue')
    ))
    scatter.add_trace(go.Scatter(
        x=list(range(len(y_predicted))),
        y=y_predicted,
        mode='markers',
        name='Predicted',
        marker=dict(color='orange')
    ))
    scatter.update_layout(title='Scatter Plot: Actual vs Predicted Prices')

    # Line Chart
    line = go.Figure()
    line.add_trace(go.Scatter(
        y=y_actual,
        mode='lines',
        name='Actual',
        line=dict(color='blue')
    ))
    line.add_trace(go.Scatter(
        y=y_predicted,
        mode='lines',
        name='Predicted',
        line=dict(color='orange')
    ))
    line.update_layout(title='Line Chart: Actual vs Predicted')

    # Box Plot
    box = go.Figure()
    box.add_trace(go.Box(y=y_actual, name='Actual', marker_color='blue'))
    box.add_trace(go.Box(y=y_predicted, name='Predicted', marker_color='orange'))
    box.update_layout(title='Box Plot: Actual vs Predicted Prices')

    # Histogram
    hist = go.Figure()
    hist.add_trace(go.Histogram(x=y_actual, name='Actual', marker_color='blue', opacity=0.6))
    hist.add_trace(go.Histogram(x=y_predicted, name='Predicted', marker_color='orange', opacity=0.6))
    hist.update_layout(title='Histogram: Actual vs Predicted Prices', barmode='overlay')

    return "".join([
        scatter.to_html(full_html=False, include_plotlyjs='cdn'),
        line.to_html(full_html=False, include_plotlyjs=False),
        box.to_html(full_html=False, include_plotlyjs=False),
        hist.to_html(full_html=False, include_plotlyjs=False)
    ])
