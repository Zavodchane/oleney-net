import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from dash import dash
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import plotly.express as px
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

Base = declarative_base()

class Image(Base):
    __tablename__ = "image"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    make = Column(String, nullable=True)
    model = Column(String, nullable=True)
    datetime = Column(String, nullable=True)
    class_ = Column(String, nullable=True)
    hash = Column(String, nullable=True)

# Подключаемся к базе данных
engine = create_engine('sqlite:///images.db')
Session = sessionmaker(bind=engine)
session = Session()

# Запрос данных из базы данных
query = session.query(Image).all()
session.close()

# Преобразуем данные в DataFrame
data = [(item.id, item.name, item.make, item.model, item.datetime, item.class_, item.hash) for item in query]
df = pd.DataFrame(data, columns=['ID', 'Name', 'Make', 'Model', 'Datetime', 'Class', 'Hash'])

fig_histogram_class = px.histogram(
    df,
    x='Class',
    color='Class',  # Разукрашиваем в зависимости от значения колонки 'Class'
    title='Распределение по классам',
    color_discrete_sequence=px.colors.qualitative.Plotly  # Используем набор цветов Plotly
)

fig_pie_class = px.pie(
    df,
    names='Class',
    title='Распределение по классам',
    color_discrete_sequence=px.colors.qualitative.Plotly  # Используем другой набор цветов для разнообразия
)

fig_pie_make = px.pie(
    df,
    names='Make',
    title='Распределение по маркам съемочного оборудывания',
    color_discrete_sequence=px.colors.qualitative.Set3  # Используем другой набор цветов для разнообразия
)

fig_histogram_make = px.histogram(
    df,
    x='Datetime',
    color='Datetime',  # Разукрашиваем в зависимости от значения колонки 'Class'
    title='Распределение по маркам съемочного оборудывания',
    color_discrete_sequence=px.colors.qualitative.Plotly  # Используем набор цветов Plotly
)

px.line()

# Создаем приложение Dash
app = dash.Dash()

app.layout = html.Div(children=[
    html.H1(children='Статистика по фотографиям'),

    html.Div(style={'display': 'flex', 'justifyContent': 'space-around'}, children=[
        dcc.Graph(
            id='histogram',
            figure=fig_histogram_class,
            style={'flex': '1', 'marginRight': '10px'}
        ),
        dcc.Graph(
            id='pie-chart',
            figure=fig_pie_class,
            style={'flex': '1', 'marginLeft': '10px'}
        )
    ]),
    html.Div(style={'display': 'flex', 'justifyContent': 'space-around'}, children=[
        dcc.Graph(
            id='histogram_1',
            figure=fig_histogram_make,
            style={'flex': '1', 'marginRight': '10px'}
        ),
        dcc.Graph(
            id='pie-chart_1',
            figure=fig_pie_make,
            style={'flex': '1', 'marginLeft': '10px'}
        )
    ])
])