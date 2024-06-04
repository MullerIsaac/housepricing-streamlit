import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Interface do usuário
st.set_page_config(page_title="Boston House Price Predictor", page_icon="🏠")

st.title("🏠 Preditor de Preços de casas")
st.write("### Insira as características da região para prever o preço de uma casa")

# Carregar os dados
@st.cache_data
def load_data():
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = pd.read_csv('boston_house.csv')
    return data

data = load_data()

# Separar as features e o alvo
X = data.drop('MEDV', axis=1)
y = data['MEDV']

# Pipeline de pré-processamento
numeric_features = X.columns
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = Pipeline(steps=[
    ('num', numeric_transformer)
])

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model.fit(X_train, y_train)

# Fazer predições no conjunto de teste
y_pred = model.predict(X_test)

# Calcular métricas de avaliação
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Sidebar: inputs do usuário
st.sidebar.header("Características das Casas")
num_columns = 2  # Ajuste o número de colunas conforme necessário
input_columns = st.sidebar.columns(num_columns)

features = {}
for i, col in enumerate(X.columns):
    current_column = input_columns[i % num_columns]
    if col == 'CHAS':
        features[col] = current_column.selectbox(col, [0, 1])
    elif col == 'RAD':
        features[col] = current_column.slider(col, 0, 10, step=1)
    else:
        features[col] = current_column.number_input(col, value=float(X[col].mean()))

# Converter inputs para DataFrame
input_data = pd.DataFrame([features])

# Prever o preço
if st.sidebar.button("Prever"):
    prediction = model.predict(input_data)
    st.sidebar.success(f"Preço previsto: ${prediction[0]*1000:,.2f}")  # Multiplicar por 1000 para converter de milhares para dólares

# Área principal: sumarização dos dados de aprendizado de máquina
st.write("### Sumarização do Modelo de Aprendizado de Máquina")
st.write("#### Métricas de Avaliação")
st.write(f"- Mean Squared Error (MSE): {mse:.2f}")
st.write(f"- R-squared (R2): {r2:.2f}")

# Exibir um sumário dos dados
st.write("### Resumo dos dados utilizados:")
st.write(data.head())

# Seção final: explicação do dataset
st.write("### Sobre o Dataset Boston Housing")
st.markdown("""
O dataset **Boston Housing** é amplamente utilizado para testar algoritmos de regressão. Ele contém 506 amostras e 13 features, além da variável alvo que representa o valor mediano das casas ocupadas pelos proprietários em milhares de dólares.

#### Features
- **CRIM**: Taxa de criminalidade per capita por cidade.
- **ZN**: Proporção de terrenos residenciais zoneados para lotes acima de 25.000 pés quadrados.
- **INDUS**: Proporção de acres de negócios não varejistas por cidade.
- **CHAS**: Variável dummy Charles River (1 se o terreno faz fronteira com o rio; 0 caso contrário).
- **NOX**: Concentração de óxidos nítricos (partes por 10 milhões).
- **RM**: Número médio de quartos por habitação.
- **AGE**: Proporção de unidades ocupadas pelos proprietários construídas antes de 1940.
- **DIS**: Distâncias ponderadas até cinco centros de emprego de Boston.
- **RAD**: Índice de acessibilidade às rodovias radiais.
- **TAX**: Taxa de imposto sobre a propriedade de valor total por $10.000.
- **PTRATIO**: Razão aluno-professor por cidade.
- **B**: 1000(Bk - 0.63)^2 onde Bk é a proporção de pessoas de ascendência afro-americana por cidade.
- **LSTAT**: Porcentagem da população de status socioeconômico mais baixo.
- **MEDV**: Valor mediano das casas ocupadas pelos proprietários (em milhares de dólares).

#### Fonte
O dataset foi retirado do repositório de aprendizado de máquina UCI e pode ser acessado através da biblioteca `sklearn` no Python.
""")
