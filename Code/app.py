import streamlit as st 
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt



def setItensSideBar():
    sepal_length = st.sidebar.slider("Sepal length", 4.3, 7.9, 5.0)
    sepal_width = st.sidebar.slider("Sepal width", 2.0, 4.4, 3.2)
    petal_length = st.sidebar.slider("Petal length", 1.0, 6.9, 2.5)
    petal_width = st.sidebar.slider("Petal width", 0.1, 2.5, 0.5)

    itens = {
            'Sepal length': sepal_length,
            'Sepal width': sepal_width,
            'Petal length': petal_length,
            'Petal width': petal_width
    }
    return pd.DataFrame(itens, index=[0])

def main():
    #Caregamento da base de dados
    df = load_iris()
    #Separação da base de dados
    X = df.data
    y = df.target
    
    #configuração do titulo e subtitulo
    st.title("Classificador RandomForest")
    st.write("Base de dados Iris")
    
    #configuração do titulo do menu lateral
    st.sidebar.header("Configure os parêmetros para fazer a predição")
    
    #titulo do data frame com os valores selecionado pelo usuário
    #este valor e usado na predição
    st.subheader("Parâmetros usados na predição")
    
    #Função para montar os slider de seleção de valores
    df_sb = setItensSideBar()
    st.write(df_sb)
    
    rfc = RandomForestClassifier()
    rfc.fit(X, y)
    #Fazendo as previsões
    previsao = rfc.predict(df_sb)
    previsao_proba = rfc.predict_proba(df_sb)
    
    #Plotando na tela a categoria predita
    st.subheader("Categoria predita")
    if previsao == 0:
        st.write("Setosa")
    elif previsao == 1:
        st.write("Versicolor")
    elif previsao == 2:
        st.write("Virginica")
    
    #criando o grafico de pizza para as problabilidades
    st.subheader("Probabilidade para cada classe")
    fig, ax = plt.subplots()
    previsao_probas = previsao_proba[0]
    ax.pie(previsao_probas, labels=['Setosa', 'Versicolor', 'Virginica'], autopct='%1.1f%%')
    st.pyplot(fig)
     
   
    # Exibir o código do aplicativo Streamlit
    st.subheader("Código do aplicativo")
    with open("app.py", "r") as file:
        code = file.read()
    st.code(code)
    
if __name__ == "__main__":
    main()


