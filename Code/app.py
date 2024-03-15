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
            'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width
    }
    return pd.DataFrame(itens, index=[0])

def main():
    df = load_iris()
    X = df.data
    y = df.target
    
    st.title("Classificador RandomForest")
    st.write("Base de dados Iris")
    
    st.sidebar.header("Parametros")
    
    st.subheader("Parâmetros")
    
    df_sb = setItensSideBar()
    st.write(df_sb)
    
    rfc = RandomForestClassifier()
    rfc.fit(X, y)
    
    previsao = rfc.predict(df_sb)
    previsao_proba = rfc.predict_proba(df_sb)
    
    st.subheader("Categoria predita")
    if previsao == 0:
        st.write("Setosa")
    elif previsao == 1:
        st.write("Versicolor")
    elif previsao == 2:
        st.write("Virginica")
    
    st.subheader("Probabilidade para cada classe")
    fig, ax = plt.subplots()
    previsao_probas = previsao_proba[0]
    ax.pie(previsao_probas, labels=['Setosa', 'Versicolor', 'Virginica'], autopct='%1.1f%%')
    st.pyplot(fig)

if __name__ == "__main__":
    main()
