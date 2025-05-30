# **Descrição**: Este notebook apresenta um exemplo de uma rede neural profunda com mais de uma camada para um problema de classificação.


# # **Saúde Fetal**

# As Cardiotocografias (CTGs) são opções simples e de baixo custo para avaliar a saúde fetal, permitindo que os profissionais de saúde atuem na prevenção da mortalidade infantil e materna. O próprio equipamento funciona enviando pulsos de ultrassom e lendo sua resposta, lançando luz sobre a frequência cardíaca fetal (FCF), movimentos fetais, contrações uterinas e muito mais.

# Este conjunto de dados contém 2126 registros de características extraídas de exames de Cardiotocografias, que foram então classificados por três obstetras especialistas em 3 classes:

# - Normal
# - Suspeito
# - Patológico

# **O que faz:** Classificar a condição de saúde do feto com base em dados clínicos.
# 🔍 Ele responde à pergunta:
# Com base nos sinais captados no exame (cardiotocografia), qual a probabilidade de o feto estar em uma condição:

# Normal-Suspeita-Patológica

# O que o modelo recebe de entrada (X)?
# Dados como:

# *   Frequência cardíaca fetal média
# *   Número de acelerações e desacelerações
# *   Movimentos fetais
# *   Contrações uterinas
# *   Contrações uterinas
# * Contrações uterinas
# * Variação entre batimentos cardíacos
# * E outros indicadores extraídos da cardiotocografia

# O que ele retorna?
# Uma probabilidade para cada classe. Exemplo: [0.80, 0.15, 0.05]
# Interpretação:
# * 80% de chance de ser Normal
# * 15% de ser Suspeito
# * 5% de ser Patológico

#A partir daqui, colocar no saude_fetal_ml.py
# 1. Importações
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import mlflow
import mlflow.tensorflow

# 2. Carregar dados
url = "https://raw.githubusercontent.com/renansantosmendes/lectures-cdas-2023/master/fetal_health_reduced.csv"
df = pd.read_csv(url)

# 3. Separar dados e rótulos
X = df.drop("fetal_health", axis=1)
y = df["fetal_health"] - 1

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Criar modelo de rede neural
def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')  # 3 classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 6. Iniciar experimento com MLflow
mlflow.tensorflow.autolog()
with mlflow.start_run():
    model = create_model()
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)

    # 7. Avaliação
    preds = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    mlflow.log_metric("accuracy", acc)