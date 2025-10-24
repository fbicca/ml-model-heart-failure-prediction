# 💖 Heart Failure Predictor (Preditor de Insuficiência Cardíaca)

Este pacote oferece uma ferramenta simples para prever o risco de insuficiência cardíaca com base em dados clínicos, utilizando um modelo pré-treinado de Machine Learning.

O modelo baseia-se em 11 características clínicas e foi treinado para tarefas de **Classificação** (Doença: Sim/Não).

## 🚀 Instalação

Você pode instalar o pacote via `pip`:

```bash
pip install seu_nome_do_pacote
```

_(Nota: Substitua `seu_nome_do_pacote` pelo nome real que você dará no PyPI)_

## 💡 Como Usar

Para fazer uma previsão, você precisará carregar o modelo (`.pkl`) e o `StandardScaler` que foram salvos durante o treinamento, além de garantir que seus novos dados de entrada sejam pré-processados corretamente.

### 1\. Preparação dos Dados de Entrada

O modelo espera que as 11 características de entrada estejam no formato exato que foram usadas no treinamento (incluindo a codificação One-Hot para variáveis categóricas).

Assumindo que você tem um novo paciente com os dados abaixo, você deve formatá-los em um DataFrame.

**Exemplo de um novo input (assumindo a estrutura original do CSV):**

| Característica     | Valor  |
| :----------------- | :----- |
| **Age** (Idade)    | 45     |
| **Sex** (Gênero)   | M      |
| **ChestPainType**  | ASY    |
| **RestingBP**      | 120    |
| **Cholesterol**    | 240    |
| **FastingBS**      | 0      |
| **RestingECG**     | Normal |
| **MaxHR**          | 150    |
| **ExerciseAngina** | N      |
| **Oldpeak**        | 1.5    |
| **ST_Slope**       | Flat   |

### 2\. Código de Previsão

Use o seguinte script para carregar o modelo e o scaler, aplicar as transformações necessárias e obter a previsão:

```python
import joblib
import pandas as pd
# Importante: O nome das colunas precisa ser igual ao do seu dataset original!
from seu_nome_do_pacote.utils import load_model, load_scaler # Se você estruturar seu pacote assim

# --- ⚠️ CUIDADO: Substitua pelo seu novo dado de paciente ---
novo_paciente = pd.DataFrame([{
    'Age': 45,
    'Sex': 'M',
    'ChestPainType': 'ASY',
    'RestingBP': 120,
    'Cholesterol': 240,
    'FastingBS': 0,
    'RestingECG': 'Normal',
    'MaxHR': 150,
    'ExerciseAngina': 'N',
    'Oldpeak': 1.5,
    'ST_Slope': 'Flat'
}])
# ----------------------------------------------------------------

# 1. Carregar o modelo e o scaler salvos (Assuma que estão no diretório)
# Se você tiver empacotado os arquivos .pkl, carregue-os de dentro do seu pacote.
model = joblib.load('modelo_insuficiencia_cardiaca.pkl')
scaler = joblib.load('scaler_dados.pkl')

# 2. Codificação One-Hot (Muito importante!)
# Crie as colunas dummy para o novo dado, garantindo que TODAS as colunas do treino existam
# e que as categorias corretas sejam incluídas.
# NOTA: Esta é a parte mais complexa. É ideal criar uma lista de colunas de treino (X_train.columns)
# para garantir que o formato seja idêntico.
df_encoded_cols = model.feature_names_in_ # Obtém as colunas esperadas pelo modelo
novo_paciente_encoded = pd.get_dummies(novo_paciente, drop_first=True)

# Adicionar colunas faltantes e reordenar para corresponder ao treino
missing_cols = set(df_encoded_cols) - set(novo_paciente_encoded.columns)
for c in missing_cols:
    novo_paciente_encoded[c] = 0
novo_paciente_aligned = novo_paciente_encoded[df_encoded_cols]

# 3. Escalonamento
novo_paciente_scaled = scaler.transform(novo_paciente_aligned)

# 4. Previsão
pred = model.predict(novo_paciente_scaled)
prob = model.predict_proba(novo_paciente_scaled)[:, 1] # Probabilidade de ser a classe positiva (1)

if pred[0] == 1:
    print(f"\nResultado da Previsão: ALTO RISCO de Insuficiência Cardíaca.")
else:
    print(f"\nResultado da Previsão: BAIXO RISCO de Insuficiência Cardíaca.")

print(f"Probabilidade de Alto Risco: {prob[0]*100:.2f}%")
```

## 🛠️ Detalhes Técnicos do Treinamento

- **Modelo:** Regressão Logística (`sklearn.linear_model.LogisticRegression`)
- **Target (Variável Alvo):** `HeartDisease` (0 ou 1)
- **Pré-processamento:** `StandardScaler` (escalonamento) e `pd.get_dummies` (codificação One-Hot).
- **Métrica Foco:** **Recall** (Sensibilidade) para minimizar Falsos Negativos (deixar de diagnosticar a doença em um paciente que realmente a tem).

## 📄 Estrutura do Pacote

Para que o código de previsão funcione, seu pacote deve incluir os arquivos persistidos:

```
seu_nome_do_pacote/
├── __init__.py
├── modelo_insuficiencia_cardiaca.pkl  <-- Modelo
├── scaler_dados.pkl                  <-- Scaler
└── predictor.py                      <-- Script de predição (opcional)
```
