# 📊 EDA Churn Analysis - Guia de Estudos Completo

## 🎯 Visão Geral do Projeto

Este projeto faz parte do curso **"Machine Learning em Inteligência Artificial"** da Rocketseat e tem como objetivo realizar uma **Análise Exploratória de Dados (EDA)** para entender os fatores que influenciam o **Churn** (abandono de clientes) em uma empresa de telecomunicações.

### 📋 O que é Churn?

**Churn** é a taxa de abandono de clientes - uma métrica crucial para empresas que precisam:

- Identificar clientes com maior probabilidade de cancelar
- Entender os motivos do abandono
- Desenvolver estratégias de retenção
- Otimizar recursos de marketing e vendas

---

## 📁 Estrutura dos Dados

### Datasets Utilizados

```
datasets/
├── churn_customers.csv   # Dados demográficos dos clientes
├── churn_services.csv    # Serviços contratados
└── churn_contracts.csv   # Informações contratuais e financeiras
```

### 📊 Descrição dos Dados

#### 🧑‍🤝‍🧑 churn_customers.csv

- **customerID**: Identificador único do cliente
- **gender**: Gênero (Male/Female)
- **SeniorCitizen**: Cliente idoso (0=Não, 1=Sim)
- **Partner**: Tem parceiro (Yes/No)
- **Dependents**: Tem dependentes (Yes/No)

#### 📱 churn_services.csv

- **customerID**: Identificador único do cliente
- **PhoneService**: Serviço de telefone
- **MultipleLines**: Múltiplas linhas
- **InternetService**: Tipo de internet (DSL/Fiber optic/No)
- **OnlineSecurity**: Segurança online
- **OnlineBackup**: Backup online
- **DeviceProtection**: Proteção de dispositivos
- **TechSupport**: Suporte técnico
- **StreamingTV**: TV streaming
- **StreamingMovies**: Filmes streaming

#### 💰 churn_contracts.csv

- **customerID**: Identificador único do cliente
- **Contract**: Tipo de contrato (Month-to-month/One year/Two year)
- **PaperlessBilling**: Faturamento sem papel
- **PaymentMethod**: Método de pagamento
- **MonthlyCharges**: Cobrança mensal
- **TotalCharges**: Total cobrado
- **tenure**: Tempo de contrato (em meses)
- **Churn**: Se o cliente abandonou (Yes/No) - **VARIÁVEL TARGET**

---

## 🔄 Metodologia EDA Aplicada

### 1️⃣ **Preparação dos Dados**

#### Importação de Bibliotecas

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
```

#### Carregamento dos DataFrames

```python
df_customers = pd.read_csv('./datasets/churn_customers.csv')
df_services = pd.read_csv('./datasets/churn_services.csv')
df_contracts = pd.read_csv('./datasets/churn_contracts.csv')
```

#### Unificação dos Dados

**Estratégia de Merge:**

1. `df_customers` + `df_services` → usando `customerID`
2. Resultado + `df_contracts` → usando `customerID` (left) e `customerID` (right)

```python
df_churn = df_customers.merge(df_services, on='IDCliente', how='inner')\
    .merge(df_contracts, left_on=['IDCliente'], right_on=['customerID'])
```

### 2️⃣ **Transformações Realizadas**

#### Conversão de Tipos de Dados

```python
# TotalCharges de string para float
df_contracts.TotalCharges = pd.to_numeric(df_contracts.TotalCharges, errors='coerce')
```

#### Renomeação de Colunas

```python
# Método 1: Usando dicionário
df_customers.rename(columns={'SeniorCitizen': 'Above65Yo'}, inplace=True)

# Método 2: Renomeando todas as colunas
df_customers.columns = ['IDCliente', 'Genero', 'Mais65Anos', 'TemParceiro', 'TemDependentes']
```

### 3️⃣ **Tratamento de Dados Ausentes**

#### Detecção

```python
# Detectar valores ausentes
df_churn.isna().sum()
df_churn.TotalCharges.isna().sum()  # 11 valores ausentes encontrados
```

#### Estratégias de Tratamento

```python
# Opção 1: Remover linhas com valores ausentes
df_churn.dropna(axis=0)

# Opção 2: Remover colunas com valores ausentes
df_churn.dropna(axis=1)

# Opção 3: Imputação com média
media_TotalCharges = df_churn.TotalCharges.mean()
df_churn.fillna(value={'TotalCharges': media_TotalCharges})

# Opção 4: Imputação com valor específico
df_churn.fillna(value={'TotalCharges': 0})
```

---

## 📈 Análises Realizadas

### 🔍 **Análise Univariada**

#### Análise da Variável Target (Churn)

```python
# Distribuição absoluta
df_churn.Churn.value_counts()
# Resultado: ~73% ativos, ~27% churn

# Distribuição percentual
df_churn.Churn.value_counts(normalize=True)

# Visualização
ax = df_churn.Churn.value_counts().plot.bar()
ax.bar_label(ax.containers[0])
```

#### Análise de Variáveis Categóricas

```python
# Tipos de contrato
df_churn.Contract.unique()
df_churn.Contract.value_counts().plot.bar()
```

#### Análise de Variáveis Numéricas

```python
# Histograma do tempo de contrato
df_churn.tenure.plot.hist()

# Medidas de posição
df_churn.tenure.mean()    # Média
df_churn.tenure.median()  # Mediana
df_churn.tenure.mode()    # Moda

# Medidas de dispersão
df_churn.tenure.std()     # Desvio padrão
cv = df_churn.tenure.std() / df_churn.tenure.mean() * 100  # Coeficiente de variação
```

#### Filtros e Agrupamentos

```python
# Clientes com 1 mês de contrato
len(df_churn[(df_churn.tenure == 1)])

# Clientes entre 1 e 6 meses
len(df_churn[(df_churn.tenure >= 1) & (df_churn.tenure <= 6)])

# Agrupamento por tempo de contrato
df_churn.groupby(['tenure'])['tenure'].count().sort_values(ascending=False)
```

### 🔍 **Análise Bivariada**

#### Tabelas de Contingência

```python
# Churn vs Tipo de Contrato (quantidade)
pd.crosstab(df_churn.Churn, df_churn.Contract, margins=True, margins_name='Total')

# Churn vs Tipo de Contrato (proporção)
pd.crosstab(df_churn.Churn, df_churn.Contract, normalize='index', margins=True, margins_name='Total')
```

#### Testes de Hipóteses (Chi-Square)

**Conceito:**

- **H0 (Hipótese Nula)**: As variáveis são independentes
- **H1 (Hipótese Alternativa)**: As variáveis não são independentes
- **Critério**: Se p-value ≤ 0.05, rejeitamos H0

**Implementação:**

```python
# 1. Criar tabela de contingência
df_crosstab = pd.crosstab(df_churn.Churn, df_churn.Contract)

# 2. Aplicar teste Chi-Square
from scipy.stats import chi2_contingency
chi_scores = chi2_contingency(df_crosstab)

# 3. Extrair resultados
scores = pd.Series(chi_scores[0])     # Qui-quadrado
p_values = pd.Series(chi_scores[1])   # P-value

# 4. Criar DataFrame de resultados
df_results = pd.DataFrame({
    'Qui2': scores,
    'P-Value': p_values
})
```

#### Correlação entre Variáveis Numéricas

```python
# Correlação de Pearson
df_churn.tenure.corr(df_churn.TotalCharges)

# Correlação de Spearman
df_churn.tenure.corr(df_churn.TotalCharges, method='spearman')

# Gráfico de dispersão
df_churn.plot.scatter(x='tenure', y='TotalCharges')
```

---

## 🎯 Hipóteses Testadas e Resultados

### ✅ **Hipótese 1: Contrato Mensal → Maior Churn**

**Método:** Teste Chi-Square  
**Resultado:** P-value < 0.05 ✅  
**Conclusão:** 88% dos clientes que abandonaram tinham contrato mensal  
**Status:** CONFIRMADA - Forte correlação

### ✅ **Hipótese 2: Tempo < 6 meses → Maior Churn**

**Método:** Criação de variável categórica + Chi-Square

```python
df_churn['TempoMenor6Meses'] = np.where(df_churn.tenure < 6, 'Yes', 'No')
```

**Resultado:** P-value < 0.05 ✅  
**Conclusão:** Clientes novos são mais propensos ao churn  
**Status:** CONFIRMADA - Correlação moderada

### ✅ **Hipótese 3: Idade > 65 anos → Maior Churn**

**Método:** Criação de variável categórica + Chi-Square

```python
df_churn['FaixaEtaria'] = np.where(df_churn.Mais65Anos == 0, 'Abaixo65', 'Acima65')
```

**Resultado:** P-value < 0.05 ✅  
**Conclusão:** Clientes idosos têm maior tendência ao churn  
**Status:** CONFIRMADA - Correlação significativa

### ✅ **Hipótese 4: Tempo de Contrato vs Valor Pago**

**Método:** Correlação de Pearson e Spearman  
**Resultado:** Correlação ~0.8  
**Conclusão:** Forte correlação positiva  
**Status:** CONFIRMADA

---

## 📊 Principais Insights Encontrados

### 🔴 **Fatores de Alto Risco de Churn:**

1. **Contrato Mensal** - 88% dos churns
2. **Clientes Novos** - < 6 meses de contrato
3. **Idade Avançada** - > 65 anos
4. **Baixo Valor Total** - Correlação inversa com tenure

### 🟢 **Fatores de Retenção:**

1. **Contratos Anuais/Bianuais**
2. **Clientes Antigos** - > 6 meses
3. **Maior Valor Total Pago**

---

## 🛠️ Ferramentas e Técnicas Utilizadas

### 📚 **Bibliotecas Python**

- **pandas**: Manipulação de dados
- **numpy**: Operações numéricas e condicionais
- **matplotlib**: Visualizações básicas
- **scipy.stats**: Testes estatísticos

### 📊 **Técnicas Estatísticas**

- **Medidas de Tendência Central**: Média, mediana, moda
- **Medidas de Dispersão**: Desvio padrão, coeficiente de variação
- **Teste Chi-Square**: Independência entre variáveis categóricas
- **Correlação de Pearson/Spearman**: Relação entre variáveis numéricas

### 🔧 **Técnicas de Manipulação de Dados**

- **Merge de DataFrames**: Inner join com chaves diferentes
- **Tratamento de Missing Values**: Detecção e imputação
- **Feature Engineering**: Criação de variáveis categóricas
- **Transformação de Tipos**: String para numérico

---

## 📖 Como Criar Novas Hipóteses

### 🔍 **Passo a Passo para Análise Bivariada**

#### **Para Variáveis Categóricas vs Churn:**

1. **Formular Hipótese**

   ```
   Exemplo: "Clientes que usam método de pagamento X são mais propensos ao churn"
   ```

2. **Criar Tabela de Contingência**

   ```python
   pd.crosstab(df_churn.Churn, df_churn.PaymentMethod, margins=True, margins_name='Total')
   ```

3. **Aplicar Teste Chi-Square**

   ```python
   df_crosstab = pd.crosstab(df_churn.Churn, df_churn.PaymentMethod)
   chi_scores = chi2_contingency(df_crosstab)
   ```

4. **Interpretar Resultados**
   - P-value ≤ 0.05: Hipótese confirmada
   - P-value > 0.05: Não há evidência estatística

#### **Para Variáveis Numéricas vs Churn:**

1. **Transformar em Categórica**

   ```python
   # Exemplo: MonthlyCharges alto vs baixo
   df_churn['ChargesAlto'] = np.where(df_churn.MonthlyCharges > df_churn.MonthlyCharges.median(), 'Alto', 'Baixo')
   ```

2. **Seguir mesmo processo de variáveis categóricas**

#### **Para Correlação entre Variáveis Numéricas:**

```python
# Correlação
df_churn.variavel1.corr(df_churn.variavel2)

# Visualização
df_churn.plot.scatter(x='variavel1', y='variavel2')
```

---

## 🎯 Próximos Passos Sugeridos

### 📈 **Análises Complementares**

1. **Detecção de Outliers** (Boxplots, IQR)
2. **Análise Multivariada** (3+ variáveis)
3. **Segmentação de Clientes** (Clustering)
4. **Feature Engineering Avançada**

### 🤖 **Preparação para Machine Learning**

1. **Codificação de Variáveis Categóricas** (One-hot encoding)
2. **Normalização/Padronização** de variáveis numéricas
3. **Divisão Train/Test**
4. **Seleção de Features**

### 📊 **Visualizações Avançadas**

1. **Heatmaps de Correlação**
2. **Gráficos de Distribuição Avançados**
3. **Dashboard Interativo**

---

## 📚 Referências do Curso Rocketseat

### 🎓 **Módulos Relacionados**

- **Estatística para Devs**: Medidas, dispersão, correlação e gráficos
- **Análise Exploratória de Dados**: EDA com Pandas (28 aulas, 4h24min)
- **Próximos**: Aprendizado de Máquina, Algoritmos Supervisionados

### 📖 **Conceitos Aplicados**

- ✅ Coleta e preparação de dados
- ✅ Tratamento de dados ausentes
- ✅ Formulação de hipóteses
- ✅ Análise univariada e bivariada
- 🔄 Detecção de outliers (em andamento)
- 🔄 Automatizações (próximo)

---

## 💡 Dicas de Estudo

### 🔥 **Boas Práticas Identificadas no Projeto**

1. **Documentação**: Células markdown explicando cada etapa
2. **Validação Estatística**: Uso de testes de hipóteses
3. **Visualização**: Gráficos com labels informativos
4. **Metodologia**: Abordagem científica para EDA

### 📝 **Para Novos Estudos**

1. **Sempre formule hipóteses ANTES de analisar**
2. **Use testes estatísticos para validar descobertas**
3. **Documente insights e conclusões**
4. **Visualize dados de forma clara e informativa**

---

## 🏆 Status do Projeto

- ✅ **Preparação de Dados**: Completo
- ✅ **Análise Univariada**: Completo
- ✅ **Análise Bivariada**: Completo
- ✅ **Testes de Hipóteses**: 4 hipóteses testadas
- 🔄 **Detecção de Outliers**: Em andamento (célula 139)
- ⏳ **Próximas análises**: Aguardando continuação do curso

**Total de Células**: 139 (sendo 12 markdown e 127 código)  
**Execuções**: Todas as células executadas com sucesso  
**Datasets**: 3 arquivos CSV unificados em 1 DataFrame principal

---

_Este README é um documento vivo que deve ser atualizado conforme o progresso no curso e no projeto!_ 🚀
