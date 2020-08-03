# Correlacionando dados de instrumentos

**A ideia principal é estabelecer quais são os índices de correlação entre instrumentos instalados em uma represa com relação à régua de nível da represa (variável target).**
**Dessa forma, é possível priorizar o monitoramento desses instrumentos e estabelecer níveis e intensidades de variações nesses instrumentos em função de variações nas medidas da régua.**


### Roteiro Básico:

- Carga de arquivos .csv com informações de instrumentos e as medições realizadas por esses instrumentos;
- Tratamento dos dados;
- Definição de índice de correlação linear (PEARSON) de cada instrumento com o target;
- Apresentação de resultados em gráficos e tabelas.

**Importante:** Os dados foram anonimizados.

### Follow me at Linkdin

https://www.linkedin.com/in/robertodelamora/



# Inicialização do ambiente


```python
# Essentials
import pandas as pd
import numpy as np

import datetime
from datetime import datetime as dt
import gc
import math
import warnings, time
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
%matplotlib inline

# Plots
import matplotlib.pyplot as plt
import seaborn as sns

# Stats
from scipy import stats

# Estebelece limites para visualização no notebook
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',500)

# Limita a 3 casas decimais a apresentação das variaveis tipo float
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) 
```

# Funções recorrentes


```python
# Apresenta 3 tipos de gráficos para cada variável - Scatter, Histograma e Boxplot
def graficos(df, var1, var2):

    # cria espaço para 3 gráficos em cada linha 
    f, axes = plt.subplots(nrows=1, ncols=3)
    plt.rcParams['figure.figsize'] = [20, 5]
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    # Gráfico 1 - Scatter plot comparando variável com target 
    axes[0].scatter(x = df[var2], y = df[var1], alpha=0.8)
    axes[0].set_title(var2 + ' x ' + var1)

    # Gráfico 2 - Histograma 
    df[var2].hist(bins = 30, ax = axes[1])
    axes[1].set_title('Distribution of ' + var2, fontsize=12)
    
    # Gráfico 3 - Boxplot 
    df.boxplot(column = var2, ax = axes[2],fontsize=12)
    
    plt.show()
```

# FASE 1 - Preparação de datasets de Instrumentos e Medições


**Faz o tratamento dos dados dos 2 datasets e cria nova tabela para uso nos estudos**


## Etapa 1.1 - Dados de Instrumentos

**Realiza carga da tabela de instrumentos**

São utilizadas somente as variáveis de identificação do instrumento e valor de cota. Os dados de cota serão adotados nos casos em que não houver indicação de valores nas medições.

### Carga dos dados


```python
cols = ['cod_inst', 'cota_fundo']
df_inst = pd.read_csv('instrumentos.csv', sep=';', encoding='latin1', decimal=',', usecols=cols)
```


```python
df_inst.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cod_inst</th>
      <th>cota_fundo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nvl_agua_01</td>
      <td>925.762</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nvl_agua_02</td>
      <td>918.023</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nvl_agua_03</td>
      <td>938.437</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nvl_agua_04</td>
      <td>926.793</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nvl_agua_05</td>
      <td>918.652</td>
    </tr>
  </tbody>
</table>
</div>



### Tratamento básico


```python
# Preenche campos vazios na variavel cota_fundo
df_inst['cota_fundo'] = df_inst['cota_fundo'].fillna(0)
```


```python
# Valida se não existem campos com NaN
df_inst.isnull().sum()
```




    cod_inst      0
    cota_fundo    0
    dtype: int64




```python
# Ordena registros por código de instrumento
df_inst = df_inst.sort_values(['cod_inst'])
df_inst.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cod_inst</th>
      <th>cota_fundo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nvl_agua_01</td>
      <td>925.762</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nvl_agua_02</td>
      <td>918.023</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nvl_agua_03</td>
      <td>938.437</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nvl_agua_04</td>
      <td>926.793</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nvl_agua_05</td>
      <td>918.652</td>
    </tr>
  </tbody>
</table>
</div>




```python
# ========================================
# Gravando arquivo de referencia
df_inst.to_csv('df_inst.csv', sep=';', encoding='utf-8', decimal=',', index=False)
```


```python
del df_inst, cols
gc.collect();
```

## Etapa 1.2 - Dados de Medições

**Realiza carga da tabela com informações sobre as medições dos instrumentos ao longo do tempo.**

Em análise prévia, foram identificadas variáveis essenciais à análise e somente essas são carregadas.

Também são realizados filtros para eliminação de registros sem utilidade (sem dados) e preenchimento de campos sem valores de medição com os dados oriundos da tabela de instrumentos tratada anteriormente.

### Carga dos dados

Faz a carga somente das variáveis que interessam na análise. 


```python
cols = ['num_os', 'cod_inst', 'tipo_inst', 'situacao', 'condicao', 'data_med', 'valor', 'unidade']
df_med = pd.read_csv('medicoes.csv', sep=';', encoding='latin1', decimal=',', usecols=cols)
```


```python
print('Dimensões de df_medset:', df_med.shape)
df_med.head(3)
```

    Dimensões de df_medset: (5786, 8)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_os</th>
      <th>cod_inst</th>
      <th>tipo_inst</th>
      <th>situacao</th>
      <th>condicao</th>
      <th>data_med</th>
      <th>valor</th>
      <th>unidade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4708.000</td>
      <td>piez_01</td>
      <td>piezometro</td>
      <td>Não Realizada</td>
      <td>NaN</td>
      <td>10/02/2005 07:40</td>
      <td>nan</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4709.000</td>
      <td>piez_01</td>
      <td>piezometro</td>
      <td>Não Realizada</td>
      <td>NaN</td>
      <td>10/03/2005 07:40</td>
      <td>nan</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4710.000</td>
      <td>piez_01</td>
      <td>piezometro</td>
      <td>Não Realizada</td>
      <td>NaN</td>
      <td>11/04/2005 07:40</td>
      <td>nan</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_med.dtypes
```




    num_os       float64
    cod_inst      object
    tipo_inst     object
    situacao      object
    condicao      object
    data_med      object
    valor        float64
    unidade       object
    dtype: object



### Tratamento básico


```python
# Variável data_med foi carregada como string e com informações de data e hora.
# Remove informação de hh:mm, converte para datetime e ajusta formato para aaaa-mm-dd
df_med['data_med'] = df_med['data_med'].str.split(' ').str[0] 
df_med['data_med'] = pd.to_datetime(df_med['data_med'], format="%d/%m/%Y")
df_med.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_os</th>
      <th>cod_inst</th>
      <th>tipo_inst</th>
      <th>situacao</th>
      <th>condicao</th>
      <th>data_med</th>
      <th>valor</th>
      <th>unidade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4708.000</td>
      <td>piez_01</td>
      <td>piezometro</td>
      <td>Não Realizada</td>
      <td>NaN</td>
      <td>2005-02-10</td>
      <td>nan</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4709.000</td>
      <td>piez_01</td>
      <td>piezometro</td>
      <td>Não Realizada</td>
      <td>NaN</td>
      <td>2005-03-10</td>
      <td>nan</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4710.000</td>
      <td>piez_01</td>
      <td>piezometro</td>
      <td>Não Realizada</td>
      <td>NaN</td>
      <td>2005-04-11</td>
      <td>nan</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Removendo registros com variável 'situacao'= 'Não realizadas'. Não possuem dados para avaliar.
df_med.drop(df_med[df_med['situacao'] == 'Não Realizada'].index, axis = 0, inplace=True)

# Uma vez que situação agora só possui um status, não contribui mais para a análise 
df_med.drop(['situacao'], axis=1, inplace=True)
```


```python
# Preenchendo campo 'unidade' com tipo de medida de acordo com tipo_inst
# Piezometros, Medidor de Nivel de Água e Régua do Reservatório = m
# Medidor de vazão = LPS
for i in range(0, len(df_med)):
    if df_med.iloc[i, 2] == 'piezometro':
        df_med.iloc[i, 6] = 'm'
    elif df_med.iloc[i, 2] == 'medidor de nivel de agua':
        df_med.iloc[i, 6] = 'm'
    elif df_med.iloc[i, 2] == 'regua de reservatorio':
        df_med.iloc[i, 6] = 'm'
    else: 
        df_med.iloc[i, 6] = 'LPS'

# Apresentação para verificação
df_med.groupby(['tipo_inst', 'unidade'])['unidade'].count()
```




    tipo_inst                 unidade
    medidor de nivel de agua  m           703
    medidor de vazao          LPS         341
    piezometro                m          3607
    regua do reservatorio     LPS         465
    Name: unidade, dtype: int64




```python
# Remove registros com OS em branco
df_med.drop(df_med[df_med['num_os'].isna()].index, axis=0, inplace=True)
```


```python
# Completa campos faltantes
df_med['valor'] = df_med['valor'].fillna(0)
df_med['condicao'] = df_med['condicao'].fillna('Nao')
```


```python
print(df_med.dtypes)
print()
print(df_med.isnull().sum())
df_med.head(3)
```

    num_os              float64
    cod_inst             object
    tipo_inst            object
    condicao             object
    data_med     datetime64[ns]
    valor               float64
    unidade              object
    dtype: object
    
    num_os       0
    cod_inst     0
    tipo_inst    0
    condicao     0
    data_med     0
    valor        0
    unidade      0
    dtype: int64
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_os</th>
      <th>cod_inst</th>
      <th>tipo_inst</th>
      <th>condicao</th>
      <th>data_med</th>
      <th>valor</th>
      <th>unidade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>36</th>
      <td>457.000</td>
      <td>piez_01</td>
      <td>piezometro</td>
      <td>SECO</td>
      <td>2008-07-30</td>
      <td>0.000</td>
      <td>m</td>
    </tr>
    <tr>
      <th>37</th>
      <td>23938.000</td>
      <td>piez_01</td>
      <td>piezometro</td>
      <td>SECO</td>
      <td>2009-02-02</td>
      <td>0.000</td>
      <td>m</td>
    </tr>
    <tr>
      <th>38</th>
      <td>15591.000</td>
      <td>piez_01</td>
      <td>piezometro</td>
      <td>SECO</td>
      <td>2009-02-06</td>
      <td>0.000</td>
      <td>m</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Remove registros de OS que só possuem medições da regua
# Mesmo essa sendo a variável target, sozinha e sem outros instrumentos para comparação, ela não agrega conteúdo

# Obtem lista de números de OS
num_os = df_med['num_os'].unique()

# Cria loop para leitura e validação. 
# Se OS só contem 1 registro (Régua do Reservatório), então é removida
# Se OS não contém registro relativo à Régua do Reservatório também é removida
for os in num_os:
    df = df_med[df_med['num_os'] == os]
    if (len(df) == 1) or (len(df[df['tipo_inst']=='regua do reservatorio'].index.values) == 0):
        df_med.drop(df_med[df_med['num_os'] == os].index, axis=0, inplace=True)
```


```python
# ========================================
# Carregando dataset de instrumentos para preencher campos vazios com dados de referencia
df_inst = pd.read_csv('df_inst.csv', sep=';', decimal=',', encoding='utf-8')
```


```python
df_inst.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cod_inst</th>
      <th>cota_fundo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nvl_agua_01</td>
      <td>925.762</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nvl_agua_02</td>
      <td>918.023</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nvl_agua_03</td>
      <td>938.437</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nvl_agua_04</td>
      <td>926.793</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nvl_agua_05</td>
      <td>918.652</td>
    </tr>
  </tbody>
</table>
</div>




```python
for i in range(0,len(df_med)):
    if (df_med.iloc[i, 3] == 'SECO') & (df_med.iloc[i, 5] == 0):
        inst = df_med.iloc[i, 1]
        cota = 0
        for j in range(0, len(df_inst)):
            if df_inst.iloc[j, 0] == inst:
                cota = df_inst.iloc[j, 1]
        df_med.iloc[i, 5] = cota       
```


```python
df_med.isnull().sum()
```




    num_os       0
    cod_inst     0
    tipo_inst    0
    condicao     0
    data_med     0
    valor        0
    unidade      0
    dtype: int64




```python
# Análise visual para entender se medições dentro de cada OS são executadas no mesmo dia
pd.DataFrame(df_med.groupby(['num_os', 'data_med'])['tipo_inst'].count())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>tipo_inst</th>
    </tr>
    <tr>
      <th>num_os</th>
      <th>data_med</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">457.000</th>
      <th>2008-07-02</th>
      <td>4</td>
    </tr>
    <tr>
      <th>2008-07-30</th>
      <td>14</td>
    </tr>
    <tr>
      <th>4708.000</th>
      <th>2005-02-10</th>
      <td>9</td>
    </tr>
    <tr>
      <th>4709.000</th>
      <th>2005-03-10</th>
      <td>10</td>
    </tr>
    <tr>
      <th>4710.000</th>
      <th>2005-04-11</th>
      <td>12</td>
    </tr>
    <tr>
      <th>4711.000</th>
      <th>2005-05-06</th>
      <td>11</td>
    </tr>
    <tr>
      <th>4712.000</th>
      <th>2005-06-10</th>
      <td>9</td>
    </tr>
    <tr>
      <th>4713.000</th>
      <th>2005-07-25</th>
      <td>9</td>
    </tr>
    <tr>
      <th>4714.000</th>
      <th>2005-08-17</th>
      <td>9</td>
    </tr>
    <tr>
      <th>4715.000</th>
      <th>2005-09-05</th>
      <td>9</td>
    </tr>
    <tr>
      <th>4716.000</th>
      <th>2005-10-17</th>
      <td>9</td>
    </tr>
    <tr>
      <th>4717.000</th>
      <th>2005-11-08</th>
      <td>9</td>
    </tr>
    <tr>
      <th>4718.000</th>
      <th>2005-12-09</th>
      <td>6</td>
    </tr>
    <tr>
      <th>4719.000</th>
      <th>2006-01-09</th>
      <td>9</td>
    </tr>
    <tr>
      <th>4720.000</th>
      <th>2006-02-20</th>
      <td>9</td>
    </tr>
    <tr>
      <th>4721.000</th>
      <th>2006-03-13</th>
      <td>9</td>
    </tr>
    <tr>
      <th>4722.000</th>
      <th>2006-04-17</th>
      <td>9</td>
    </tr>
    <tr>
      <th>4723.000</th>
      <th>2006-05-12</th>
      <td>9</td>
    </tr>
    <tr>
      <th>4724.000</th>
      <th>2006-06-13</th>
      <td>10</td>
    </tr>
    <tr>
      <th>4725.000</th>
      <th>2006-07-17</th>
      <td>10</td>
    </tr>
    <tr>
      <th>4726.000</th>
      <th>2006-08-09</th>
      <td>10</td>
    </tr>
    <tr>
      <th>4727.000</th>
      <th>2006-09-11</th>
      <td>10</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">4728.000</th>
      <th>2006-10-02</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2006-10-06</th>
      <td>8</td>
    </tr>
    <tr>
      <th>4729.000</th>
      <th>2006-11-10</th>
      <td>9</td>
    </tr>
    <tr>
      <th>4730.000</th>
      <th>2006-12-06</th>
      <td>5</td>
    </tr>
    <tr>
      <th>4731.000</th>
      <th>2007-01-08</th>
      <td>5</td>
    </tr>
    <tr>
      <th>4732.000</th>
      <th>2007-02-09</th>
      <td>5</td>
    </tr>
    <tr>
      <th>4733.000</th>
      <th>2007-03-06</th>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">4737.000</th>
      <th>2007-07-04</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2007-07-05</th>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">4739.000</th>
      <th>2007-09-11</th>
      <td>4</td>
    </tr>
    <tr>
      <th>2007-09-14</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4740.000</th>
      <th>2007-10-08</th>
      <td>6</td>
    </tr>
    <tr>
      <th>4741.000</th>
      <th>2007-11-05</th>
      <td>6</td>
    </tr>
    <tr>
      <th>4742.000</th>
      <th>2007-12-04</th>
      <td>6</td>
    </tr>
    <tr>
      <th>4743.000</th>
      <th>2008-01-03</th>
      <td>6</td>
    </tr>
    <tr>
      <th>4744.000</th>
      <th>2008-02-07</th>
      <td>6</td>
    </tr>
    <tr>
      <th>4745.000</th>
      <th>2008-03-04</th>
      <td>6</td>
    </tr>
    <tr>
      <th>4746.000</th>
      <th>2008-04-02</th>
      <td>6</td>
    </tr>
    <tr>
      <th>4747.000</th>
      <th>2008-05-06</th>
      <td>6</td>
    </tr>
    <tr>
      <th>4748.000</th>
      <th>2008-06-02</th>
      <td>6</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">15591.000</th>
      <th>2009-02-03</th>
      <td>4</td>
    </tr>
    <tr>
      <th>2009-02-06</th>
      <td>15</td>
    </tr>
    <tr>
      <th>27166.000</th>
      <th>2009-08-12</th>
      <td>20</td>
    </tr>
    <tr>
      <th>28159.000</th>
      <th>2009-09-10</th>
      <td>20</td>
    </tr>
    <tr>
      <th>29051.000</th>
      <th>2009-10-08</th>
      <td>20</td>
    </tr>
    <tr>
      <th>29727.000</th>
      <th>2009-11-12</th>
      <td>20</td>
    </tr>
    <tr>
      <th>30495.000</th>
      <th>2009-12-10</th>
      <td>20</td>
    </tr>
    <tr>
      <th>31325.000</th>
      <th>2010-01-13</th>
      <td>20</td>
    </tr>
    <tr>
      <th>32660.000</th>
      <th>2010-02-01</th>
      <td>20</td>
    </tr>
    <tr>
      <th>33305.000</th>
      <th>2010-03-11</th>
      <td>20</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">34203.000</th>
      <th>2010-04-03</th>
      <td>3</td>
    </tr>
    <tr>
      <th>2010-04-12</th>
      <td>17</td>
    </tr>
    <tr>
      <th>35067.000</th>
      <th>2010-05-13</th>
      <td>7</td>
    </tr>
    <tr>
      <th>35939.000</th>
      <th>2010-06-15</th>
      <td>20</td>
    </tr>
    <tr>
      <th>36865.000</th>
      <th>2010-07-19</th>
      <td>20</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">41920.000</th>
      <th>2011-01-26</th>
      <td>20</td>
    </tr>
    <tr>
      <th>2011-01-31</th>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">42481.000</th>
      <th>2010-08-24</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2010-09-22</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2010-10-18</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2010-11-19</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2010-12-21</th>
      <td>1</td>
    </tr>
    <tr>
      <th>43083.000</th>
      <th>2011-02-24</th>
      <td>21</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">44527.000</th>
      <th>2011-04-26</th>
      <td>20</td>
    </tr>
    <tr>
      <th>2011-04-29</th>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">45392.000</th>
      <th>2011-05-06</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2011-05-23</th>
      <td>20</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">46332.000</th>
      <th>2011-06-20</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2011-06-21</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2011-06-22</th>
      <td>16</td>
    </tr>
    <tr>
      <th>2011-06-28</th>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">47330.000</th>
      <th>2011-07-01</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2011-07-21</th>
      <td>19</td>
    </tr>
    <tr>
      <th>48410.000</th>
      <th>2011-08-23</th>
      <td>17</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">48902.000</th>
      <th>2011-09-14</th>
      <td>16</td>
    </tr>
    <tr>
      <th>2011-09-21</th>
      <td>4</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">49883.000</th>
      <th>2011-10-19</th>
      <td>16</td>
    </tr>
    <tr>
      <th>2011-10-20</th>
      <td>4</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">50762.000</th>
      <th>2011-11-03</th>
      <td>17</td>
    </tr>
    <tr>
      <th>2011-11-21</th>
      <td>3</td>
    </tr>
    <tr>
      <th>51528.000</th>
      <th>2011-12-03</th>
      <td>19</td>
    </tr>
    <tr>
      <th>52770.000</th>
      <th>2012-01-16</th>
      <td>16</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">54116.000</th>
      <th>2012-02-01</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2012-02-02</th>
      <td>7</td>
    </tr>
    <tr>
      <th>2012-02-03</th>
      <td>9</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">54823.000</th>
      <th>2012-03-01</th>
      <td>17</td>
    </tr>
    <tr>
      <th>2012-03-21</th>
      <td>1</td>
    </tr>
    <tr>
      <th>55267.000</th>
      <th>2012-04-20</th>
      <td>21</td>
    </tr>
    <tr>
      <th>56238.000</th>
      <th>2012-05-22</th>
      <td>21</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">57288.000</th>
      <th>2012-06-04</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2012-06-15</th>
      <td>20</td>
    </tr>
    <tr>
      <th>57731.000</th>
      <th>2012-07-11</th>
      <td>21</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">58831.000</th>
      <th>2012-08-16</th>
      <td>17</td>
    </tr>
    <tr>
      <th>2012-08-20</th>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">59826.000</th>
      <th>2012-09-24</th>
      <td>18</td>
    </tr>
    <tr>
      <th>2012-09-28</th>
      <td>3</td>
    </tr>
    <tr>
      <th>60286.000</th>
      <th>2012-10-22</th>
      <td>21</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">61373.000</th>
      <th>2012-11-03</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2012-11-21</th>
      <td>20</td>
    </tr>
    <tr>
      <th>67729.000</th>
      <th>2013-05-15</th>
      <td>20</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">69524.000</th>
      <th>2013-06-10</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2013-06-11</th>
      <td>19</td>
    </tr>
    <tr>
      <th>71074.000</th>
      <th>2013-07-23</th>
      <td>20</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">72258.000</th>
      <th>2013-08-26</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2013-08-27</th>
      <td>16</td>
    </tr>
    <tr>
      <th>2013-08-30</th>
      <td>3</td>
    </tr>
    <tr>
      <th>73506.000</th>
      <th>2013-09-06</th>
      <td>17</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">74743.000</th>
      <th>2013-10-03</th>
      <td>14</td>
    </tr>
    <tr>
      <th>2013-10-16</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2013-10-23</th>
      <td>1</td>
    </tr>
    <tr>
      <th>75653.000</th>
      <th>2013-11-18</th>
      <td>17</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">77037.000</th>
      <th>2013-12-03</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2013-12-11</th>
      <td>16</td>
    </tr>
    <tr>
      <th>77551.000</th>
      <th>2014-01-16</th>
      <td>17</td>
    </tr>
    <tr>
      <th>79834.000</th>
      <th>2014-03-17</th>
      <td>33</td>
    </tr>
    <tr>
      <th>82530.000</th>
      <th>2014-05-07</th>
      <td>33</td>
    </tr>
    <tr>
      <th>84350.000</th>
      <th>2014-06-11</th>
      <td>33</td>
    </tr>
    <tr>
      <th>85021.000</th>
      <th>2014-07-14</th>
      <td>33</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">87144.000</th>
      <th>2014-08-11</th>
      <td>7</td>
    </tr>
    <tr>
      <th>2014-08-13</th>
      <td>1</td>
    </tr>
    <tr>
      <th>87479.000</th>
      <th>2014-09-04</th>
      <td>32</td>
    </tr>
    <tr>
      <th>89032.000</th>
      <th>2014-10-13</th>
      <td>8</td>
    </tr>
    <tr>
      <th>89832.000</th>
      <th>2014-11-07</th>
      <td>32</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">91147.000</th>
      <th>2014-12-05</th>
      <td>15</td>
    </tr>
    <tr>
      <th>2014-12-09</th>
      <td>11</td>
    </tr>
    <tr>
      <th>2014-12-10</th>
      <td>6</td>
    </tr>
    <tr>
      <th>92341.000</th>
      <th>2015-01-07</th>
      <td>32</td>
    </tr>
    <tr>
      <th>93672.000</th>
      <th>2015-03-12</th>
      <td>32</td>
    </tr>
    <tr>
      <th>93837.000</th>
      <th>2015-02-10</th>
      <td>32</td>
    </tr>
    <tr>
      <th>96624.000</th>
      <th>2015-04-07</th>
      <td>30</td>
    </tr>
    <tr>
      <th>97984.000</th>
      <th>2015-05-14</th>
      <td>31</td>
    </tr>
    <tr>
      <th>100916.000</th>
      <th>2015-07-07</th>
      <td>29</td>
    </tr>
    <tr>
      <th>105135.000</th>
      <th>2015-10-19</th>
      <td>30</td>
    </tr>
    <tr>
      <th>105770.000</th>
      <th>2015-12-07</th>
      <td>30</td>
    </tr>
    <tr>
      <th>109492.000</th>
      <th>2016-01-06</th>
      <td>30</td>
    </tr>
    <tr>
      <th>114382.000</th>
      <th>2016-05-19</th>
      <td>18</td>
    </tr>
    <tr>
      <th>115759.000</th>
      <th>2016-06-02</th>
      <td>20</td>
    </tr>
    <tr>
      <th>118510.000</th>
      <th>2016-08-04</th>
      <td>26</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">120036.000</th>
      <th>2016-09-12</th>
      <td>24</td>
    </tr>
    <tr>
      <th>2016-09-13</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2016-09-19</th>
      <td>3</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">124113.000</th>
      <th>2016-12-21</th>
      <td>26</td>
    </tr>
    <tr>
      <th>2016-12-22</th>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">125717.000</th>
      <th>2017-01-01</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2017-01-08</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-01-10</th>
      <td>31</td>
    </tr>
    <tr>
      <th>2017-01-18</th>
      <td>1</td>
    </tr>
    <tr>
      <th>127059.000</th>
      <th>2017-02-06</th>
      <td>35</td>
    </tr>
    <tr>
      <th>128516.000</th>
      <th>2017-03-14</th>
      <td>32</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">130359.000</th>
      <th>2017-04-01</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-04-12</th>
      <td>34</td>
    </tr>
    <tr>
      <th>131960.000</th>
      <th>2017-05-10</th>
      <td>34</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">133661.000</th>
      <th>2017-06-08</th>
      <td>27</td>
    </tr>
    <tr>
      <th>2017-06-20</th>
      <td>8</td>
    </tr>
    <tr>
      <th rowspan="24" valign="top">134975.000</th>
      <th>2017-05-01</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-05-03</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-05-04</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-05-06</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-05-07</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-05-08</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-05-09</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-05-13</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-05-14</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-05-18</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-05-21</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-05-23</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-05-26</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-05-28</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-06-03</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-06-04</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-06-07</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-06-09</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-06-10</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-06-11</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-06-15</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-06-17</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-06-20</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-06-22</th>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">135570.000</th>
      <th>2017-07-01</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-07-06</th>
      <td>34</td>
    </tr>
    <tr>
      <th>137350.000</th>
      <th>2017-08-04</th>
      <td>35</td>
    </tr>
    <tr>
      <th>138891.000</th>
      <th>2017-09-01</th>
      <td>34</td>
    </tr>
    <tr>
      <th>140538.000</th>
      <th>2017-10-04</th>
      <td>34</td>
    </tr>
    <tr>
      <th>142135.000</th>
      <th>2017-11-09</th>
      <td>32</td>
    </tr>
    <tr>
      <th rowspan="33" valign="top">143224.000</th>
      <th>2017-10-05</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-10-08</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-10-09</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-10-11</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-10-16</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-10-19</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-10-23</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-10-26</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-10-27</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-10-30</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-10-31</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-11-02</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-11-03</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-11-04</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-11-05</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-11-06</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-11-07</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-11-08</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-11-10</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-11-11</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-11-12</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-11-13</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-11-14</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-11-15</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-11-16</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-11-17</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-11-18</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-11-19</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-11-20</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-11-21</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-11-22</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-11-23</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-11-24</th>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">143691.000</th>
      <th>2017-12-05</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2017-12-06</th>
      <td>30</td>
    </tr>
    <tr>
      <th>144928.000</th>
      <th>2018-01-08</th>
      <td>39</td>
    </tr>
    <tr>
      <th>156199.000</th>
      <th>2018-02-06</th>
      <td>36</td>
    </tr>
    <tr>
      <th>158289.000</th>
      <th>2018-03-07</th>
      <td>36</td>
    </tr>
    <tr>
      <th>160260.000</th>
      <th>2018-04-10</th>
      <td>36</td>
    </tr>
    <tr>
      <th>161626.000</th>
      <th>2018-05-04</th>
      <td>36</td>
    </tr>
    <tr>
      <th>163056.000</th>
      <th>2018-06-05</th>
      <td>39</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">164465.000</th>
      <th>2018-07-03</th>
      <td>38</td>
    </tr>
    <tr>
      <th>2018-07-09</th>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">166200.000</th>
      <th>2018-07-31</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2018-08-01</th>
      <td>34</td>
    </tr>
    <tr>
      <th>2018-08-06</th>
      <td>1</td>
    </tr>
    <tr>
      <th>167649.000</th>
      <th>2018-09-04</th>
      <td>39</td>
    </tr>
    <tr>
      <th>169134.000</th>
      <th>2018-10-18</th>
      <td>39</td>
    </tr>
    <tr>
      <th>170622.000</th>
      <th>2018-11-06</th>
      <td>39</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">172115.000</th>
      <th>2018-12-03</th>
      <td>3</td>
    </tr>
    <tr>
      <th>2018-12-05</th>
      <td>36</td>
    </tr>
    <tr>
      <th>173731.000</th>
      <th>2019-01-03</th>
      <td>39</td>
    </tr>
    <tr>
      <th>175322.000</th>
      <th>2019-01-29</th>
      <td>40</td>
    </tr>
    <tr>
      <th>175484.000</th>
      <th>2019-02-05</th>
      <td>39</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">176983.000</th>
      <th>2019-02-19</th>
      <td>37</td>
    </tr>
    <tr>
      <th>2019-02-20</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2019-02-21</th>
      <td>1</td>
    </tr>
    <tr>
      <th>177584.000</th>
      <th>2019-03-07</th>
      <td>39</td>
    </tr>
    <tr>
      <th>178933.000</th>
      <th>2019-03-07</th>
      <td>40</td>
    </tr>
    <tr>
      <th>179371.000</th>
      <th>2019-03-21</th>
      <td>40</td>
    </tr>
    <tr>
      <th>179372.000</th>
      <th>2019-03-26</th>
      <td>39</td>
    </tr>
    <tr>
      <th>179882.000</th>
      <th>2019-04-09</th>
      <td>39</td>
    </tr>
    <tr>
      <th>182382.000</th>
      <th>2019-04-23</th>
      <td>40</td>
    </tr>
    <tr>
      <th>183210.000</th>
      <th>2019-05-07</th>
      <td>39</td>
    </tr>
    <tr>
      <th>185569.000</th>
      <th>2019-05-21</th>
      <td>40</td>
    </tr>
    <tr>
      <th>186845.000</th>
      <th>2019-06-05</th>
      <td>39</td>
    </tr>
    <tr>
      <th>190179.000</th>
      <th>2019-06-27</th>
      <td>40</td>
    </tr>
    <tr>
      <th>190614.000</th>
      <th>2019-07-02</th>
      <td>39</td>
    </tr>
    <tr>
      <th>192988.000</th>
      <th>2019-07-16</th>
      <td>40</td>
    </tr>
    <tr>
      <th>194926.000</th>
      <th>2019-07-31</th>
      <td>39</td>
    </tr>
    <tr>
      <th>199384.000</th>
      <th>2019-08-30</th>
      <td>39</td>
    </tr>
    <tr>
      <th>204455.000</th>
      <th>2019-10-03</th>
      <td>40</td>
    </tr>
    <tr>
      <th>205441.000</th>
      <th>2019-10-11</th>
      <td>40</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">205557.000</th>
      <th>2019-10-16</th>
      <td>38</td>
    </tr>
    <tr>
      <th>2019-10-17</th>
      <td>1</td>
    </tr>
    <tr>
      <th>206505.000</th>
      <th>2019-10-24</th>
      <td>39</td>
    </tr>
    <tr>
      <th>207394.000</th>
      <th>2019-10-29</th>
      <td>39</td>
    </tr>
    <tr>
      <th>208620.000</th>
      <th>2019-11-07</th>
      <td>39</td>
    </tr>
    <tr>
      <th>210445.000</th>
      <th>2019-11-19</th>
      <td>39</td>
    </tr>
    <tr>
      <th>211332.000</th>
      <th>2019-11-28</th>
      <td>39</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">212391.000</th>
      <th>2019-12-03</th>
      <td>36</td>
    </tr>
    <tr>
      <th>2019-12-04</th>
      <td>3</td>
    </tr>
    <tr>
      <th>213482.000</th>
      <th>2019-12-10</th>
      <td>36</td>
    </tr>
    <tr>
      <th>214352.000</th>
      <th>2019-12-19</th>
      <td>39</td>
    </tr>
    <tr>
      <th>215345.000</th>
      <th>2019-12-26</th>
      <td>39</td>
    </tr>
    <tr>
      <th>216216.000</th>
      <th>2020-01-03</th>
      <td>36</td>
    </tr>
    <tr>
      <th>217411.000</th>
      <th>2020-01-09</th>
      <td>38</td>
    </tr>
    <tr>
      <th>218270.000</th>
      <th>2020-01-17</th>
      <td>38</td>
    </tr>
    <tr>
      <th>219260.000</th>
      <th>2020-01-22</th>
      <td>38</td>
    </tr>
  </tbody>
</table>
</div>



**Observação importante:**

Nesse ponto da análise, um questionamento que surgiu é se, em cada OS, as medições de todos os instrumentos ocorriam no no mesmo dia. Isso é um fator importante já que condições metereológicas podem influenciar nos resultados e até mesmo a vazão natural da água na estrutura é muito afetada pelo tempo.

Uma forma simples de fazer uma análise foi o agrupamento dos dados por OS e a apresentação da tabela acima.

Com base nessa análise visual, a hipótese das medições seguirem um padrão de prazo foi descartada. Há grandes periodos entre a medições dos instrumentos numa mesma OS. 

Isso é um fator de impacto para as correlações já que a influência de fatores externos (regime de chuvas e secas, evaporação natural, fluxo da drenagem da estrutura) pode gerar desvios significativos.

Nesse trabalho, não foi realizado nenhum ajuste ou tratamento nos dados para corrigir ou mitigar essas variações.


```python
# ========================================
# Gravando arquivo de referencia
df_med.to_csv('df_ref.csv', sep=';', encoding='utf-8', decimal=',', index=False)
```


```python
del df_inst, df_med, df
del cols, cota, inst, j, i
del num_os, os
gc.collect()
```




    40



## Etapa 1.3 - Pluviometria

**Realiza carga da planilha com informações sobre precipitação pluviométrica na região.**

São realizadas operações para complementar registros faltantes e retirar duplicadas, visando criar uma sequência regular e constante de dias.

### Carga dos dados

Faz a carga somente das variáveis que interessam na análise. 


```python
# Carga da tabela de pluviometria em formato xlsx
cols = ['cod_inst', 'data_med', 'vlr_plu']
df_plu = pd.read_csv('pluviometria.csv', sep=';', decimal=',', encoding='latin1', usecols=cols)
df_plu.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cod_inst</th>
      <th>data_med</th>
      <th>vlr_plu</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pluv_01</td>
      <td>26/06/2010 08:00</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pluv_01</td>
      <td>27/06/2010 08:00</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pluv_01</td>
      <td>28/06/2010 08:00</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pluv_01</td>
      <td>29/06/2010 08:00</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pluv_01</td>
      <td>30/06/2010 08:00</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Reordenar colunas
df_plu = df_plu[['data_med', 'vlr_plu', 'cod_inst']]
df_plu.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data_med</th>
      <th>vlr_plu</th>
      <th>cod_inst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>26/06/2010 08:00</td>
      <td>0.000</td>
      <td>pluv_01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27/06/2010 08:00</td>
      <td>0.000</td>
      <td>pluv_01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28/06/2010 08:00</td>
      <td>0.000</td>
      <td>pluv_01</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Variável data_med foi carregada como string e com informações de data e hora.
# Remove informação de hh:mm, converte para datetime e ajusta formato para aaaa-mm-dd
df_plu['data_med'] = df_plu['data_med'].str.split(' ').str[0] 
df_plu['data_med'] = pd.to_datetime(df_plu['data_med'], format="%d/%m/%Y")
```


```python
# Ordena dados por data_medicao
df_plu = df_plu.sort_values(['data_med'])
df_plu.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data_med</th>
      <th>vlr_plu</th>
      <th>cod_inst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010-06-26</td>
      <td>0.000</td>
      <td>pluv_01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2010-06-27</td>
      <td>0.000</td>
      <td>pluv_01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2010-06-28</td>
      <td>0.000</td>
      <td>pluv_01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010-06-29</td>
      <td>0.000</td>
      <td>pluv_01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2010-06-30</td>
      <td>0.000</td>
      <td>pluv_01</td>
    </tr>
  </tbody>
</table>
</div>



Em discussões com o time técnico, identificou-se que há vários dias em que as medições não foram coletadas e que, nesses casos, a primeira coleta após esses períodos contém o acumulado das medições desses dias.

Assim, para os dias em que não houve coleta, ficou acordado que os valores de medição seriam preenchidos com zero.


```python
# Acrescentar registros para dias faltantes. As medições desses dias ficarão com valor zero
from datetime import timedelta

for i in range(1, len(df_plu)):
    start = df_plu.iloc[i-1,0]                  # captura data anterior
    end   = df_plu.iloc[i,0]                    # captura data posterior
    delta = (end-start).days                    # define o número de dias entre as datas
      
    if (delta > 1):
        for j in range(0, int(delta)-1):
            df_plu.loc[len(df_plu)+1] = [df_plu.iloc[i-1,0]+timedelta(days=1*(j+1)), 0, 'pluv_01']
                
df_plu = df_plu.sort_values('data_med')             
```


```python
# Preencher variável valor com dados faltantes com 0
df_plu['vlr_plu'] = df_plu['vlr_plu'].fillna(0)
```


```python
# Agrupa registro de dias repetidos em um único registro somando valores
df_plu = df_plu.groupby(['data_med', 'cod_inst'])['vlr_plu'].sum().to_frame(name='vlr_plu').reset_index()
df_plu.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data_med</th>
      <th>cod_inst</th>
      <th>vlr_plu</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010-06-26</td>
      <td>pluv_01</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2010-06-27</td>
      <td>pluv_01</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2010-06-28</td>
      <td>pluv_01</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010-06-29</td>
      <td>pluv_01</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2010-06-30</td>
      <td>pluv_01</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# ========================================
# Gravando arquivo tratado
df_plu.to_csv('df_plu.csv', sep=';', encoding='utf-8', decimal=',', index=False)
```


```python
df_plu.plot(x='data_med', y='vlr_plu', kind='hist', figsize=(15,5), grid='True', title='Histograma Pluviometria');
```


![png](output_47_0.png)



```python
df_plu.plot(x='data_med', y='vlr_plu', kind='area', figsize=(15,5), grid='True', title='Pluviometria em mm');
```


![png](output_48_0.png)



```python
del df_plu, cols, delta, end, i, j, start
gc.collect();
```




    8302



# Continua....



