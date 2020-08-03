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

### Carga e tratamento básico dos dados


```python
cols = ['cod_inst', 'cota_fundo']
df_inst = pd.read_csv('instrumentos.csv', sep=';', encoding='latin1', decimal=',', usecols=cols)
```


```python
# Preenche campos vazios na variavel cota_fundo
df_inst['cota_fundo'] = df_inst['cota_fundo'].fillna(0)
```


```python
# Ordena registros por código de instrumento
df_inst = df_inst.sort_values(['cod_inst'])
df_inst.head()
```




<div>
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



## Etapa 1.2 - Dados de Medições

**Realiza carga da tabela com informações sobre as medições dos instrumentos ao longo do tempo.**

Em análise prévia, foram identificadas variáveis essenciais à análise e somente essas são carregadas.

Também são realizados filtros para eliminação de registros sem utilidade (sem dados) e preenchimento de campos sem valores de medição com os dados oriundos da tabela de instrumentos tratada anteriormente.

### Carga e tratamento básico dos dados

Faz a carga somente das variáveis que interessam na análise. 


```python
cols = ['num_os', 'cod_inst', 'tipo_inst', 'situacao', 'condicao', 'data_med', 'valor', 'unidade']
df_med = pd.read_csv('medicoes.csv', sep=';', encoding='latin1', decimal=',', usecols=cols)
```


```python
print('Dimensões de df_med:', df_med.shape)
df_med.head(3)
```

    Dimensões de df_med: (5786, 8)
    




<div>
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
      <td>4708.000</td>
      <td>piez_02</td>
      <td>piezometro</td>
      <td>Realizada</td>
      <td>NaN</td>
      <td>10/02/2005 07:40</td>
      <td>911.910</td>
      <td>m</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4708.000</td>
      <td>piez_03</td>
      <td>piezometro</td>
      <td>Realizada</td>
      <td>NaN</td>
      <td>10/02/2005 07:40</td>
      <td>937.946</td>
      <td>m</td>
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




```python
# Variável data_med foi carregada como string e com informações de data e hora.
# Remove informação de hh:mm, converte para datetime e ajusta formato para aaaa-mm-dd
df_med['data_med'] = df_med['data_med'].str.split(' ').str[0] 
df_med['data_med'] = pd.to_datetime(df_med['data_med'], format="%d/%m/%Y")
df_med.head(3)
```




<div>
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
      <td>4708.000</td>
      <td>piez_02</td>
      <td>piezometro</td>
      <td>Realizada</td>
      <td>NaN</td>
      <td>2005-02-10</td>
      <td>911.910</td>
      <td>m</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4708.000</td>
      <td>piez_03</td>
      <td>piezometro</td>
      <td>Realizada</td>
      <td>NaN</td>
      <td>2005-02-10</td>
      <td>937.946</td>
      <td>m</td>
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
      <th>1</th>
      <td>4708.000</td>
      <td>piez_02</td>
      <td>piezometro</td>
      <td>Nao</td>
      <td>2005-02-10</td>
      <td>911.910</td>
      <td>m</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4708.000</td>
      <td>piez_03</td>
      <td>piezometro</td>
      <td>Nao</td>
      <td>2005-02-10</td>
      <td>937.946</td>
      <td>m</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4708.000</td>
      <td>piez_06</td>
      <td>piezometro</td>
      <td>Nao</td>
      <td>2005-02-10</td>
      <td>919.236</td>
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
# Usa dataset de instrumentos para preencher campos vazios com dados de referencia
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
# Verifica se existe algum registro sem informação
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
# Análise visual rápida para entender se medições dentro de cada OS são executadas no mesmo dia
# Durante o processo foram apresentados todos os registros, mas para essa documentação limitei somente aos 10 primeiros#
pd.DataFrame(df_med.groupby(['num_os', 'data_med'])['tipo_inst'].count()).head(10)

```




<div>
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
  </tbody>
</table>
</div>



**Observação importante:**

Nesse ponto da análise, um questionamento que surgiu é se, em cada OS, as medições de todos os instrumentos ocorriam no no mesmo dia. Isso é um fator importante já que condições metereológicas podem influenciar nos resultados e até mesmo a vazão natural da água na estrutura é muito afetada pelo tempo.

Uma forma simples de fazer uma análise foi o agrupamento dos dados por OS e a apresentação da tabela acima.

Com base nessa análise visual, a hipótese das medições seguirem um padrão de prazo foi descartada. Há grandes periodos entre a medições dos instrumentos numa mesma OS. 

Isso é um fator de impacto para as correlações já que a influência de fatores externos (regime de chuvas e secas, evaporação natural, fluxo da drenagem da estrutura) pode gerar desvios significativos.

Nesse trabalho, não foi realizado nenhum ajuste ou tratamento nos dados para corrigir ou mitigar essas variações.

## Etapa 1.3 - Pluviometria

**Realiza carga da planilha com informações sobre precipitação pluviométrica na região.**

São realizadas operações para complementar registros faltantes e retirar duplicadas, visando criar uma sequência regular e constante de dias.

### Carga e tratamento básico dos dados

Faz a carga somente das variáveis que interessam na análise. 


```python
# Carga da tabela de pluviometria em formato xlsx
cols = ['cod_inst', 'data_med', 'vlr_plu']
df_plu = pd.read_csv('pluviometria.csv', sep=';', decimal=',', encoding='latin1', usecols=cols)
```


```python
# Reordenar colunas
df_plu = df_plu[['data_med', 'vlr_plu', 'cod_inst']]
df_plu.head(3)
```




<div>
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
df_plu.head(3)
```




<div>
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
  </tbody>
</table>
</div>



Identificou-se que há vários dias em que as medições não foram coletadas e que, nesses casos, a primeira coleta após esses períodos contém o acumulado das medições desses dias.

Assim, para os dias em que não houve coleta, definiu-se que os valores de medição seriam preenchidos com zero.


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
# Gravando arquivos tratados
df_med.to_csv('df_ref.csv', sep=';', encoding='utf-8', decimal=',', index=False)
df_inst.to_csv('df_inst.csv', sep=';', encoding='utf-8', decimal=',', index=False)
df_plu.to_csv('df_plu.csv', sep=';', encoding='utf-8', decimal=',', index=False)
```

## Continua em breve....


```python

```

