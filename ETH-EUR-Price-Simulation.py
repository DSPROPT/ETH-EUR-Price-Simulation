import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Recupera dados do preço ETH/EUR do Kraken
resposta = requests.get('https://api.kraken.com/0/public/OHLC', params={'pair': 'XETHZEUR', 'interval': 1440}) # Dados diários
dados = resposta.json()['result']['XETHZEUR']
df = pd.DataFrame(dados, columns=['tempo', 'aberto', 'alto', 'baixo', 'fechado', 'vwap', 'volume', 'contagem'])
df['fechado'] = pd.to_numeric(df['fechado'])


# Usa retornos logarítmicos para estimar a deriva e volatilidade
df['retorno_log'] = np.log(df['fechado'] / df['fechado'].shift())
mu = df['retorno_log'].mean()
sigma = df['retorno_log'].std(ddof=1)


# Define parâmetros e grade de tempo
T = 1.0  # Um ano
N = 365  # Passos diários
dt = T/N
t = np.linspace(0, T, N)


# Prepara para simulação
n_simulações = 1000
simulações = []
for _ in range(n_simulações):
    W = np.random.standard_normal(size = N) 
    W = np.cumsum(W)*np.sqrt(dt) ### movimento browniano padrão ###
    X = (mu-0.5*sigma**2)*t + sigma*W 


    # Converte para Movimento Browniano Geométrico
    S = df['fechado'].iloc[-1]*np.exp(X)
    simulações.append(S)


# Prepara para animação
fig, ax = plt.subplots()


# Função de atualização de animação
def atualizar(num):
    ax.clear()
    ax.set_title('Movimento Browniano Geométrico - Simulação do Preço ETH/EUR')
    for i in range(num):
        ax.plot(t, simulações[i])
    # Ajusta limites dos eixos
    ax.set_xlim([0, 1])
    
    # Evita calcular max e min para um array vazio
    if num > 0:
        preço_max = np.max([np.max(s) for s in simulações[:num]])
        preço_min = np.min([np.min(s) for s in simulações[:num]])
        ax.set_ylim([0.9*preço_min, 1.1*preço_max])


        # Adiciona informações de contagem, alto e baixo preço na plotagem
        count_text = f'Visualizações: {num}'
        high_price_text = f'Preço alto: {preço_max:.2f} EUR'
        low_price_text = f'Preço baixo: {preço_min:.2f} EUR'
        ax.text(0.02, 0.98, count_text + '\n' + high_price_text + '\n' + low_price_text, 
                transform=ax.transAxes, verticalalignment='top')


# Cria animação
ani = FuncAnimation(fig, atualizar, n_simulações, interval=50, repeat=True)
plt.show()
