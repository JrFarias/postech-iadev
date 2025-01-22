import matplotlib.pyplot as plt
from collections import defaultdict
import streamlit as st
import numpy as np

# Função para exibir gráficos de ocupação de caminhões
def exibir_graficos(caminhoes_utilizados):
    # Dados para o gráfico de volume e peso ocupados
    ids_caminhoes = [caminhao['id'] for caminhao in caminhoes_utilizados]
    volumes_usados = [caminhao['volume_usado'] for caminhao in caminhoes_utilizados]
    pesos_usados = [caminhao['peso_usado'] for caminhao in caminhoes_utilizados]
    valores_totais = [caminhao['valor_total'] for caminhao in caminhoes_utilizados]
    
    # Gráfico de volume e peso ocupado
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Volume usado por caminhão
    ax[0].bar(ids_caminhoes, volumes_usados, color='blue', alpha=0.7)
    ax[0].set_title('Volume Ocupado por Caminhão')
    ax[0].set_xlabel('Caminhão')
    ax[0].set_ylabel('Volume Ocupado (m³)')
    
    # Peso usado por caminhão
    ax[1].bar(ids_caminhoes, pesos_usados, color='green', alpha=0.7)
    ax[1].set_title('Peso Ocupado por Caminhão')
    ax[1].set_xlabel('Caminhão')
    ax[1].set_ylabel('Peso Ocupado (kg)')
    
    st.pyplot(fig)

    # Gráfico de valor total da carga separado
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(ids_caminhoes, valores_totais, color='orange', alpha=0.7)
    ax2.set_title('Valor Total da Carga por Caminhão')
    ax2.set_xlabel('Caminhão')
    ax2.set_ylabel('Valor Total (R$)')

    # Adicionar linha vermelha em Y=4000 para o limite mínimo de valor
    ax2.axhline(y=4000, color='red', linestyle='--', label='Limite Mínimo (R$ 4000)')

    # Adicionar legenda
    ax2.legend(loc='upper right')

    st.pyplot(fig2)

    # Gráfico de eficiência de ocupação
    eficiencia_volume = [caminhao['volume_usado'] / caminhao['volume'] * 100 for caminhao in caminhoes_utilizados]
    eficiencia_peso = [caminhao['peso_usado'] / caminhao['peso'] * 100 for caminhao in caminhoes_utilizados]
    
    fig3, ax3 = plt.subplots(1, 2, figsize=(12, 5))

    # Eficiência de volume
    ax3[0].bar(ids_caminhoes, eficiencia_volume, color='blue', alpha=0.7)
    ax3[0].set_title('Eficiência de Volume por Caminhão')
    ax3[0].set_xlabel('Caminhão')
    ax3[0].set_ylabel('Eficiência de Volume (%)')

    # Eficiência de peso
    ax3[1].bar(ids_caminhoes, eficiencia_peso, color='green', alpha=0.7)
    ax3[1].set_title('Eficiência de Peso por Caminhão')
    ax3[1].set_xlabel('Caminhão')
    ax3[1].set_ylabel('Eficiência de Peso (%)')
    
    st.pyplot(fig3)

# Função para mostrar a distribuição de produtos nos caminhões
def exibir_distribuicao(caminhoes_utilizados):
    for caminhao in caminhoes_utilizados:
        st.write(f"Caminhão {caminhao['id']}:")
        st.write(f"- Volume Usado: {caminhao['volume_usado']} m³")
        st.write(f"- Peso Usado: {caminhao['peso_usado']} kg")
        st.write(f"- Valor Total da Carga: R$ {caminhao['valor_total']}")
        st.write("---")

@st.cache_data
def calcular_soma_por_produto(solution, produtos_amostra, caminhao_id, modo='volume'):
    # 1. Somar (volume/peso/valor) de cada tipo de produto para este caminhão
    soma_por_produto = defaultdict(float)
    
    total_valor = 0
    for i, produto in enumerate(produtos_amostra):
        if solution[i] == caminhao_id:
            nome_produto = produto['nome']
            if modo == 'volume':
                soma_por_produto[nome_produto] += produto['volume']
            elif modo == 'peso':
                soma_por_produto[nome_produto] += produto['peso']
            elif modo == 'valor':
                soma_por_produto[nome_produto] += produto['valor']
                total_valor += produto['valor']
            else:
                raise ValueError("O modo deve ser 'volume', 'peso' ou 'valor'.")

    return soma_por_produto, total_valor

@st.cache_data
def gerar_grafico(soma_por_produto, modo, caminhao_id, colormap='tab20'):
    # 2. Preparar dados para plotar
    labels = list(soma_por_produto.keys())
    valores = list(soma_por_produto.values())

    # (Opcional) Ordenar as fatias de maior para menor
    pares_ordenados = sorted(zip(labels, valores), key=lambda x: x[1], reverse=True)
    labels, valores = zip(*pares_ordenados)  # reatribui as duas listas
    
    # 3. Escolher paleta de cores
    cmap = plt.get_cmap(colormap)
    cores = [cmap(i) for i in np.linspace(0, 1, len(labels))]

    # 4. Criar gráfico de pizza + donut (disco de memória)
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'aspect': 'equal'})

    wedges, texts, autotexts = ax.pie(
        valores, 
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=cores,
        pctdistance=0.80,    # ajustar posição dos percentuais
        labeldistance=1.05,  # ajustar posição dos rótulos
        shadow=True,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1}
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_size(10)

    donut_central = plt.Circle((0, 0), 0.60, color='white')
    ax.add_artist(donut_central)

    # 5. Título do gráfico
    title = f"Caminhão - {str(caminhao_id)}"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig

def plot_discos_memoria_todos_caminhoes(solution, produtos_amostra, caminhoes, colormap='tab20'):
    num_caminhoes = len(caminhoes)
    num_colunas = 2  # Número de gráficos por linha
    largura = 5  # Largura fixa do gráfico
    altura = 4   # Altura fixa do gráfico
    
    # Organizar os gráficos em colunas
    for i, caminhao in enumerate(caminhoes):
        caminhao_id = caminhao['id']
        
        # Calcular soma por produto para valor (monetário)
        soma_por_produto_valor, total_valor = calcular_soma_por_produto(solution, produtos_amostra, caminhao_id, modo='valor')

        if soma_por_produto_valor:
            # Organizar a exibição em 3 gráficos por linha
            if i % num_colunas == 0:
                colunas = st.columns(num_colunas)  # Divida a tela em 3 colunas
            
            # Exibir o resumo dos dados para o caminhão, acima do gráfico
            colunas[i % num_colunas].write(f"### Resumo dos dados para o Caminhão {caminhao_id}:")
            colunas[i % num_colunas].write(f"**Valor total da carga (monetário):** R${total_valor:.2f}")
            
            # Gerar gráfico de itens (somente valores monetários)
            fig_valor = gerar_grafico(soma_por_produto_valor, modo='valor', colormap=colormap, caminhao_id=caminhao_id)
            
            # Ajustar o tamanho da figura para garantir que todos os gráficos tenham o mesmo tamanho
            fig_valor.set_size_inches(largura, altura)
            
            # Exibir gráfico na coluna correta
            colunas[i % num_colunas].pyplot(fig_valor)

        else:
            st.write(f"Nenhum produto alocado no caminhão {caminhao_id}.")
