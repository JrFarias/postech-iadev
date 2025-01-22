import streamlit as st
from ags import AlgoritmoGenetico  # Certifique-se de que o módulo esteja no mesmo diretório ou no path correto
import random
import pandas as pd
import numpy as np
import utils
from ags import heuristica_gulosa
import matplotlib.pyplot as plt

def gerar_caminhoes(valor_minimo):
    caminhoes = []
    for i in range(10):
        caminhao = {
            "id": i + 1,
            "volume": 200,
            "peso": 2000,
            "volume_usado": 0,
            "peso_usado": 0,
            "valor_minimo": valor_minimo,
            "valor_total": 0,
        }
        caminhoes.append(caminhao)
    return caminhoes


produtos = [
    {"id": 1, "nome": "Fardo de Latas", "volume": 0.2, "peso": 30, "valor": 150, "prioridade": "baixa"},
    {"id": 2, "nome": "Engradado de Garrafas", "volume": 0.7, "peso": 25, "valor": 100, "prioridade": "média"},
    {"id": 3, "nome": "Caixa de amendoim", "volume": 0.25, "peso": 10, "valor": 50, "prioridade": "alta"},
    {"id": 4, "nome": "Caixa de vinho", "volume": 0.25, "peso": 15, "valor": 100, "prioridade": "média"},
    {"id": 5, "nome": "Caixa de chá gelado", "volume": 0.4, "peso": 12, "valor": 20, "prioridade": "baixa"},
    {"id": 6, "nome": "Pacote de farinha de trigo", "volume": 0.5, "peso": 25, "valor": 35, "prioridade": "alta"},
]


def gerar_produtos_amostra():
    np.random.seed(42)
    produtos_amostra = [
        {**np.random.choice(produtos), "id": i + 1} for i in range(900)
    ]
    return produtos_amostra


def main():
    st.title("Otimização de Carga de Caminhões com Algoritmos Genéticos")

    # Introdução e descrição dos parâmetros
    st.subheader("Introdução ao Problema")
    st.write("""
    Este aplicativo utiliza algoritmos genéticos para resolver o problema de otimizar a carga de caminhões. 
    Nosso objetivo é maximizar o valor total transportado respeitando as limitações de peso e volume de cada caminhão, 
    bem como um valor mínimo de carga definido pelo usuário.
    """)

    st.subheader("Definição do Problema")
    st.write("""
    Este projeto aborda um problema real de logística: otimizar a distribuição de produtos em caminhões, de forma a maximizar o valor transportado respeitando as seguintes restrições:
    - Cada caminhão tem um limite máximo de peso e volume.
    - Os produtos transportados em cada caminhão devem atingir um valor mínimo configurado pelo usuário.
    """)

    st.image("knapsack.png")

    st.write("""
    **Objetivo:** Desenvolver um algoritmo genético que encontre a melhor configuração para atender às restrições acima, maximizando o valor transportado.

    **Critérios de Sucesso:**
    1. O algoritmo deve gerar soluções viáveis para o problema.
    2. A solução deve ser comparada com métodos convencionais para demonstrar a eficácia do algoritmo genético.
    3. A alocação dos itens deve respeitar as restrições iniciais:
            - Não ultrapassar Volume.
            - Não ultrapassar a Carga.
            - Valor total da carga deve ser maior ou igual ao limite mínimo declarado (R$ 4000,00)
    """)

    # Entrada para valor mínimo dos caminhões
    st.subheader("Configuração dos Caminhões")
    valor_minimo = st.number_input(
        "Valor Mínimo de Carga por Caminhão",
        value=4000,
        help="Define o valor mínimo que cada caminhão deve transportar para o problema de otimização."
    )
    
    with st.expander("Parâmetros do Algoritmo Genético"):
        st.info("""
        **Descrição dos Parâmetros do Algoritmo Genético:**
        - **Tamanho da População:** Representa o número de soluções possíveis que serão geradas em cada iteração. 
        Populações maiores permitem maior exploração, mas aumentam o tempo de processamento.
        Exemplo: Se forem 300 indivíduos, o algoritmo tentará 300 soluções por geração.
        - **Número de Gerações:** Quantidade de ciclos de evolução para o algoritmo. 
        Gerações maiores aumentam as chances de encontrar soluções ideais, mas podem ser mais lentas.
        - **Taxa de Mutação:** Controla a probabilidade de alterações aleatórias nos indivíduos.
        Por exemplo, uma taxa de 5% significa que 5% da população sofrerá mutações.
        - **Método de Inicialização:** Define como a população inicial será gerada.
        - `Random`: Soluções completamente aleatórias.
        - `Heuristic`: Usa regras pré-definidas para criar soluções iniciais.
        - **Método de Seleção:** Estratégia para escolher indivíduos para reprodução.
        - `Roulette`: Baseado em probabilidade proporcional ao fitness.
        - `Tournament`: Indivíduos competem, e o melhor é selecionado.
        - **Método de Mutação:** Define como a alteração será feita.
        - `Swap`: Troca posições de dois elementos.
        - `Inversion`: Inverte uma sequência dentro de um indivíduo.
        - `Substitution`: Substitui um elemento por outro.
        - **Taxa de Mutação Dinâmica:** Permite ajustar a taxa de mutação conforme o algoritmo evolui. 
        Por exemplo, a taxa pode diminuir nas gerações finais para evitar alterações indesejadas.
        """)

    caminhoes = gerar_caminhoes(int(valor_minimo))
    produtos_amostra = gerar_produtos_amostra()

    # Agregação do dataframe de produtos
    produtos_df = pd.DataFrame(produtos_amostra)
    produtos_agrupados = produtos_df.groupby("nome").agg(
        quantidade=("id", "count"),
        volume_unitario=("volume", "first"),
        peso_unitario=("peso", "first"),
        valor_unitario=("valor", "first"),
        prioridade=("prioridade", "first")
    ).reset_index()

    st.write("### Caminhões:")
    st.dataframe(pd.DataFrame(caminhoes), hide_index=True)

    st.write("### Produtos Agrupados:")
    st.dataframe(produtos_agrupados, hide_index=True)

    # Configurações do algoritmo genético
    st.header("Configurações do Algoritmo Genético")
    col1, col2 = st.columns(2)

    with col1:
        population_size = st.slider(
            "Tamanho da População", 50, 1000, 300, 50,
            help="Número de indivíduos (soluções possíveis) em cada geração."
        )
        generations = st.slider(
            "Número de Gerações", 10, 5000, 50, 10,
            help="Quantidade de iterações para evolução do algoritmo."
        )
        mutation_rate = st.slider(
            "Taxa de Mutação", 0.01, 0.5, 0.05, 0.01,
            help="Frequência de alterações nos indivíduos em cada geração."
        )

    with col2:
        initial_population_method = st.selectbox(
            "Método de Inicialização", ["random", "heuristic"], 0,
            help="Define como a população inicial será criada."
        )
        selection_method = st.selectbox(
            "Método de Seleção", ["roulette", "tournament"], 0,
            help="Como os indivíduos serão escolhidos para reprodução."
        )
        mutation_method = st.selectbox(
            "Método de Mutação", ["swap", "inversion", "substitution"], 0,
            help="Tipo de modificação aplicada nos indivíduos."
        )
        dynamic_mutation_active = st.checkbox(
            "Ativar Taxa de Mutação Dinâmica", False,
            help="Permite ajustar dinamicamente a taxa de mutação."
        )

    st_placeholder = st.empty()

    # Executar o algoritmo genético
    if st.button("Executar Algoritmo Genético"):
        st.write("Executando o algoritmo...")

        ag = AlgoritmoGenetico(
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            initial_population_method=initial_population_method,
            selection_method=selection_method,
            mutation_method=mutation_method,
            dynamic_mutation_active=dynamic_mutation_active,
            caminhoes=caminhoes,
            produtos_amostra=produtos_amostra
        )

        # Passar o espaço reservado para atualização do gráfico
        best_solution, best_fitness, caminhoes_utilizados, fitness_history = ag.genetic_algorithm(st_placeholder)
    
        # Exibir os resultados finais
        st.subheader("Melhor Solução Encontrada")
        st.write("Fitness da Melhor Solução:", best_fitness)

        st.subheader("Distribuição de Produtos nos Caminhões")
        utils.exibir_graficos(caminhoes_utilizados=caminhoes_utilizados)
        utils.plot_discos_memoria_todos_caminhoes(solution=best_solution, caminhoes=caminhoes_utilizados, produtos_amostra=ag.produtos_amostra)

        st.header("Testes e Resultados")

        # Comparação com métodos convencionais
        st.subheader("Comparação de Resultados")
        st.write("""
        Além do algoritmo genético, testamos o desempenho de um método convencional (heurística gulosa) para resolver o mesmo problema. 
        Os resultados estão apresentados abaixo.
        """)

        # Simulação do método convencional
        resultados_convencionais = heuristica_gulosa(caminhoes, produtos_amostra)

        # Criar DataFrame para os resultados convencionais
        resultados_convencionais_df = []
        for caminhao in resultados_convencionais:
            peso_usado = caminhao["peso_usado"]
            volume_usado = caminhao["volume_usado"]
            valor_total = caminhao["valor_total"]
            
            resultados_convencionais_df.append({
                "Caminhão ID": caminhao["id"],
                "Volume Usado": volume_usado,
                "Peso Usado": peso_usado,
                "Valor Total": valor_total,
            })

        # Exibir os resultados convencionais em um DataFrame
        st.write("**Resultados do Método Convencional (Heurística Gulosa):**")
        st.dataframe(pd.DataFrame(resultados_convencionais_df), hide_index=True)

        # Agora, para a solução genérica
        ids_caminhoes = [caminhao['id'] for caminhao in caminhoes_utilizados]
        volumes_usados = [caminhao['volume_usado'] for caminhao in caminhoes_utilizados]
        pesos_usados = [caminhao['peso_usado'] for caminhao in caminhoes_utilizados]
        valores_totais = [caminhao['valor_total'] for caminhao in caminhoes_utilizados]


        # Exibir a melhor solução genérica no formato ajustado
        df_comparacao_genetico = pd.DataFrame({
            'Caminhão ID': ids_caminhoes,
            'Volume Usado': volumes_usados,
            'Peso Usado': pesos_usados,
            'Valor Total': valores_totais,
        })

        st.write("### Melhor Solução Encontrada (Algoritmo Genético):")
        st.dataframe(df_comparacao_genetico, hide_index=True)

        st.subheader("Conclusão")

        st.write("""
        Após a execução do algoritmo genético e do método convencional (heurística gulosa), os resultados mostraram diferenças significativas na alocação de produtos nos caminhões.
        
        #### 1. Por que os Algoritmos Genéticos se Sobressaíram Nesse Problema?

        Exploração Ampla: Ao contrário da heurística gulosa, que segue um critério fixo, o algoritmo genético explora várias soluções possíveis, permitindo uma melhor adaptação às restrições do problema, como volume e peso.

        Evita Soluções Subótimas: O AG não fica preso a soluções locais, pois gera novas soluções a partir de uma população de candidatos, oferecendo uma chance maior de encontrar a solução ótima.

        Melhor Uso de Recursos: O AG conseguiu otimizar de maneira mais eficiente o uso de recursos (como espaço e peso), resultando em soluções mais equilibradas.
        
        #### 2. Motivos dos Zeros no Método Heurístico Guloso

        O método heurístico guloso apresentou resultados com **valor total igual a zero** em alguns caminhões devido a uma combinação de fatores:

        - **Produtos Não Alocados**: Em alguns casos, os produtos não conseguiam ser alocados aos caminhões devido às **restrições de peso** ou **volume**.
                 Nestes casos, quando o peso ou o volume ultrapassaram dos limites.

        Portanto, enquanto a heurística gulosa é eficiente e rápida, ela não é tão eficaz quanto os algoritmos genéticos em problemas com muitas variáveis e restrições, como no nosso caso.
        """)

        st.subheader("Como tornar o model mais robusto e realista?")
        st.write("""
                 - Inclusão de Novas Variáveis
                 - Melhor Cálculo de Fatores de Penalidade
                 - Melhoria nas Operações de Cruzamento e Mutação
                 - Ajuste do Tamanho da População e Taxas de Mutação
                 """)

if __name__ == "__main__":
    main()
