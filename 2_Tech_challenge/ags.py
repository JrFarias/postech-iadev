import numpy as np
import random
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

class AlgoritmoGenetico:

    def __init__(self, caminhoes, produtos_amostra, population_size=300, generations=50, mutation_rate=0.05, initial_population_method: str = "", selection_method: str = "", mutation_method: str = "", dynamic_mutation_active: bool = False):
        
        self.caminhoes = caminhoes
        self.produtos_amostra = produtos_amostra
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        
        self.fitness_scores = {
            "generations": [],
            "best_fitness": [],
        }
        self.initial_population_method = initial_population_method
        self.selection_method = selection_method
        self.mutation_method = mutation_method
        self.dynamic_mutation_active = dynamic_mutation_active

    def fitness(self, solution):
        valor_total = 0
        penalidade = 0
        
        # Resetar volume e peso usados antes de calcular a fitness
        for caminhao in self.caminhoes:
            caminhao["volume_usado"] = 0
            caminhao["peso_usado"] = 0

        for caminhao in self.caminhoes:
            volume_usado = 0
            peso_usado = 0
            valor_total_caminhao = 0  # Inicializa o valor total para cada caminhão

            for produto, alocacao in zip(self.produtos_amostra, solution):
                if alocacao == caminhao["id"]:
                    volume_usado += produto["volume"]
                    peso_usado += produto["peso"]
                    valor_total_caminhao += produto["valor"]  # Acumula o valor dos produtos no caminhão

            # Atualizar volume e peso usados
            caminhao["volume_usado"] += volume_usado
            caminhao["peso_usado"] += peso_usado

            # Penalidades por exceder capacidade
            if volume_usado > caminhao["volume"]:
                penalidade += (volume_usado - caminhao["volume"]) * 10
            if peso_usado > caminhao["peso"]:
                penalidade += (peso_usado - caminhao["peso"]) * 10

            valor_total = sum(produto["valor"] for produto, alocacao in zip(self.produtos_amostra, solution) if alocacao == caminhao["id"])
            if valor_total < caminhao["valor_minimo"]:
                penalidade += (caminhao["valor_minimo"] - valor_total) * 10
            
            prioridades = [produto['prioridade'] for produto in self.produtos_amostra]
            if 'alta' in prioridades and 'média' in prioridades or 'baixa' in prioridades:
                penalidade += 80  # Penalidade por misturar produtos com diferentes prioridades de entrega


        return valor_total - penalidade

    def generate_heuristic_solution(self):
        solution = []
        for produto in self.produtos_amostra:
            allocated = False
            for caminhao in self.caminhoes:
                if caminhao["volume_usado"] + produto["volume"] <= caminhao["volume"] and caminhao["peso_usado"] + produto["peso"] <= caminhao["peso"]:
                    solution.append(caminhao["id"])
                    caminhao["volume_usado"] += produto["volume"]
                    caminhao["peso_usado"] += produto["peso"]
                    allocated = True
                    break
            if not allocated:
                # Fallback to allocate to a random truck if no suitable truck is found
                random_truck = random.choice(self.caminhoes)
                solution.append(random_truck["id"])
                random_truck["volume_usado"] += produto["volume"]
                random_truck["peso_usado"] += produto["peso"]
        return solution

    # Função para gerar uma solução inicial
    def generate_solution(self):
        return [random.choice([c["id"] for c in self.caminhoes]) for _ in self.produtos_amostra]

    # Função para criar a população inicial
    def initialize_population(self):
        if (self.aggregate_fitness_score == "heuristic"):
            return [self.generate_heuristic_solution() for _ in range(self.population_size)]
        else:
            return [self.generate_solution() for _ in range(self.population_size)]

    # Função de seleção por torneio
    def selection(self, population, fitness_scores):
        if (self.selection_method == "tournament"):
            total_fitness = sum(fitness_scores)
            selection_probs = [f / total_fitness for f in fitness_scores]
            return random.choices(population, weights=selection_probs, k=2)
        else:
            return random.choices(population, weights=fitness_scores, k=2)

    # Função de cruzamento (crossover)
    def crossover(self, parent1, parent2):
        point = random.randint(0, len(parent1) - 1)
        child = parent1[:point] + parent2[point:]
        return child

    # Função de mutação
    def mutate(self, solution, mutation_rate):
        if (self.mutation_method == "swap"):
            #  Exemplo: Mutação de troca
            index1 = random.randint(0, len(solution) - 1)
            index2 = random.randint(0, len(solution) - 1)
            solution[index1], solution[index2] = solution[index2], solution[index1]
        elif (self.mutation_method == "inversion"):
            #  Mutação de inversão
            index1 = random.randint(0, len(solution) - 1)
            index2 = random.randint(0, len(solution) - 1)
            if index1 > index2:
                index1, index2 = index2, index1
            solution[index1:index2] = reversed(solution[index1:index2])
        else:
            # Mutação de substituição
            if random.random() < mutation_rate:
                index = random.randint(0, len(solution) - 1)
                solution[index] = random.choice([c["id"] for c in self.caminhoes])

    def plot_fitness(self, fitness_history):
        plt.ion()
        fig, ax = plt.subplots()

        line, = ax.plot([], [], marker='o', markersize=4, markevery=10)
        ax.set_xlabel('Gerações')
        ax.set_ylabel('Melhor Fitness')
        ax.set_title('Evolução do Fitness ao Longo das Gerações')

        for i in range(len(fitness_history)):
            line.set_data(range(i + 1), fitness_history[:i + 1])
            ax.set_xlim(0, len(fitness_history))
            ax.set_ylim(min(fitness_history) - 10, max(fitness_history) + 10)

            clear_output(wait=True)
            display(fig)
            plt.pause(0.1)

        plt.ioff()
        plt.show()

    def plot_fitness_dynamic(self, fitness_history, generation):
        plt.clf()  # Limpa o gráfico anterior
        plt.plot(range(len(fitness_history)), fitness_history, marker='o', linestyle='-', color='b', markersize=4)
        plt.xlabel('Gerações')
        plt.ylabel('Melhor Fitness')
        plt.title(f'Evolução do Fitness | Geração: {generation + 1}')
        plt.grid(True)
        plt.pause(0.1)  # Pausa para dar a sensação de animação

    def aggregate_fitness_score(self, generation, best_fitness):
        self.fitness_scores['generations'].append(generation)
        self.fitness_scores['best_fitness'].append(best_fitness)

    def genetic_algorithm(self, st_placeholder):
        fitness_history = []
        population = self.initialize_population()
        mutation_rate = self.mutation_rate
        previous_best_fitness = None

        for generation in range(self.generations):
            # Avaliar a população
            fitness_scores = [self.fitness(sol) for sol in population]

            # Nova geração
            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.selection(population, fitness_scores)
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)

                self.mutate(child1, mutation_rate)
                self.mutate(child2, mutation_rate)

                new_population.extend([child1, child2])

            population = new_population

            # Melhor solução da geração
            best_fitness = max(fitness_scores)
            fitness_history.append(best_fitness)

            # Atualizar gráfico no Streamlit
            df = pd.DataFrame({
                "Geração": list(range(1, len(fitness_history) + 1)),
                "Fitness": fitness_history
            })

            # Criar gráfico com Altair
            chart = (
                alt.Chart(df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Geração", title="Geração"),
                    y=alt.Y("Fitness", scale=alt.Scale(domain=[min(fitness_history), max(fitness_history)]), title="Fitness"),
                )
                .properties(width=700, height=400)
            )

            st_placeholder.altair_chart(chart, use_container_width=True)

            if self.dynamic_mutation_active:
                # Atualizar a taxa de mutação dinamicamente
                if previous_best_fitness is not None and best_fitness == previous_best_fitness:
                    mutation_rate = min(mutation_rate * 1.5, 0.5)
                else:
                    mutation_rate = self.mutation_rate

            previous_best_fitness = best_fitness
            self.aggregate_fitness_score(generation, best_fitness)

        # Melhor solução final
        final_fitness_scores = [self.fitness(sol) for sol in population]
        best_final_solution = population[np.argmax(final_fitness_scores)]

        caminhoes_utilizados = [
            {**caminhao, "valor_total": sum(produto["valor"] for produto, alocacao in zip(self.produtos_amostra, best_final_solution) if alocacao == caminhao["id"])}
            for caminhao in self.caminhoes
            if caminhao["volume_usado"] > 0 or caminhao["peso_usado"] > 0
        ]

        return best_final_solution, max(final_fitness_scores), caminhoes_utilizados, fitness_history
    
def heuristica_gulosa(caminhoes, produtos):
    # Ordenar produtos por valor/volume (ou outro critério)
    produtos_ordenados = sorted(produtos, key=lambda p: p["valor"] / p["volume"], reverse=True)
    
    for caminhao in caminhoes:
        for produto in produtos_ordenados[:]:  # Use uma cópia da lista para evitar problemas de modificação enquanto itera
            if not produto.get("alocado", False):  # Verifica se o produto já foi alocado
                if (
                    produto["volume"] + caminhao["volume_usado"] <= caminhao["volume"]
                    and produto["peso"] + caminhao["peso_usado"] <= caminhao["peso"]
                ):
                    # Adiciona o produto ao caminhão
                    caminhao["volume_usado"] += produto["volume"]
                    caminhao["peso_usado"] += produto["peso"]
                    caminhao["valor_total"] += produto["valor"]

                    # Marcar o produto como alocado
                    produto["alocado"] = True  # Agora ele está alocado e não será considerado novamente
                    
                    # Remover produto alocado da lista de produtos
                    produtos_ordenados.remove(produto)
                

    # Certifique-se de que a função retorne os caminhões
    return caminhoes

