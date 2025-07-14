import time
import json
import networkx as nx
import leiden_algt as leiden_algt

"""
Gera um grafo de benchmark LFR com comunidades conhecidas.
O grafo gerado é sempre o mesmo, com base nos parâmetros fornecidos (mesma seed).

Args:
    n (int): Número de nós no grafo.
    mu (float): Parâmetro de mistura (quão misturadas são as comunidades).
                    Valores mais altos tornam a detecção mais difícil.

Returns:
    G (nx.Graph): O grafo gerado.
    communities (dict): Um dicionário {nó: id_comunidade} representando a verdade fundamental (ground truth).
"""
def gerar_grafo_lfr(n, mu, min_degree, max_degree):
    # tau1 e tau2 são expoentes para a distribuição de grau e tamanho de comunidade.
    # Valores comuns são 2 e 1, respectivamente.
    # min_degree e max_degree são os graus mínimo e máximo dos nós.
    params = {
        'tau1': 2, 'tau2': 1.1, 'mu': mu,
        'min_degree': min_degree, 'max_degree': max_degree
    }
    
    seed = 42
    G = nx.generators.community.LFR_benchmark_graph(n, **params, seed=seed)

    G.remove_edges_from(nx.selfloop_edges(G))
    
    ground_truth_map = {}
    for node, data in G.nodes(data=True):
        # A informação da comunidade está em data['community'], que é um set.
        # Ex: para o nó 5, pode ser {3}, significando que ele pertence à comunidade 3.
        community_id = list(data['community'])[0]
        ground_truth_map[node] = community_id

    num_comunidades_verdadeiras = len(set(ground_truth_map.values()))
            
    print(f"\nGrafo LFR gerado com {n} nós e mu={mu}.")
    print(f"Número de comunidades 'verdadeiras': {num_comunidades_verdadeiras}")

    params = {'n': n, 'mu': mu, 'max_degree': max_degree, 'min_degree': min_degree, 'seed': seed, 'comunidades_verdadeiras': num_comunidades_verdadeiras}
    
    return G, ground_truth_map, params

def salvar_resultados_json(dados):
    # Gera um nome de arquivo com base na data e hora atuais
    filename = f"experimento_resultados.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        # indent=4 deixa o arquivo bem formatado e legível
        json.dump(dados, f, ensure_ascii=False, indent=4)
        
    print(f"\n>>> Experimento salvo com sucesso no arquivo: {filename}")
    
# --- Carrega o Grafo ---
G, ground_truth, params_g = gerar_grafo_lfr(n=400, mu=0.3, max_degree=15, min_degree=3)

# --- VISUALIZAÇÃO "ANTES" (usando a verdade fundamental para colorir) ---
print("\nMostrando o grafo com as comunidades 'verdadeiras' (Ground Truth)...")
leiden_algt.desenhar_grafo_comunidades(G, ground_truth, "Grafo LFR (Ground Truth)")

# --- EXECUÇÃO COMPLETA DO ALGORITMO ---
print("\nExecutando o algoritmo Leiden...")
leiden_start_time = time.perf_counter()
particao_leiden, modularidades_leiden = leiden_algt.leiden(G)
leiden_end_time = time.perf_counter()

# Calcula e exibe o tempo de execução
leiden_duration = leiden_end_time - leiden_start_time
print(f"\n>>> Tempo de execução do Leiden: {leiden_duration:.4f} segundos")

# --- VISUALIZAÇÃO "DEPOIS" ---
print("\nAlgoritmo executado. Mostrando o grafo com as comunidades encontradas...")
if particao_leiden:
    leiden_algt.desenhar_grafo_comunidades(G, particao_leiden, "Grafo Final (Comunidades Encontradas pelo Leiden)")
    
    modularidade_final = leiden_algt.calcular_modularidade(G, particao_leiden)
    qtd_comunidades_encontradas = len(set(particao_leiden.values()))
    
    print(f"\nModularidade da Partição Final: {modularidade_final:.4f}")
    print(f"Número de comunidades encontradas pelo Leiden: {qtd_comunidades_encontradas}")
else:
    print("O algoritmo não retornou uma partição válida.")

print("\nExecutando o algoritmo Louvain...")
louvain_start_time = time.perf_counter()
particao_louvain, modularidades_louvain = leiden_algt.louvain(G)
louvain_end_time = time.perf_counter()

# Calcula e exibe o tempo de execução
louvain_duration = louvain_end_time - louvain_start_time
print(f"\n>>> Tempo de execução do Louvain: {louvain_duration:.4f} segundos")

# --- VISUALIZAÇÃO "DEPOIS" DO LOUVAIN ---
print("\nMostrando o grafo com as comunidades encontradas pelo Louvain...")
if particao_louvain:
    leiden_algt.desenhar_grafo_comunidades(G, particao_louvain, "Grafo Final (Comunidades Encontradas pelo Louvain)")
    
    modularidade_louvain = leiden_algt.calcular_modularidade(G, particao_louvain)
    qtd_comunidades_louvain = len(set(particao_louvain.values()))
    
    print(f"\nModularidade da Partição Final do Louvain: {modularidade_louvain:.4f}")
    print(f"Número de comunidades encontradas pelo Louvain: {qtd_comunidades_louvain}")
else:
    print("O algoritmo Louvain não retornou uma partição válida.")

# --- Coleta de Dados para Salvar ---
dados_experimento = {
    "parametros_grafo": params_g,
    "resultados_leiden": None,
    "resultados_louvain": None
}

if particao_leiden:
    dados_experimento["resultados_leiden"] = {
        "tempo_execucao_segundos": round(leiden_duration, 4),
        "modularidade_final": leiden_algt.calcular_modularidade(G, particao_leiden),
        "modularidades_por_nivel": modularidades_leiden,
        "num_comunidades": len(set(particao_leiden.values())),
    }

if particao_louvain:
    dados_experimento["resultados_louvain"] = {
        "tempo_execucao_segundos": round(louvain_duration, 4),
        "modularidade_final": leiden_algt.calcular_modularidade(G, particao_louvain),
        "modularidades_por_nivel": modularidades_louvain,
        "num_comunidades": len(set(particao_louvain.values())),
    }

# --- Salvando os resultados em um arquivo JSON ---
salvar_resultados_json(dados_experimento)