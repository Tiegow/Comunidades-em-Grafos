import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import defaultdict

# =============================================================================
# FUNÇÕES AUXILIARES DO ALGORITMO
# =============================================================================

# Agrupa nós por ID de comunidade. Retorna um dicionário {cid: {nós}}.
def agrupar_por_comunidade(particao):
    mapa_comunidades = defaultdict(set)
    for no, id_comunidade in particao.items():
        mapa_comunidades[id_comunidade].add(no)
    return mapa_comunidades

# Calcula a modularidade de cada partição de um grafo.
def calcular_modularidade(G, particao):
    comunidades = agrupar_por_comunidade(particao)

    # 'm' é o número total de arestas no grafo. Se o grafo for ponderado,
    # G.size(weight='weight') soma o peso de todas as arestas.
    m = G.size(weight='weight')
    if m == 0:
        return 0.0

    modularidade_total = 0.0
    for id_comunidade, nos_comunidade in comunidades.items():
        soma_graus = sum(G.degree(no, weight='weight') for no in nos_comunidade)
        subgrafo = G.subgraph(nos_comunidade)
        peso_arestas_internas = subgrafo.size(weight='weight')
        
        frac_interna = peso_arestas_internas / m
        frac_graus = (soma_graus / (2 * m))
        
        modularidade_total += (frac_interna - frac_graus**2)
        
    return modularidade_total

# Função auxiliar para calcular as propriedades de uma única comunidade.
def calc_propriedades(G, nodes_in_community):
    # Soma dos graus de todos os nós na comunidade
    grau_total = sum(G.degree(node, weight='weight') for node in nodes_in_community)
    
    # Soma dos pesos das arestas INTERNAS à comunidade
    subgraph = G.subgraph(nodes_in_community)
    peso_total = subgraph.size(weight='weight')
    
    return grau_total, peso_total

# Calcula o ganho de modularidade (Delta Q) ao mover um nó para uma nova comunidade.
def calc_delta_q(G, particao, no_a_mover, id_comunidade_alvo):
    m = G.size(weight='weight')
    if m == 0:
        return 0.0

    id_comunidade_atual = particao[no_a_mover]

    # Se a comunidade alvo é a mesma, o ganho é zero.
    if id_comunidade_atual == id_comunidade_alvo:
        return 0.0

    # Pega os nós das duas comunidades relevantes (a atual e a alvo)
    nos_em_alvo = {n for n, c in particao.items() if c == id_comunidade_alvo}
    nos_em_atual = {n for n, c in particao.items() if c == id_comunidade_atual}

    # Calcula as propriedades ANTES da movimentação
    grau_comun_atual, peso_comun_atual = calc_propriedades(G, nos_em_atual)
    grau_alvo_atual, peso_alvo_atual = calc_propriedades(G, nos_em_alvo)

    # Contribuição para Q das duas comunidades ANTES da mudança
    contrib_antes = (peso_comun_atual / m - (grau_comun_atual / (2 * m))**2) + \
                    (peso_alvo_atual / m - (grau_alvo_atual / (2 * m))**2)

    # Calcula as propriedades DEPOIS da movimentação hipotética
    no_k = G.degree(no_a_mover, weight='weight')
    
    # Calcula k_{u,C} e k_{u,D}: soma dos pesos das arestas do nó para cada comunidade
    no_k_para_atual = 0
    no_k_para_alvo = 0
    for vizinho, data in G.adj[no_a_mover].items():
        peso = data.get('weight', 1.0)
        if particao[vizinho] == id_comunidade_atual:
            no_k_para_atual += peso
        if particao[vizinho] == id_comunidade_alvo:
            no_k_para_alvo += peso

    # Propriedades da comunidade ATUAL depois que o nó sai
    novo_grau_comun = grau_comun_atual - no_k
    novo_peso_comun = peso_comun_atual - no_k_para_atual
    
    # Propriedades da comunidade ALVO depois que o nó entra
    novo_grau_alvo = grau_alvo_atual + no_k
    novo_peso_alvo = peso_alvo_atual + no_k_para_alvo

    # Contribuição para Q das duas comunidades DEPOIS da mudança
    contrib_depois = (novo_peso_comun / m - (novo_grau_comun / (2 * m))**2) + \
                    (novo_peso_alvo / m - (novo_grau_alvo / (2 * m))**2)
    
    return contrib_depois - contrib_antes

# Desenha o grafo, colorindo os nós de acordo com as comunidades.
def desenhar_grafo_comunidades(G, comunidades, titulo):
    comunidades_unicas = set(comunidades.values())
    cores = plt.cm.get_cmap('viridis', len(comunidades_unicas))
    mapa_cores = {}
    for i, id_comunidade in enumerate(comunidades_unicas):
        mapa_cores[id_comunidade] = cores(i)
    cores_no = [mapa_cores.get(comunidades[node], 'black') for node in G.nodes()]
    pos = nx.spring_layout(G, seed=42)
    plt.figure(num=titulo, figsize=(12, 9))
    nx.draw(
        G, pos,
        with_labels=False,
        node_color=cores_no,
        cmap=plt.cm.viridis,
        edge_color='gray',
        node_size=500,
        linewidths=1,           
        edgecolors='black'      
    )
    plt.title(titulo, fontsize=16)
    plt.show()

# =============================================================================
# FUNÇÕES CENTRAIS DO ALGORITMO
# =============================================================================

# Fase de Movimentação Local. 
# Seu objetivo é, de forma iterativa, melhorar a modularidade da partição
# movendo nós individuais entre comunidades até que nenhuma melhoria adicional seja possível.
def encontrar_movimentacao_local(H, G_completo, particao_inicial):
    comunidades = particao_inicial.copy()
    melhorou = True
    
    while melhorou:
        melhorou = False
        lista_nos = list(H.nodes())
        random.shuffle(lista_nos)
        
        for no in lista_nos:
            comunidade_atual = comunidades[no]
            melhor_comunidade = comunidade_atual
            melhor_delta_q = 0.0 # Começamos com um ganho de zero

            # Itera sobre os vizinhos para encontrar a melhor comunidade para se mover
            vizinhos_comunidades = {comunidades[vizinho] for vizinho in H.neighbors(no)}
            
            for comunidade_vizinho in vizinhos_comunidades:
                if comunidade_atual != comunidade_vizinho:
                    
                    # Calcula o ganho de modularidade de forma eficiente
                    delta_q = calc_delta_q(G_completo, comunidades, no, comunidade_vizinho)
                    
                    # Se o ganho for o melhor encontrado até agora para este nó, armazena
                    if delta_q > melhor_delta_q:
                        melhor_delta_q = delta_q
                        melhor_comunidade = comunidade_vizinho
            
            # Se encontramos uma movimentação que melhora a modularidade (delta_q > 0),
            # movemos o nó para a melhor comunidade encontrada.
            if melhor_delta_q > 0:
                comunidades[no] = melhor_comunidade
                melhorou = True # Sinaliza que o laço principal deve continuar
                
    return comunidades

# Fase de Refinamento.
# Para cada comunidade, esta função verifica se ela deve ser dividida em sub-comunidades menores e mais densas.
# Recebe o grafo completo e a partição encontrada na fase de movimentação local, que será refinada
def refinar_particao(G, particao_alvo):
    particao_refinada_final = particao_alvo.copy()

    # Prepara um contador para gerar novos IDs de comunidade. Isso é essencial para quando
    # uma comunidade se divide em duas ou mais, para que as novas não tenham IDs conflitantes.
    novo_id_comunidade = max(particao_alvo.values()) + 1 if particao_alvo else 0

    # Itera sobre cada comunidade encontrada na fase anterior.
    ids_comunidades = set(particao_alvo.values())
    for id_comunidade in ids_comunidades:

        # --- ETAPA 1: ISOLAR A COMUNIDADE ---
        # Pega todos os nós que pertencem à comunidade que estamos analisando agora.
        nos_comunidade = [no for no, cid in particao_alvo.items() if cid == id_comunidade]
        if len(nos_comunidade) <= 1:
            continue
        subgrafo = G.subgraph(nos_comunidade)

        # --- ETAPA 2: DESMONTAR E TESTAR A COESÃO ---
        # Prepara uma partição temporária para o teste. Começa como uma cópia da partição completa.
        particao_unica = particao_alvo.copy()

        # "Desmonta" a comunidade: dentro da partição temporária, cada nó da comunidade alvo é colocado em sua própria comunidade individual.
        for no in nos_comunidade:
            particao_unica[no] = no

        # Roda o algoritmo de movimentação local sob condições especiais:
        # 1. 'subgrafo': Os nós só podem considerar se mover para comunidades de vizinhos que estão DENTRO do subgrafo.
        # 2. 'G': O cálculo de modularidade ainda usa o grafo COMPLETO como referência.
        # 3. 'particao_unica': O ponto de partida é o estado "desmontado".
        # O resultado nos dirá se os nós se reagrupam em um único bloco ou em vários.
        particao_refinada_por_subgrafo = encontrar_movimentacao_local(subgrafo, G, particao_unica)
        mapa_id = {}
        particao_foi_dividida = False

        # --- ETAPA 3: JULGAR O RESULTADO E ATUALIZAR A PARTIÇÃO FINAL ---
        comunidades_locais = {no: cid for no, cid in particao_refinada_por_subgrafo.items() if no in nos_comunidade}

        # Contamos quantas sub-comunidades foram formadas.
        # Se for mais de 1, a comunidade original não era coesa e precisa ser dividida.
        if len(set(comunidades_locais.values())) > 1:
            particao_foi_dividida = True
        for no, cid_local in comunidades_locais.items():
            if particao_foi_dividida:
                # Se a comunidade foi dividida, precisamos criar novos IDs globais para as novas sub-comunidades.                
                if cid_local not in mapa_id:
                    mapa_id[cid_local] = novo_id_comunidade
                    novo_id_comunidade += 1
                particao_refinada_final[no] = mapa_id[cid_local]
            else:
                # Se a comunidade permaneceu coesa, mantemos o ID original da comunidade.              
                particao_refinada_final[no] = id_comunidade
    return particao_refinada_final

# Fase de Agrupamento.
def agrupar_grafo(G, particao):
    G_agr = nx.Graph()
    G_agr.add_nodes_from(set(particao.values()))
    for u, v, data in G.edges(data=True):
        c_u = particao[u]
        c_v = particao[v]
        peso = data.get('weight', 1.0)
        if G_agr.has_edge(c_u, c_v):
            G_agr[c_u][c_v]['weight'] += peso
        else:
            G_agr.add_edge(c_u, c_v, weight=peso)
    return G_agr

# Função principal que executa o algoritmo de Leiden. O objetivo é encontrar a partição do grafo original que maximiza a modularidade.
def leiden(G_orig):
    # G é o grafo que será processado em cada nível. Ele muda a cada iteração,
    # tornando-se um grafo de "super-nós". Começa como uma cópia do original.
    G = G_orig.copy()

    # Dicionário que rastreia as comunidades através dos níveis.
    # Mapeia os nós do GRAFO ORIGINAL para os nós do grafo G ATUAL.
    no_em_mapa_comunidade = {no: no for no in G.nodes()}

    melhor_particao_geral = {no: no for no in G.nodes()}
    melhor_modularidade = -1.0

    modularidades_por_nivel = []

    # Laço principal do algoritmo. Cada iteração representa a passagem por um nível completo de detecção (movimentação, refinamento e agrupamento).
    while True:
        # --- FASE A: Detecção de Comunidades no Nível Atual ---
        print("\n--- Iniciando novo nível do algoritmo ---")

        # Para o grafo atual 'G', começamos com uma partição onde cada nó está em sua própria comunidade.
        particao_inicial = {no: no for no in G.nodes()}

        # Roda as duas primeiras fases do Leiden no grafo G atual.
        # particao resultante mapeia os nós de G (que podem ser super-nós) para seus novos IDs de comunidade.
        particao = encontrar_movimentacao_local(G, G, particao_inicial)
        particao = refinar_particao(G, particao)
        
        # --- FASE B: Mapeamento da Partição de Volta aos Nós Originais ---
        # Para cada nó original, encontramos seu "super-nó" correspondente no grafo G atual
        particao_atual_mapeada = {}
        for no_original, id_comunidade in no_em_mapa_comunidade.items():
            if id_comunidade in particao:
                 particao_atual_mapeada[no_original] = particao[id_comunidade]

        # --- FASE C: Avaliação e Critério de Parada ---
        modularidade_atual = calcular_modularidade(G_orig, particao_atual_mapeada)
        print(f"Modularidade neste nível: {modularidade_atual:.4f}")

        if modularidade_atual <= melhor_modularidade:
            print("Modularidade não melhorou. Fim do algoritmo.")
            return melhor_particao_geral, modularidades_por_nivel
        
        # --- FASE D: Atualização de Estado e Preparação para o Próximo Nível ---
        melhor_modularidade = modularidade_atual
        melhor_particao_geral = particao_atual_mapeada

        modularidades_por_nivel.append(melhor_modularidade)

        no_em_mapa_comunidade = melhor_particao_geral.copy()

        # Fase de Agrupamento: cria o grafo do próximo nível onde cada nó é uma comunidade da 'particao' que encontramos. (G_novo),
        G_novo = agrupar_grafo(G, particao)
        
        if len(G_novo.nodes()) == len(G.nodes()):
            print("Nenhum agrupamento adicional possível. Fim do algoritmo.")
            return melhor_particao_geral, modularidades_por_nivel
        
        # Atualizamos o grafo de trabalho 'G' para o grafo recém-agregado e o loop 'while' recomeça a partir dele.
        G = G_novo

# Função principal que executa o algoritmo de Louvain.
def louvain(G_orig):
    # G é o grafo que será processado em cada nível. Começa como o original.
    G = G_orig.copy()

    # Mapeia os nós do grafo original para seus IDs de comunidade no nível atual
    no_em_mapa_comunidade = {no: no for no in G_orig.nodes()}

    # Armazena a melhor partição encontrada em todo o processo
    melhor_particao_geral = {no: no for no in G_orig.nodes()}
    melhor_modularidade = -1.0

    modularidades_por_nivel = []

    # Laço principal do algoritmo. Cada iteração é um nível completo.
    while True:
        print("\n--- Iniciando novo nível do algoritmo LOUVAIN ---")

        # --- FASE 1: MOVIMENTAÇÃO LOCAL ---
        # Começamos com cada nó em sua própria comunidade no grafo atual 'G'.
        particao_inicial = {no: no for no in G.nodes()}
        
        # Encontra a melhor partição no nível atual movendo os nós.
        particao = encontrar_movimentacao_local(G, G, particao_inicial)

        # --- MAPEAMENTO DA PARTIÇÃO DE VOLTA AOS NÓS ORIGINAIS ---
        # Traduz a partição do grafo atual (que pode ter super-nós) para os nós originais.
        particao_atual_mapeada = {}
        for no_original, id_comunidade_antigo in no_em_mapa_comunidade.items():
            if id_comunidade_antigo in particao:
                particao_atual_mapeada[no_original] = particao[id_comunidade_antigo]
        
        # --- AVALIAÇÃO E CRITÉRIO DE PARADA ---
        # Calcula a modularidade da nova partição no grafo original.
        modularidade_atual = calcular_modularidade(G_orig, particao_atual_mapeada)
        print(f"Modularidade neste nível: {modularidade_atual:.4f}")

        if modularidade_atual <= melhor_modularidade:
            print("Modularidade não melhorou. Fim do algoritmo.")
            return melhor_particao_geral, modularidades_por_nivel
        
        # --- ATUALIZAÇÃO DE ESTADO E PREPARAÇÃO PARA O PRÓXIMO NÍVEL ---
        melhor_modularidade = modularidade_atual
        melhor_particao_geral = particao_atual_mapeada

        modularidades_por_nivel.append(melhor_modularidade)

        no_em_mapa_comunidade = melhor_particao_geral.copy()

        # --- FASE 2: AGRUPAMENTO ---
        # Cria o grafo do próximo nível, onde cada nó é uma comunidade da partição encontrada.
        G_novo = agrupar_grafo(G, particao)
        
        # Se o agrupamento não mudou o número de nós, não há mais como agregar.
        if len(G_novo.nodes()) == len(G.nodes()):
            print("Nenhum agrupamento adicional possível. Fim do algoritmo.")
            return melhor_particao_geral, modularidades_por_nivel
        
        # Prepara para a próxima iteração com o novo grafo agregado.
        G = G_novo