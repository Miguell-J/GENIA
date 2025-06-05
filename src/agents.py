import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from typing import List
import random
from matplotlib import cm

D = 64  
N = 10  
AGENTS = 2  
STEPS = 10
K_NEIGHBORS = 5  
key = jax.random.PRNGKey(42)

class MemoryNode:
    def __init__(self, ci, tau, nu, omega):
        self.ci = ci
        self.tau = tau
        self.nu = nu
        self.omega = omega

class SyntheticMemory:
    def __init__(self, nodes):
        self.nodes = nodes

    def to_graph(self):
        G = nx.DiGraph()
        embeddings = jnp.stack([node.ci for node in self.nodes])
        norm_embeddings = embeddings / jnp.linalg.norm(embeddings, axis=1, keepdims=True)
        sims = norm_embeddings @ norm_embeddings.T
        sims = np.array(sims)

        for i, node in enumerate(self.nodes):
            G.add_node(i, nu=node.nu, omega=node.omega, ci=node.ci)
            top_k = sims[i].argsort()[-(K_NEIGHBORS + 1):][::-1]  # inclui ele mesmo
            for j in top_k:
                if i != j:
                    G.add_edge(i, j, weight=sims[i][j])
        return G

def calcular_vetor_atencao(x_t, C, WQ, WK, E):
    Q = WQ @ (x_t + E)
    K = C @ WK.T
    attn_scores = jnp.dot(K, Q) / jnp.sqrt(Q.shape[0])
    attn_scores = attn_scores - jnp.max(attn_scores)
    return jax.nn.softmax(attn_scores)

class WorldModel(hk.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

    def __call__(self, x):
        x = hk.Linear(self.hidden_size)(x)
        x = jax.nn.relu(x)
        return hk.Linear(self.output_size)(x)

def prever_proxima_entrada(params, narrativa, predict_next_fn):
    if narrativa:
        agg = jnp.mean(jnp.stack([node.ci for node in narrativa]), axis=0)
    else:
        agg = jnp.zeros(D)
    return predict_next_fn(params, agg)

def calcular_reforco(pred, x_t, mem):
    erro = jnp.linalg.norm(pred - x_t)
    return jnp.array([erro * abs(node.nu) for node in mem.nodes])

def atualizar_memoria(mem, a_t, reforco):
    new_nodes = []
    for i, node in enumerate(mem.nodes):
        new_nu = node.nu + float(reforco[i]) * float(a_t[i])
        new_omega = node.omega * (1 - float(a_t[i])) + float(a_t[i])
        new_nodes.append(MemoryNode(node.ci, node.tau, new_nu, new_omega))
    return SyntheticMemory(new_nodes)

def construir_narrativa(mem):
    return sorted(mem.nodes, key=lambda x: (x.omega * abs(x.nu)), reverse=True)

class Agent:
    def __init__(self, key, D, N, world_model, predict_next_fn, params):
        self.key = key
        self.E = jax.random.normal(key, (D,))
        self.mem = SyntheticMemory([MemoryNode(
            ci=jax.random.normal(key, (D,)),
            tau=i,
            nu=float(np.random.uniform(-1, 1)),
            omega=1.0) for i in range(N)])
        self.WQ = jax.random.normal(key, (D, D))
        self.WK = jax.random.normal(key, (D, D))
        self.predict_next_fn = predict_next_fn
        self.params = params

    def simular_passo(self, x_t):
        C = jnp.stack([node.ci for node in self.mem.nodes])
        a_t = calcular_vetor_atencao(x_t, C, self.WQ, self.WK, self.E)
        narrativa = construir_narrativa(self.mem)
        pred = prever_proxima_entrada(self.params, narrativa, self.predict_next_fn)
        reforco = calcular_reforco(pred, x_t, self.mem)
        self.mem = atualizar_memoria(self.mem, a_t, reforco)
        return construir_narrativa(self.mem), a_t

def criar_modelo_mundo():
    world_model = hk.without_apply_rng(hk.transform(lambda x: WorldModel(128, D)(x)))
    params = world_model.init(key, jax.random.normal(key, (D,)))
    return world_model, params

def simular_experimento(steps=STEPS):
    world_model, params = criar_modelo_mundo()
    predict_next_fn = world_model.apply
    agentes = [Agent(jax.random.PRNGKey(i), D, N, world_model, predict_next_fn, params) for i in range(AGENTS)]
    resultados = []

    for t in range(steps):
        x_t = jax.random.normal(key, (D,))
        for idx, agente in enumerate(agentes):
            narrativa, _ = agente.simular_passo(x_t)
            resultados.append({"t": t, f"agent_{idx}_top_narrative": [(float(n.ci[0]), float(n.nu), float(n.omega)) for n in narrativa[:3]]})

        if t == steps // 2:
            for i in range(AGENTS - 1):
                idx1 = random.randint(0, N - 1)
                idx2 = random.randint(0, N - 1)
                agentes[i].mem.nodes[idx1], agentes[i+1].mem.nodes[idx2] = agentes[i+1].mem.nodes[idx2], agentes[i].mem.nodes[idx1]

    grafos = [agente.mem.to_graph() for agente in agentes]
    return resultados, grafos

def plotar_grafos(grafos):
    fig, axs = plt.subplots(1, len(grafos), figsize=(6 * len(grafos), 6))
    if len(grafos) == 1:
        axs = [axs]
    for i, G in enumerate(grafos):
        pos = nx.spring_layout(G, seed=42)
        node_colors = [d['omega'] for _, d in G.nodes(data=True)]
        nx.draw(G, pos, with_labels=False, node_color=node_colors, cmap=cm.coolwarm, node_size=40, ax=axs[i], edge_color='#999999')
        axs[i].set_title(f"Affective Graph - Agent {i}")
    plt.tight_layout()
    plt.show()

# Executar experimento
resultados, grafos = simular_experimento()
plotar_grafos(grafos)
pd.DataFrame(resultados).to_csv("results.csv", index=False)
print("Simulation complete.")