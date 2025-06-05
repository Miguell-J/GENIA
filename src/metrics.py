import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List
import pandas as pd

D = 64
N = 200
T = 100
key = jax.random.PRNGKey(42)

class MemoryNode:
    def __init__(self, ci: jnp.ndarray, tau: int, nu: float, omega: float):
        self.ci = ci
        self.tau = tau
        self.nu = nu
        self.omega = omega

class SyntheticMemory:
    def __init__(self, nodes: List[MemoryNode]):
        self.nodes = nodes

    def get_graph(self):
        G = nx.DiGraph()
        for i, node in enumerate(self.nodes):
            G.add_node(i, nu=node.nu, omega=node.omega)
            for j, other in enumerate(self.nodes):
                if i != j:
                    sim = jnp.dot(node.ci, other.ci) / (jnp.linalg.norm(node.ci) * jnp.linalg.norm(other.ci))
                    if sim > 0.9:
                        G.add_edge(i, j, weight=float(sim))
        return G

def calcular_vetor_atencao(x_t, C, WQ, WK, E):
    Q = WQ @ (x_t + E)
    K = C @ WK.T
    attn_scores = jnp.dot(K, Q) / jnp.sqrt(Q.shape[0])
    attn_scores -= jnp.max(attn_scores)
    return jax.nn.softmax(attn_scores)

class WorldModel(hk.Module):
    def __init__(self, hidden_size: int, output_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = hk.Linear(self.hidden_size)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(self.output_size)(x)
        return x

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
    return sorted(mem.nodes, key=lambda x: (x.omega, abs(x.nu)), reverse=True)

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
        self.history = []
        self.narrativas = []

    def simular_passo(self, x_t):
        C = jnp.stack([node.ci for node in self.mem.nodes])
        a_t = calcular_vetor_atencao(x_t, C, self.WQ, self.WK, self.E)
        narrativa = construir_narrativa(self.mem)
        pred = prever_proxima_entrada(self.params, narrativa, self.predict_next_fn)
        reforco = calcular_reforco(pred, x_t, self.mem)
        self.mem = atualizar_memoria(self.mem, a_t, reforco)
        narrativa = construir_narrativa(self.mem)
        self.narrativas.append(narrativa)
        self.history.append([(float(nu), float(omega)) for nu, omega in zip(a_t, reforco)])
        return narrativa

def criar_modelo_mundo():
    world_model = hk.without_apply_rng(hk.transform(lambda x: WorldModel(128, D)(x)))
    params = world_model.init(key, jax.random.normal(key, (D,)))
    return world_model, params

def calcular_metricas(agent):
    metricas = []
    for t, narrativa in enumerate(agent.narrativas):
        if narrativa:
            coerencia = sum(n.omega * abs(n.nu) for n in narrativa[:5])
            entropia = -sum(n.omega * np.log(n.omega + 1e-8) for n in narrativa[:5])
        else:
            coerencia = 0
            entropia = 0
        metricas.append((t, coerencia, entropia))
    return pd.DataFrame(metricas, columns=["Tempo", "Narrative Coherence", "Narrative Entropy"])

def simular_agentes():
    world_model, params = criar_modelo_mundo()
    predict_next_fn = world_model.apply
    agent = Agent(jax.random.PRNGKey(0), D, N, world_model, predict_next_fn, params)

    for t in range(T):
        x_t = jax.random.normal(jax.random.PRNGKey(t), (D,))
        agent.simular_passo(x_t)

    G = agent.mem.get_graph()
    pos = nx.spring_layout(G)
    node_colors = [G.nodes[i]['nu'] for i in G.nodes()]
    node_sizes = [300 + 500 * G.nodes[i]['omega'] for i in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=node_sizes, cmap=plt.cm.coolwarm)
    plt.title("Final Affective Graph of the Agent")
    plt.show()

    df_ref = pd.DataFrame([dict(enumerate([omega for _, omega in step])) for step in agent.history])
    df_ref.plot(title="Evolution of Reinforcements per Node")
    plt.xlabel("Time")
    plt.ylabel("Reinforcement")
    plt.legend(title="Node")
    plt.show()

    df_metrics = calcular_metricas(agent)
    df_metrics.set_index("Tempo").plot(title="Narrative Metrics")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    simular_agentes()