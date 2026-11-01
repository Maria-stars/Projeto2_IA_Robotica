"""Aprendizado e avaliação. Q-learning e SARSA usam os mesmos parâmetros."""
import gymnasium as gym
import numpy as np


def criar_ambiente(chuva=False):
    # Mantém o limite de 200 passos. Nenhuma animação durante os experimentos.
    return gym.make("Taxi-v4", is_rainy=chuva, rainy_probability=0.6)


def escolher_acao(q, estado, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(q.shape[1]))
    # Sorteia entre ações empatadas para não favorecer sempre a ação 0.
    melhores = np.flatnonzero(q[estado] == q[estado].max())
    return int(rng.choice(melhores))


def treinar(algoritmo, chuva, config, seed):
    """Uma execução independente: retorna Q-table e histórico por episódio.

    Colunas do histórico: episódio, reward acumulado, timesteps e epsilon.
    """
    if algoritmo not in ("q_learning", "sarsa"):
        raise ValueError("Algoritmo deve ser q_learning ou sarsa.")
    env = criar_ambiente(chuva)
    rng = np.random.default_rng(seed)
    q = np.zeros((env.observation_space.n, env.action_space.n))
    visitas = np.zeros_like(q, dtype=int)
    historico = np.zeros((config["episodios"], 4))
    try:
        for episodio in range(config["episodios"]):
            estado, _ = env.reset(seed=seed if episodio == 0 else None)
            epsilon = max(config["epsilon_min"],
                          config["epsilon_start"] * config["epsilon_decay"] ** episodio)
            acao = escolher_acao(q, estado, epsilon, rng)
            total_reward = 0.0
            passos = 0
            while True:
                proximo, reward, terminou, truncou, _ = env.step(acao)
                proxima_acao = None
                alvo = reward
                if not terminou:
                    if algoritmo == "q_learning":
                        alvo += config["gamma"] * q[proximo].max()
                    else:
                        # SARSA aprende com a ação escolhida pela política e a executa.
                        proxima_acao = escolher_acao(q, proximo, epsilon, rng)
                        alvo += config["gamma"] * q[proximo, proxima_acao]

                # Conserva a taxa por visita usada no projeto original.
                visitas[estado, acao] += 1
                alpha = config["alpha"] / (
                    1 + config["alpha_decay"] * (visitas[estado, acao] - 1)
                ) ** config["alpha_power"]
                q[estado, acao] += alpha * (alvo - q[estado, acao])
                total_reward += reward
                passos += 1
                # Truncamento encerra a coleta, mas não elimina o valor futuro do alvo.
                if terminou or truncou:
                    break
                estado = proximo
                acao = (proxima_acao if algoritmo == "sarsa"
                        else escolher_acao(q, estado, epsilon, rng))
            historico[episodio] = (episodio + 1, total_reward, passos, epsilon)
    finally:
        env.close()
    return q, historico


def avaliar(q, chuva, tentativas=100):
    """Sem exploração nem aprendizado. Cada linha é uma tentativa de teste."""
    env = criar_ambiente(chuva)
    registros = []
    try:
        for tentativa in range(tentativas):
            # Mesmas sementes nos algoritmos; diferentes das sementes de treinamento.
            seed = 1_000_000 + tentativa
            estado, _ = env.reset(seed=seed)
            total_reward = 0.0
            passos = 0
            while True:
                acao = int(np.argmax(q[estado]))
                estado, reward, terminou, truncou, _ = env.step(acao)
                total_reward += reward
                passos += 1
                if terminou or truncou:
                    break
            registros.append({"tentativa": tentativa + 1, "seed": seed,
                              "timesteps": passos, "reward": total_reward,
                              "sucesso": int(terminou), "truncado": int(truncou)})
    finally:
        env.close()
    return registros
