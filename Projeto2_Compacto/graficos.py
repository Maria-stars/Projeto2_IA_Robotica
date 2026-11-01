"""Gera novamente os gráficos a partir dos CSVs, sem repetir o treinamento."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROTULOS = {"q_learning": "Q-learning", "sarsa": "SARSA",
           "deterministico": "Determinístico", "chuva": "Chuva (p=0,6)"}


def media_movel(valores, janela):
    janela = min(janela, len(valores))
    return np.arange(janela, len(valores) + 1), np.convolve(
        valores, np.ones(janela) / janela, mode="valid")


def finalizar(ax, chuva, janela):
    ax.set_xlabel("Episódios")
    ax.set_ylabel(f"Reward acumulado por episódio — média móvel ({janela})")
    if not chuva:
        ax.axhline(8, color="black", linestyle="--", label="Referência: reward médio ≥ 8")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def gerar_graficos(pasta):
    pasta = Path(pasta)
    config = json.loads((pasta / "config.json").read_text(encoding="utf-8"))
    figuras = pasta / "figuras"
    figuras.mkdir(exist_ok=True)
    janela = min(config["janela_media"], config["episodios"])
    for cenario in ("deterministico", "chuva"):
        historicos = {}
        for algoritmo in ("q_learning", "sarsa"):
            historicos[algoritmo] = np.array([
                np.loadtxt(pasta / "historicos" / f"{algoritmo}_{cenario}_{i}.csv",
                           delimiter=",", skiprows=1, ndmin=2)[:, 1]
                for i in range(1, config["execucoes"] + 1)
            ])
            fig, ax = plt.subplots(figsize=(10, 5))
            for i, rewards in enumerate(historicos[algoritmo], start=1):
                x, y = media_movel(rewards, janela)
                ax.plot(x, y, label=f"Execução {i}", alpha=0.75)
            ax.set_title(f"{ROTULOS[algoritmo]} — {ROTULOS[cenario]}")
            finalizar(ax, cenario == "chuva", janela)
            fig.tight_layout()
            fig.savefig(figuras / f"{algoritmo}_{cenario}.png", dpi=150)
            plt.close(fig)

        # Um único gráfico compara os algoritmos, usando as cinco execuções.
        fig, eixos = plt.subplots(1, 2, figsize=(14, 5))
        limites_finais = []
        for algoritmo, matriz in historicos.items():
            curvas = np.array([media_movel(r, janela)[1] for r in matriz])
            x = media_movel(matriz[0], janela)[0]
            media, dp = curvas.mean(axis=0), curvas.std(axis=0, ddof=0)
            recorte = x >= max(janela, int(config["episodios"] * 0.8))
            limites_finais.extend((media[recorte] - dp[recorte]).tolist())
            limites_finais.extend((media[recorte] + dp[recorte]).tolist())
            for ax in eixos:
                linha, = ax.plot(x, media, label=ROTULOS[algoritmo])
                ax.fill_between(x, media - dp, media + dp,
                                color=linha.get_color(), alpha=0.15)
        for ax in eixos:
            finalizar(ax, cenario == "chuva", janela)
        eixos[0].set_title("Treinamento completo")
        eixos[1].set_title("Ampliação dos últimos 20% dos episódios")
        eixos[1].set_xlim(max(janela, int(config["episodios"] * 0.8)),
                          max(janela + 1, config["episodios"]))
        # Recalcula apenas a escala visual do painel final; preserva os dados originais.
        if cenario == "deterministico":
            limites_finais.append(8)
        inferior, superior = min(limites_finais), max(limites_finais)
        margem = max(2, (superior - inferior) * 0.1)
        eixos[1].set_ylim(inferior - margem, superior + margem)
        fig.suptitle(f"Comparação — {ROTULOS[cenario]} | faixa: ±1 DP entre execuções")
        fig.tight_layout()
        fig.savefig(figuras / f"comparacao_{cenario}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saida", type=Path,
                        default=Path(__file__).resolve().parent / "resultados")
    gerar_graficos(parser.parse_args().saida)
