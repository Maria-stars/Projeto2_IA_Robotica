"""Executa o Projeto 2. Configurações ficam no dicionário CONFIG abaixo."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from agentes import treinar, avaliar
from graficos import gerar_graficos

CONFIG = {
    "episodios": 20000,
    "execucoes": 5,
    "tentativas": 100,
    "alpha": 0.6,
    "alpha_decay": 0.01,
    "alpha_power": 0.6,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.005,
    "epsilon_decay": 0.9995,  # Não é 1.000: o notebook original arredondava.
    "seed": 2026,
    "janela_media": 100,
}
PASTA = Path(__file__).resolve().parent / "resultados"


def salvar_csv(caminho, linhas):
    with caminho.open("w", newline="", encoding="utf-8-sig") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)


def avaliar_tabelas(pasta, config):
    """100 testes por tabela e condição, sem escolher a melhor usando o teste."""
    detalhes, resumos = [], []
    for treino in ("deterministico", "chuva"):
        # Itens 2/4: seco -> seco/chuva. Item 5: chuva -> chuva.
        testes = ("deterministico", "chuva") if treino == "deterministico" else ("chuva",)
        for algoritmo in ("q_learning", "sarsa"):
            for execucao in range(1, config["execucoes"] + 1):
                nome = f"{algoritmo}_{treino}_{execucao}"
                q = np.load(pasta / "tabelas" / f"{nome}.npy", allow_pickle=False)
                for teste in testes:
                    registros = avaliar(q, teste == "chuva", config["tentativas"])
                    chave = {"algoritmo": algoritmo, "treino": treino,
                             "teste": teste, "execucao": execucao}
                    detalhes.extend([{**chave, **r} for r in registros])
                    todos = [r["timesteps"] for r in registros]
                    concluidos = [r["timesteps"] for r in registros if r["sucesso"]]
                    # Não confunde 200 passos de uma falha com tempo de conclusão.
                    resumos.append({**chave, "tentativas": len(registros),
                        "sucessos": len(concluidos), "falhas": len(todos) - len(concluidos),
                        "media_passos_sucessos": float(np.mean(concluidos)) if concluidos else "",
                        "dp_passos_sucessos": float(np.std(concluidos, ddof=1)) if len(concluidos) > 1 else "",
                        "media_passos_todas": float(np.mean(todos)),
                        "dp_passos_todas": float(np.std(todos, ddof=1)) if len(todos) > 1 else "",
                        "reward_medio": float(np.mean([r["reward"] for r in registros]))})
    salvar_csv(pasta / "avaliacoes.csv", detalhes)
    salvar_csv(pasta / "resumo.csv", resumos)
    for r in resumos:
        media = r["media_passos_sucessos"]
        texto = f"{media:.2f}" if media != "" else "sem entregas"
        print(f'{r["algoritmo"]} | {r["treino"]} -> {r["teste"]} | '
              f'execução {r["execucao"]}: {r["sucessos"]}/{r["tentativas"]} entregas, '
              f'média de passos nas entregas: {texto}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--avaliar", action="store_true", help="Reavalia tabelas salvas sem treinar")
    parser.add_argument("--episodios", type=int, default=CONFIG["episodios"])
    parser.add_argument("--execucoes", type=int, default=CONFIG["execucoes"])
    parser.add_argument("--saida", type=Path, default=PASTA)
    args = parser.parse_args()
    pasta = args.saida.resolve()
    if args.avaliar:
        config = json.loads((pasta / "config.json").read_text(encoding="utf-8"))
    else:
        if args.episodios < 1 or args.execucoes < 1:
            parser.error("episodios e execucoes devem ser positivos")
        config = {**CONFIG, "episodios": args.episodios, "execucoes": args.execucoes}
        for subpasta in ("tabelas", "historicos", "figuras"):
            (pasta / subpasta).mkdir(parents=True, exist_ok=True)
        (pasta / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        for cenario in ("deterministico", "chuva"):
            for algoritmo in ("q_learning", "sarsa"):
                for execucao in range(1, config["execucoes"] + 1):
                    print(f"Treinando {algoritmo} | {cenario} | {execucao}/{config['execucoes']}", flush=True)
                    q, historico = treinar(algoritmo, cenario == "chuva", config,
                                           config["seed"] + execucao - 1)
                    nome = f"{algoritmo}_{cenario}_{execucao}"
                    np.save(pasta / "tabelas" / f"{nome}.npy", q)
                    np.savetxt(pasta / "historicos" / f"{nome}.csv", historico,
                               delimiter=",", header="episodio,reward,timesteps,epsilon",
                               comments="", fmt=["%d", "%.6f", "%d", "%.8f"])
    avaliar_tabelas(pasta, config)
    gerar_graficos(pasta)
    print(f"\nConcluído. Resultados em: {pasta}")


if __name__ == "__main__":
    main()
