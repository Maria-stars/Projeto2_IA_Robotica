# Projeto 2 — Taxi-v4 compacto

Três arquivos Python implementam Q-learning, SARSA e os experimentos dos itens
1 a 5. São usados os mesmos hiperparâmetros para os dois algoritmos e ambientes.
Não há mascaramento de ações. Requer Python 3.11 ou superior.

## Executar

Extraia o ZIP. No terminal, entre na pasta Projeto2_Compacto e execute:

```powershell
python -m pip install -r requirements.txt
python executar.py
```

O comando faz 20 treinamentos: 5 repetições × 2 algoritmos × 2 ambientes,
com 20.000 episódios por treinamento. Depois avalia as tabelas e salva gráficos.
Os gráficos são arquivos PNG; nenhuma janela precisa ficar aberta.
O treinamento completo pode demorar alguns minutos, dependendo do computador.

Para verificar rapidamente a instalação (não serve como resultado final):

```powershell
python executar.py --episodios 40 --execucoes 1 --saida resultados_teste
```

Para executar novamente só as avaliações, sem treinar:

```powershell
python executar.py --avaliar
```

Para refazer apenas os gráficos:

```powershell
python graficos.py
```

Esses dois comandos exigem os resultados de um treinamento anterior. A execução
normal sobrescreve arquivos da mesma configuração em resultados/. Para comparar
configurações sem sobrescrever, use --saida com outra pasta. Não copie a pasta
.venv de outro computador; instale as dependências com requirements.txt.

## Arquivos necessários

| Arquivo | Função |
|---|---|
| agentes.py | Escolha de ações, treinamento Q-learning/SARSA e avaliação |
| executar.py | Hiperparâmetros, cinco repetições, gravação dos dados e resumos |
| graficos.py | Leitura dos históricos e geração dos gráficos |
| requirements.txt | Três dependências diretas |
| README.md | Instruções e interpretação |
| LICENSE | Licença preservada do projeto original |

Os únicos scripts executáveis diretamente são executar.py e graficos.py.
agentes.py contém funções chamadas por executar.py.

## Epsilon: o valor real é 0.9995

No original, Config.epsilon_decay = 0.9995. O notebook usava formatação com
três casas decimais, exibindo 1.000. Isso não significa que o decaimento real seja 1.
Mantivemos a configuração e agora a salvamos sem esse arredondamento:

- epsilon_start = 1.0: inicialmente, todas as escolhas são exploratórias.
- epsilon_min = 0.005: no fim, há 0,5% de exploração.
- epsilon_decay = 0.9995: multiplicação por episódio.

Fórmula: epsilon(i) = max(0.005, 1.0 × 0.9995**i), com i começando em zero.
Após 1.000 episódios: aproximadamente 0.6065; após 5.000: 0.0820;
após 10.000: 0.00673. O mínimo passa a ser usado no episódio 10.595.
A numeração inclui o primeiro episódio, que usa i=0.

Essa escolha faz sentido para os 20.000 episódios originais. Com apenas 5.000,
o agente ainda explora cerca de 8,2% no fim: não é erro, mas é outro cronograma.
Se o valor fosse exatamente 1.0, o epsilon nunca diminuiria.
Não há garantia de convergência só por escolher esses números.

Também conservamos alpha=0.6, gamma=0.99 e a redução de alpha por visita do
original: alpha efetivo = 0.6 / (1 + 0.01 × (visitas - 1))**0.6.
O cálculo aparece explicitamente em agentes.py. Para estudar uma taxa constante,
basta usar alpha_decay=0.0 em CONFIG; isso muda o experimento e exige novo treino.
Todas as configurações ficam em CONFIG, no começo de executar.py.

## Correspondência com o enunciado

| Item | Dados e figuras gerados |
|---|---|
| 1 | historicos/q_learning_deterministico_1.csv até _5.csv; figura q_learning_deterministico.png |
| 2 | avaliacoes.csv e resumo.csv, treino=deterministico e teste=deterministico |
| SARSA | Mesmos arquivos com prefixo sarsa, mantendo os hiperparâmetros |
| 3 | comparacao_deterministico.png, duas curvas médias e faixa entre repetições |
| 4 | avaliacoes.csv e resumo.csv, treino=deterministico e teste=chuva |
| 5 | Históricos de treino com chuva, figuras individuais e comparacao_chuva.png; avaliação treino=chuva e teste=chuva |

Cada tabela é avaliada em 100 tentativas por condição. Assim, são 100 tentativas
para cada uma das cinco repetições, não uma única média que mistura tudo.
São 3.000 avaliações ao todo: 2 algoritmos × 5 tabelas × 3 condições de teste.
Para uma apresentação curta, a execução 1 pode ser a representante predefinida;
apresente também a variação entre as cinco, sem escolher pelo melhor teste.

No item 4, nenhuma tabela é retreinada: aplica-se a política aprendida no seco
à chuva. No item 5, novas tabelas começam zeradas e aprendem na chuva.
O ambiente chuvoso usa is_rainy=True, rainy_probability=0.6. O parâmetro indica
60% de chance do movimento pretendido e 20% para cada lateral. Pegar e entregar
continuam determinísticos. A versão fixa Gymnasium 1.3.0 inclui Taxi-v4.

## Como ler os resultados

- Cada linha de um histórico registra episódio, soma das recompensas daquele
  episódio (sem desconto), timesteps e epsilon. A soma reinicia em cada episódio.
- Os gráficos usam média móvel de 100 episódios, identificada no eixo Y. Os CSVs
  preservam os valores brutos. O comparativo usa a média das cinco repetições;
  a faixa é ±1 desvio padrão populacional entre curvas suavizadas, não um
  intervalo de confiança. O painel final amplia os últimos 20% do treinamento.
- A linha +8 é uma referência de reward médio do Taxi determinístico, não o
  melhor episódio observado, nem uma exigência para cada episódio. Não é
  imposta como meta na chuva. A exploração durante o treinamento pode impedir
  que a média móvel chegue a +8 mesmo com uma boa política gulosa final.
- Avaliação usa epsilon=0 e não altera a tabela. Sementes de teste são separadas
  das de treinamento; as mesmas sementes de teste são usadas nos algoritmos.
- O limite de 200 passos é mantido. Truncamentos contam como falhas, não entregas.
  resumo.csv apresenta média e DP dos passos nas entregas e também de todas as
  tentativas, além de sucessos, falhas e reward médio. Todos os DPs de avaliação
  são amostrais (ddof=1); valores sem dados suficientes ficam vazios.
- Se todas as 100 tentativas terminarem, as estatísticas de conclusão cobrem
  as 100 tentativas. Havendo falhas, informe a quantidade junto das médias.

Não incluímos os resultados antigos como se fossem desta versão. executar.py
regenera tudo. As discussões dos itens 3, 4 e 5 devem usar os resultados medidos:
compare velocidade de melhora das curvas, estabilidade entre execuções,
reward final, média/DP dos passos e taxa de sucesso. Se houver falhas, considere-as
antes de afirmar que um agente é mais rápido. Q-learning usa o máximo no alvo;
SARSA usa a próxima ação escolhida pela política, incluindo exploração.

## O que foi simplificado

Retiramos o ambiente virtual, histórico Git, caches, notebook, gerador de notebook,
checkpoints de validação durante o treinamento e dependências de Jupyter/pandas.
Mantivemos a lógica de aprendizado e os hiperparâmetros originais; os resultados
são recalculados. Suprimimos o teste extra chuva -> seco, que não é pedido.
A redução do projeto é estrutural; não prova que uma política seja ótima.

Referências:
- https://gymnasium.farama.org/environments/toy_text/taxi/
- https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
- https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/__init__.py
