# Detecção de fraude em transações de cartão de crédito

Projeto em Python que compara regressão logística, Random Forest e XGBoost em uma base altamente desbalanceada. O notebook contém as tabelas de resultados, as curvas ROC e precisão–recall, a matriz de confusão e explicações SHAP **com as saídas da execução salvas**.

## Problema e dados

Na base da [ULB/Worldline usada pelo tutorial do TensorFlow](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data), há **284.807 transações**, das quais **492 são fraudes (0,1727%)**. Classificar tudo como normal produziria **99,8273% de acurácia**, mas **recall de fraude igual a zero**. Por isso, avalio principalmente o recall da classe 1, acompanhado de precisão, F1, área sob a curva precisão–recall (AP) e número de falsos alertas.

Os dados têm `Time`, `Amount`, `Class` e `V1`–`V28`. Os componentes `V*` são anonimizados por PCA. O notebook carrega o CSV **diretamente por URL** a cada execução; a base não está neste projeto. A URL pública é a indicada no [exemplo oficial do TensorFlow](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv), que utiliza a mesma base do [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Preparação e método

1. Verificação de esquema, rótulos, ausência de valores faltantes e distribuição das classes.
2. Criação de `LogAmount = log1p(Amount)` e de `TimeSin`/`TimeCos` com período de 24 h relativo ao início da coleta. `Time` e `Amount` originais permanecem disponíveis. A fase temporal não representa, necessariamente, a hora civil local.
3. Divisão aleatória **estratificada**: treino com 170.883 transações (295 fraudes), validação com 56.962 (99 fraudes) e teste com 56.962 (98 fraudes). O `StandardScaler` de cada pipeline é ajustado exclusivamente no treino correspondente.
4. Treinamento com pesos de classe: regressão logística (`balanced`), Random Forest (`balanced_subsample`) e XGBoost (`scale_pos_weight` calculado no treino). Como comparação adicional, regressões logísticas com *undersampling* de normais e *oversampling* por duplicação de fraudes. A reamostragem ocorre **só no treino**.
5. Para **cada modelo**, escolha do limiar na validação que maximiza recall entre os limiares com **precisão ≥ 50%**. Empates são resolvidos pela maior precisão. O modelo principal é escolhido pela mesma regra na validação. O teste é avaliado uma única vez, sem alterar modelo ou limiar com base nos resultados.

O limite de 50% é um cenário didático de carga de investigação. Em produção, precisaria ser definido com os custos de fraude, o custo de analisar alertas e a capacidade operacional. Pesos de classe e reamostragem podem afetar a calibração; os números produzidos pelos modelos são tratados como pontuações para ordenar alertas.

## Comparação no teste

Os limiares desta tabela foram escolhidos **na validação**; a coluna “Selecionado” registra a decisão tomada antes de observar o teste. As métricas de precisão, recall e F1 referem-se à **classe fraude (1)**.

| Modelo | Limiar | Recall | Precisão | F1 | AP (PR-AUC) | TP | FP | Selecionado |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Regressão logística com pesos | 0,995463 | 84,69% | 58,04% | 68,88% | 0,7206 | 83 | 60 | |
| Random Forest com pesos | 0,149421 | 86,73% | 55,56% | 67,73% | 0,8359 | 85 | 68 | |
| **XGBoost com pesos** | **0,189879** | **86,73%** | **52,47%** | **65,38%** | **0,8629** | **85** | **77** | **Sim** |
| Logística com undersampling | 0,927377 | 84,69% | 47,43% | 60,81% | 0,6862 | 83 | 92 | |
| Logística com oversampling | 0,726363 | 86,73% | 52,15% | 65,13% | 0,7498 | 85 | 78 | |

Na **validação**, o XGBoost atingiu recall **82,83%**, precisão **56,16%**, F1 **66,94%** e AP **0,8062**, obtendo o maior recall entre os cinco candidatos com a restrição de precisão. No **teste**, com limiar **0,189879**, detectou **85 das 98 fraudes**; deixou **13** sem alerta e emitiu **77 falsos alertas**. Sua ROC-AUC no teste foi **0,9774**. O fato de o *undersampling* cair abaixo dos 50% de precisão no teste mostra que o piso definido na validação não é uma garantia para dados novos.

O Random Forest empatou em recall com o XGBoost no teste e teve precisão maior; **não foi escolhido após observar esse empate**, pois a seleção já estava fechada pela validação.

## Interpretação com SHAP

O notebook mostra importância das variáveis do modelo escolhido, um gráfico SHAP agregado em **120 transações escolhidas para incluir até 30 fraudes** e a decomposição da pontuação de uma transação sinalizada. Esse gráfico agregado descreve a amostra explicada, que deliberadamente tem mais fraudes que a população. Pela importância nativa do XGBoost, `V14`, `V10` e `V4` aparecem no topo. Na transação de índice original **77348** (fraude real, alerta emitido, pontuação **0,999899**), as maiores contribuições SHAP positivas são `V14` (**+3,493**), `V12` (**+1,554**) e `V10` (**+1,504**); `V8` reduz a pontuação (**−0,795**). Contribuições SHAP positivas elevam a pontuação de fraude em relação à referência; no XGBoost elas aparecem na escala de *log-odds*. `V1`–`V28` não revelam as variáveis originais; importância estatística não demonstra a causa da fraude.

## O que acrescentei ao roteiro do desafio

Em relação às etapas **descritas no enunciado** (não a uma comparação literal com as aulas, cujo link não foi fornecido): separei uma validação estratificada exclusiva para escolher limiares e modelo, defini um piso de precisão explícito, comparei as duas estratégias de reamostragem, criei variáveis cíclicas de tempo e incluí SHAP global e local para o modelo selecionado. Mantive as curvas e tabelas no notebook para que o resultado possa ser conferido sem reexecutá-lo.

## Executar

Requer Python 3.12 e conexão com a internet para baixar o CSV. No terminal, dentro desta pasta:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Abra [`fraude_cartao_credito.ipynb`](fraude_cartao_credito.ipynb) no VS Code, Jupyter ou Colab e execute as células na ordem. O treinamento pode levar alguns minutos, dependendo do computador. O arquivo [`requirements.txt`](requirements.txt) registra as versões utilizadas e [`.gitignore`](.gitignore) exclui o CSV e arquivos temporários.

## Limitações

Há somente 492 fraudes, e a divisão aleatória estratificada pode otimizar a estimativa em relação a uma implantação futura. Seria importante avaliar uma **separação temporal**, estabilidade em outras amostras, custos operacionais, calibração e monitoramento. Sem identificador de cliente/cartão, não é possível construir variáveis de comportamento individual ao longo do tempo. O resultado é um estudo de portfólio, não um sistema pronto para aprovar ou bloquear transações.
