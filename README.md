# Análise de Imparcialidade por Meio de uma Arquitetura Multiagente
## Um Estudo Experimental com o COMPAS

Este repositório contém a implementação experimental do Trabalho de Conclusão de Curso (TCC) que investiga a análise de fairness em modelos de aprendizado de máquina utilizando uma arquitetura multiagente aplicada ao dataset COMPAS.

---

## Objetivo

- Treinar um classificador baseline
- Avaliar métricas de desempenho
- Calcular métricas de fairness por grupo
- Interpretar resultados com agentes
- Visualizar resultados via Streamlit

---

## Como reproduzir (Docker)

### 1. Clonar repositório
git clone <repo>
cd <repo>

### 2. Definir chave OpenAI
export OPENAI_API_KEY=xxxx

### 3. Rodar baseline
docker compose run --rm tcc python src/train_eval.py

### 4. Rodar agentes
docker compose run --rm tcc python src/run_agents.py

### 5. Abrir Streamlit
docker compose up --build

Abrir: http://localhost:8501

---

## Estrutura

app/
src/
data/
outputs/
outputs/results/agents/

---

## Observação

Os agentes não alteram o modelo. Funcionam como camada de interpretação.
