import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="TCC COMPAS - Auditoria Multiagente", layout="wide")
st.title("TCC - Auditoria de Fairness com Arquitetura Multiagente (COMPAS)")
st.caption("Painel: métricas do baseline + fairness por grupo + análises dos agentes.")

# ---- Paths robustos (funcionam no Docker) ----
# Este arquivo: /app/app/streamlit_app.py
APP_DIR = Path(__file__).resolve().parent          # /app/app
PROJECT_ROOT = APP_DIR.parent                      # /app
OUT_DIR = PROJECT_ROOT / "outputs"                 # /app/outputs

metrics_path = OUT_DIR / "metrics.json"
race_path = OUT_DIR / "fairness_by_race.csv"
sex_path = OUT_DIR / "fairness_by_sex.csv"

# Onde você disse que ficam os agentes:
AGENTS_DIR = OUT_DIR / "results" / "agents"        # /app/outputs/results/agents
agent_race_path = AGENTS_DIR / "agent_race.md"
agent_sex_path = AGENTS_DIR / "agent_sex.md"
supervisor_path = AGENTS_DIR / "supervisor.md"

# ---- Baseline ----
col1, col2 = st.columns(2)

with col1:
    st.subheader("Métricas do modelo baseline")
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        st.json(metrics)
    else:
        st.info(f"Não encontrei {metrics_path}. Rode: `docker compose run --rm tcc python src/train_eval.py`")

with col2:
    st.subheader("Fairness por grupo")
    tab1, tab2 = st.tabs(["Raça", "Sexo"])

    with tab1:
        if race_path.exists():
            st.dataframe(pd.read_csv(race_path), use_container_width=True)
        else:
            st.info(f"Não encontrei {race_path}.")

    with tab2:
        if sex_path.exists():
            st.dataframe(pd.read_csv(sex_path), use_container_width=True)
        else:
            st.info(f"Não encontrei {sex_path}.")

st.divider()

# ---- Agentes ----
st.subheader("Resultados e Discussão (Camada Multiagente)")

# Ajuda rápida de diagnóstico de path (muito útil em Docker)
with st.expander("Diagnóstico de caminhos (Docker)"):
    st.write("PROJECT_ROOT:", str(PROJECT_ROOT))
    st.write("OUT_DIR:", str(OUT_DIR))
    st.write("AGENTS_DIR:", str(AGENTS_DIR))
    st.write("AGENTS_DIR existe?", AGENTS_DIR.exists())
    if AGENTS_DIR.exists():
        st.write("Arquivos em AGENTS_DIR:", [p.name for p in AGENTS_DIR.glob("*")])

tabA, tabB, tabC = st.tabs(["Agente (Raça)", "Agente (Sexo)", "Supervisor (Síntese)"])

with tabA:
    if agent_race_path.exists():
        st.markdown(agent_race_path.read_text(encoding="utf-8"))
    else:
        st.warning(f"Não encontrei {agent_race_path}.")

with tabB:
    if agent_sex_path.exists():
        st.markdown(agent_sex_path.read_text(encoding="utf-8"))
    else:
        st.warning(f"Não encontrei {agent_sex_path}.")

with tabC:
    if supervisor_path.exists():
        st.markdown(supervisor_path.read_text(encoding="utf-8"))
    else:
        st.warning(f"Não encontrei {supervisor_path}.")

st.divider()
st.subheader("Como gerar resultados (Docker)")
st.code(
    "docker compose run --rm tcc python src/train_eval.py\n"
    "# se você tiver um script para agentes, rode também:\n"
    "# docker compose run --rm tcc python src/run_agents.py\n"
    "docker compose up --build",
    language="bash",
)
