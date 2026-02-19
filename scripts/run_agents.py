import json
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI


def load_inputs(metrics_path: Path, race_path: Path, sex_path: Path):
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    df_race = pd.read_csv(race_path)
    df_sex = pd.read_csv(sex_path)
    return metrics, df_race, df_sex


def df_brief(df: pd.DataFrame, group_col: str, top_n: int = 6) -> str:
    cols = [group_col, "N", "FPR", "FNR", "TPR", "TNR"]
    cols = [c for c in cols if c in df.columns]
    tmp = df[cols].copy()
    # Order by FPR then FNR if available
    if "FPR" in tmp.columns:
        tmp = tmp.sort_values("FPR", ascending=False)
    return tmp.head(top_n).to_string(index=False)


def call_llm(client: OpenAI, model: str, system: str, user: str) -> str:
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.output_text


def main():
    metrics_path = Path(os.environ.get("METRICS_JSON", "outputs/metrics.json"))
    race_path    = Path(os.environ.get("FAIRNESS_RACE", "outputs/fairness_by_race.csv"))
    sex_path     = Path(os.environ.get("FAIRNESS_SEX", "outputs/fairness_by_sex.csv"))
    out_dir = Path(os.environ.get("OUT_DIR", "outputs/results")) / "agents"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
    client = OpenAI()

    metrics, df_race, df_sex = load_inputs(metrics_path, race_path, sex_path)

    race_group_col = "race" if "race" in df_race.columns else df_race.columns[-1]
    sex_group_col = "sex" if "sex" in df_sex.columns else df_sex.columns[-1]

    # Build a compact factual summary for agents (grounded in your outputs)
    global_summary = {
        "available_metric_keys": sorted(list(metrics.keys()))[:50],
    }
    for k in ["model", "accuracy", "precision", "recall", "f1", "split", "train_size", "test_size"]:
        if k in metrics:
            global_summary[k] = metrics[k]

    prompt_facts = f"""
FATOS DO EXPERIMENTO (use apenas isto; não invente números):
- Métricas globais (parciais): {json.dumps(global_summary, ensure_ascii=False)}
- Tabela (raça) amostra:
{df_brief(df_race, race_group_col)}
- Tabela (sexo) amostra:
{df_brief(df_sex, sex_group_col)}
"""

    # Agent 1 — Race analysis
    sys1 = "Você é um auditor técnico. Analise fairness por raça com base em FPR/FNR/TPR/TNR. Seja objetivo e cite quais grupos têm maiores/menores taxas. Não invente números."
    usr1 = f"{prompt_facts}\n\nTAREFA: Escreva 6–10 bullets com achados por raça e uma conclusão curta (2–3 frases)."
    race_report = call_llm(client, model, sys1, usr1)
    (out_dir / "agent_race.md").write_text(race_report, encoding="utf-8")

    # Agent 2 — Sex analysis
    sys2 = "Você é um auditor técnico. Analise fairness por sexo usando FPR/FNR/TPR/TNR. Seja objetivo. Não invente números."
    usr2 = f"{prompt_facts}\n\nTAREFA: Escreva 6–10 bullets com achados por sexo e uma conclusão curta (2–3 frases)."
    sex_report = call_llm(client, model, sys2, usr2)
    (out_dir / "agent_sex.md").write_text(sex_report, encoding="utf-8")

    # Agent 3 — Performance overview
    sys3 = "Você é um revisor científico. Resuma desempenho global do modelo (accuracy/precision/recall) e limites do experimento. Não invente números; se não houver valores, diga que não estavam disponíveis."
    usr3 = f"{prompt_facts}\n\nTAREFA: Escreva 1 parágrafo curto de desempenho global + 1 parágrafo de limitações."
    perf_report = call_llm(client, model, sys3, usr3)
    (out_dir / "agent_performance.md").write_text(perf_report, encoding="utf-8")

    # Supervisor — synthesis
    sysS = "Você é o agente supervisor. Consolide os relatórios dos agentes em um texto único para seção de Resultados. Seja direto, acadêmico e não invente números."
    usrS = f"""
RELATÓRIO AGENTE RAÇA:
{race_report}

RELATÓRIO AGENTE SEXO:
{sex_report}

RELATÓRIO AGENTE DESEMPENHO:
{perf_report}

TAREFA: Escreva um texto final em 3 blocos:
(1) Desempenho global,
(2) Fairness por raça,
(3) Fairness por sexo,
e finalize com 3 bullets de implicações.
"""
    supervisor = call_llm(client, model, sysS, usrS)
    (out_dir / "supervisor.md").write_text(supervisor, encoding="utf-8")

    print("OK: agent reports generated in", out_dir.resolve())


if __name__ == "__main__":
    main()
