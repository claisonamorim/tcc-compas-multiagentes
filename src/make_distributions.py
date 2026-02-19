import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = os.getenv("COMPAS_CSV", "/app/data/compas-scores-two-years.csv")
OUT_DIR = "/app/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Distribuição por raça
ax = df["race"].value_counts().plot(kind="bar")
ax.set_title("Distribuição por Raça")
ax.set_ylabel("Número de exemplos")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/distribuicao_raca.png", dpi=200)
plt.close()

# Distribuição por sexo
ax = df["sex"].value_counts().plot(kind="bar")
ax.set_title("Distribuição por Sexo")
ax.set_ylabel("Número de exemplos")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/distribuicao_sexo.png", dpi=200)
plt.close()

print("OK: figuras geradas em /app/outputs/")
