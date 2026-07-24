import pandas as pd
import numpy as np
from pathlib import Path

CSV_DIR = Path(r"C:\Users\sabry\Documents\Sabryna\Alana\codebench-mining-tool\csv")
OUT_DIR = CSV_DIR / "processados"
OUT_DIR.mkdir(exist_ok=True)

users_path = CSV_DIR / "users.csv"
grades_path = CSV_DIR / "grades.csv"
logins_path = CSV_DIR / "logins.csv"
executions_path = CSV_DIR / "executions.csv"

users = pd.read_csv(users_path)
grades = pd.read_csv(grades_path)
logins = pd.read_csv(logins_path)
executions = pd.read_csv(executions_path)

# =========================================================
# 1. BASE DE ESTUDANTES
# =========================================================

base = users.rename(columns={"code": "user"}).copy()

colunas_users = [
    "semester",
    "course",
    "user",
    "course_id",
    "course_name",
    "school_type",
    "shift",
    "graduation_year",
    "has_a_pc",
    "share_this_pc",
    "this_pc_has",
    "previous_experience_of",
    "worked_or_interned",
    "sex",
    "year_of_birth",
    "civil_status",
    "have_kids",
]

base = base[[col for col in colunas_users if col in base.columns]]

# =========================================================
# 2. AGREGAÇÃO DE NOTAS E DESEMPENHO
# =========================================================

grades_agg = (
    grades
    .groupby(["semester", "course", "user"], as_index=False)
    .agg(
        n_atividades=("assignment", "nunique"),
        nota_media=("grade", "mean"),
        nota_mediana=("grade", "median"),
        nota_minima=("grade", "min"),
        nota_maxima=("grade", "max"),
        total_problemas=("n_problems", "sum"),
        total_acertos=("n_correct", "sum"),
        total_erros=("n_wrong", "sum"),
        total_brancos=("n_blank", "sum"),
    )
)

grades_agg["taxa_acerto"] = np.where(
    grades_agg["total_problemas"] > 0,
    grades_agg["total_acertos"] / grades_agg["total_problemas"],
    np.nan
)

grades_agg["taxa_erro"] = np.where(
    grades_agg["total_problemas"] > 0,
    grades_agg["total_erros"] / grades_agg["total_problemas"],
    np.nan
)

grades_agg["taxa_branco"] = np.where(
    grades_agg["total_problemas"] > 0,
    grades_agg["total_brancos"] / grades_agg["total_problemas"],
    np.nan
)

# Definição simples de baixo desempenho.
# Pode ser ajustada depois, se a professora pedir outro critério.
grades_agg["baixo_desempenho"] = np.where(grades_agg["nota_media"] < 6.0, 1, 0)

# =========================================================
# 3. AGREGAÇÃO DE LOGINS
# =========================================================

logins = logins.drop_duplicates().copy()

logins["datetime_login"] = pd.to_datetime(
    logins["date"].astype(str) + " " + logins["time"].astype(str),
    errors="coerce"
)

logins["data_login"] = logins["datetime_login"].dt.date

login_events = logins[logins["event"].astype(str).str.contains("login", case=False, na=False)].copy()
logout_events = logins[logins["event"].astype(str).str.contains("logout", case=False, na=False)].copy()

logins_agg = (
    login_events
    .groupby(["semester", "course", "user"], as_index=False)
    .agg(
        n_logins=("event", "count"),
        dias_ativos_login=("data_login", "nunique"),
        primeiro_login=("datetime_login", "min"),
        ultimo_login=("datetime_login", "max"),
    )
)

logouts_agg = (
    logout_events
    .groupby(["semester", "course", "user"], as_index=False)
    .agg(
        n_logouts=("event", "count")
    )
)

# =========================================================
# 4. AGREGAÇÃO DE EXECUÇÕES
# =========================================================

execs = executions.copy()

# Ajuste importante:
# Pela inspeção dos dados, em executions.csv as colunas user e assignment parecem invertidas.
execs["execution_assignment"] = execs["user"]
execs["execution_user"] = execs["assignment"]

execs["datetime_clean"] = (
    execs["datetime"]
    .astype(str)
    .str.replace(")", "", regex=False)
    .str.strip()
)

execs["datetime_exec"] = pd.to_datetime(execs["datetime_clean"], errors="coerce")

execs["is_submission"] = execs["ex_type"].astype(str).str.contains("SUBM", case=False, na=False)
execs["is_test"] = execs["ex_type"].astype(str).str.upper().eq("TEST")

execs["has_err_bool"] = execs["has_err"].astype(str).str.lower().isin(["true", "1", "yes"])

execs["grade_str"] = execs["grade"].astype(str)
execs["submission_100"] = execs["grade_str"].str.contains("100%", na=False)
execs["submission_0"] = execs["grade_str"].str.contains("0%", na=False)

execs_agg = (
    execs
    .groupby(["semester", "course", "execution_user"], as_index=False)
    .agg(
        n_execucoes=("ex_type", "count"),
        n_submissoes=("is_submission", "sum"),
        n_testes=("is_test", "sum"),
        n_execucoes_com_erro=("has_err_bool", "sum"),
        n_submissoes_100=("submission_100", "sum"),
        n_submissoes_0=("submission_0", "sum"),
        primeiro_evento_exec=("datetime_exec", "min"),
        ultimo_evento_exec=("datetime_exec", "max"),
        tempo_execucao_medio=("exec_time", "mean"),
        complexidade_media=("complexity", "mean"),
        loc_medio=("loc", "mean"),
    )
    .rename(columns={"execution_user": "user"})
)

execs_agg["taxa_erro_execucao"] = np.where(
    execs_agg["n_execucoes"] > 0,
    execs_agg["n_execucoes_com_erro"] / execs_agg["n_execucoes"],
    np.nan
)

execs_agg["taxa_submissao_100"] = np.where(
    execs_agg["n_submissoes"] > 0,
    execs_agg["n_submissoes_100"] / execs_agg["n_submissoes"],
    np.nan
)

# =========================================================
# 5. VARIÁVEIS DAS PRIMEIRAS 3 SEMANAS
# =========================================================

# Usaremos como início da turma a primeira execução registrada no curso.
inicio_turma = (
    execs
    .dropna(subset=["datetime_exec"])
    .groupby(["semester", "course"], as_index=False)
    .agg(inicio_turma=("datetime_exec", "min"))
)

execs = execs.merge(inicio_turma, on=["semester", "course"], how="left")
execs["dias_desde_inicio"] = (execs["datetime_exec"] - execs["inicio_turma"]).dt.days
execs_3w = execs[(execs["dias_desde_inicio"] >= 0) & (execs["dias_desde_inicio"] < 21)].copy()

execs_3w_agg = (
    execs_3w
    .groupby(["semester", "course", "execution_user"], as_index=False)
    .agg(
        n_execucoes_3w=("ex_type", "count"),
        n_submissoes_3w=("is_submission", "sum"),
        n_testes_3w=("is_test", "sum"),
        n_erros_execucao_3w=("has_err_bool", "sum"),
        n_submissoes_100_3w=("submission_100", "sum"),
    )
    .rename(columns={"execution_user": "user"})
)

logins_inicio = logins.merge(inicio_turma, on=["semester", "course"], how="left")
logins_inicio["dias_desde_inicio"] = (logins_inicio["datetime_login"] - logins_inicio["inicio_turma"]).dt.days
logins_3w = logins_inicio[
    (logins_inicio["dias_desde_inicio"] >= 0)
    & (logins_inicio["dias_desde_inicio"] < 21)
    & (logins_inicio["event"].astype(str).str.contains("login", case=False, na=False))
].copy()

logins_3w_agg = (
    logins_3w
    .groupby(["semester", "course", "user"], as_index=False)
    .agg(
        n_logins_3w=("event", "count"),
        dias_ativos_3w=("data_login", "nunique"),
    )
)

# =========================================================
# 6. JUNÇÃO FINAL
# =========================================================

df = base.merge(grades_agg, on=["semester", "course", "user"], how="left")
df = df.merge(logins_agg, on=["semester", "course", "user"], how="left")
df = df.merge(logouts_agg, on=["semester", "course", "user"], how="left")
df = df.merge(execs_agg, on=["semester", "course", "user"], how="left")
df = df.merge(execs_3w_agg, on=["semester", "course", "user"], how="left")
df = df.merge(logins_3w_agg, on=["semester", "course", "user"], how="left")

# Preencher contagens ausentes com zero
colunas_contagem = [
    "n_atividades",
    "total_problemas",
    "total_acertos",
    "total_erros",
    "total_brancos",
    "n_logins",
    "n_logouts",
    "dias_ativos_login",
    "n_execucoes",
    "n_submissoes",
    "n_testes",
    "n_execucoes_com_erro",
    "n_submissoes_100",
    "n_submissoes_0",
    "n_execucoes_3w",
    "n_submissoes_3w",
    "n_testes_3w",
    "n_erros_execucao_3w",
    "n_submissoes_100_3w",
    "n_logins_3w",
    "dias_ativos_3w",
]

for col in colunas_contagem:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# Recalcular taxas das primeiras semanas
df["taxa_erro_execucao_3w"] = np.where(
    df["n_execucoes_3w"] > 0,
    df["n_erros_execucao_3w"] / df["n_execucoes_3w"],
    np.nan
)

df["taxa_submissao_100_3w"] = np.where(
    df["n_submissoes_3w"] > 0,
    df["n_submissoes_100_3w"] / df["n_submissoes_3w"],
    np.nan
)

# Salvar base final
saida_base = OUT_DIR / "base_agregada_codebench_2024_1.csv"
df.to_csv(saida_base, index=False, encoding="utf-8-sig")

# =========================================================
# 7. RELATÓRIO RÁPIDO PARA PREENCHER O DOCUMENTO
# =========================================================

resumo = []

resumo.append("BASE AGREGADA GERADA")
resumo.append("=" * 80)
resumo.append(f"Arquivo salvo em: {saida_base}")
resumo.append(f"Número de registros: {df.shape[0]}")
resumo.append(f"Número de variáveis: {df.shape[1]}")
resumo.append(f"Registros duplicados: {df.duplicated().sum()}")
resumo.append(f"Registros com pelo menos um valor faltante: {df.isnull().any(axis=1).sum()}")
resumo.append(f"Total de valores faltantes: {df.isnull().sum().sum()}")

resumo.append("\nCOLUNAS DA BASE FINAL:")
for col in df.columns:
    resumo.append(f"- {col}")

variaveis_numericas = [
    "n_logins",
    "n_logins_3w",
    "dias_ativos_3w",
    "n_submissoes",
    "n_submissoes_3w",
    "n_execucoes",
    "n_execucoes_3w",
    "n_execucoes_com_erro",
    "n_erros_execucao_3w",
    "total_acertos",
    "total_erros",
    "total_brancos",
    "taxa_acerto",
    "taxa_erro",
    "taxa_erro_execucao",
    "taxa_erro_execucao_3w",
    "taxa_submissao_100",
    "taxa_submissao_100_3w",
    "nota_media",
]

variaveis_numericas = [col for col in variaveis_numericas if col in df.columns]

desc = df[variaveis_numericas].describe().T
desc = desc[["mean", "50%", "std", "min", "max"]]
desc = desc.rename(columns={
    "mean": "media",
    "50%": "mediana",
    "std": "desvio_padrao",
    "min": "minimo",
    "max": "maximo",
})

resumo.append("\nESTATÍSTICAS DESCRITIVAS:")
resumo.append(desc.to_string())

resumo.append("\nPRIMEIRAS 10 LINHAS DA BASE FINAL:")
resumo.append(df.head(10).to_string())

saida_resumo = OUT_DIR / "resumo_base_agregada.txt"
saida_desc = OUT_DIR / "estatisticas_descritivas_base_agregada.csv"

saida_resumo.write_text("\n".join(resumo), encoding="utf-8")
desc.to_csv(saida_desc, encoding="utf-8-sig")

print("\n".join(resumo))
print(f"\nBase agregada salva em: {saida_base}")
print(f"Resumo salvo em: {saida_resumo}")
print(f"Estatísticas salvas em: {saida_desc}")