
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="Predicción de Crecimiento Semanal - Camarón", layout="wide")

st.title("🦐 Predicción de Crecimiento Semanal (Peso)")
st.caption("Interfaz interactiva para entrenar un modelo con tus datos y simular escenarios (what‑if).")

# Sidebar – carga de datos
st.sidebar.header("1) Cargar datos")
uploaded = st.sidebar.file_uploader("Sube tu archivo Excel 'data.xlsx' (hoja con columnas: company_id, pool_id, cycle_number, week_number, oxygen_avg, temperature_avg, feed_weekly, weight_avg)", type=["xlsx"])

@st.cache_data(show_spinner=False)
def load_df(file):
    df = pd.read_excel(file)
    return df

if uploaded is None:
    st.info("Sube tu archivo Excel para comenzar.")
    st.stop()

df = load_df(uploaded).copy()

# Validación mínima de columnas
required_cols = ["company_id","pool_id","cycle_number","week_number","oxygen_avg","temperature_avg","feed_weekly","weight_avg"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Faltan columnas obligatorias: {missing}")
    st.stop()

st.subheader("Vista previa de datos")
st.dataframe(df.head(20), use_container_width=True)

# Conversión de tipos básicos
for col in ["week_number","oxygen_avg","temperature_avg","feed_weekly","weight_avg"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Selección de ciclo objetivo y semana objetivo
st.sidebar.header("2) Configurar objetivo")
ciclo_objetivo = st.sidebar.selectbox("Ciclo objetivo (para predecir):", sorted(df["cycle_number"].unique()), index=0)
semana_objetivo = st.sidebar.number_input("Semana objetivo a predecir:", min_value=1, max_value=60, value=9, step=1)

# Codificar categóricas (mantener mapping por reproducibilidad)
enc_company = LabelEncoder()
enc_pool = LabelEncoder()
enc_cycle = LabelEncoder()

df["company_enc"] = enc_company.fit_transform(df["company_id"].astype(str))
df["pool_enc"]    = enc_pool.fit_transform(df["pool_id"].astype(str))
df["cycle_enc"]   = enc_cycle.fit_transform(df["cycle_number"].astype(str))

# Selección de variables predictoras
st.sidebar.header("3) Variables del modelo")
base_feats = ["week_number","oxygen_avg","temperature_avg","feed_weekly"]
extra_feats = st.sidebar.multiselect(
    "Agrega contexto (opcional):",
    options=["company_enc","pool_enc","cycle_enc"],
    default=[],
    help="Añade identificadores codificados si existen patrones por empresa/piscina/ciclo."
)
features = base_feats + extra_feats

st.write("**Variables usadas por el modelo:**", features)

# Separar entrenamiento (excluir ciclo objetivo) / datos del ciclo objetivo
df_train = df[df["cycle_number"] != ciclo_objetivo].dropna(subset=features+["weight_avg"])
df_target_cycle = df[df["cycle_number"] == ciclo_objetivo].copy()

if df_train.empty or len(df_train) < 30:
    st.warning("Muy pocos datos para entrenar. Asegúrate de tener históricos suficientes excluyendo el ciclo objetivo.")
    st.stop()

X = df_train[features]
y = df_train["weight_avg"]

# Entrenar modelo
test_size = st.sidebar.slider("Proporción de prueba", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

n_estimators = st.sidebar.slider("Árboles (n_estimators)", 100, 800, 300, 50)
max_depth = st.sidebar.slider("Profundidad máxima (max_depth)", 2, 30, 0, 1)
max_depth = None if max_depth == 0 else max_depth

model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

mcol1, mcol2, mcol3 = st.columns(3)
mcol1.metric("R² (test)", f"{r2:0.3f}")
mcol2.metric("MAE (g)", f"{mae:0.3f}")
mcol3.metric("Observaciones", f"{len(df_train):,}")

# Importancia de variables
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
st.subheader("Importancia de variables")
st.bar_chart(importances)

# Panel para simulación (what-if) de la semana objetivo
st.subheader("Simulación de la semana objetivo (what‑if)")
defaults = {}

# Si existe la semana objetivo en el ciclo objetivo, pre‑poblar con sus valores, sino usar medianas del ciclo/entreno
row_week = df_target_cycle[df_target_cycle["week_number"] == semana_objetivo]
if len(row_week) == 1:
    defaults["week_number"] = int(row_week["week_number"].iloc[0])
    defaults["oxygen_avg"] = float(row_week["oxygen_avg"].iloc[0])
    defaults["temperature_avg"] = float(row_week["temperature_avg"].iloc[0])
    defaults["feed_weekly"] = float(row_week["feed_weekly"].iloc[0])
else:
    defaults["week_number"] = int(semana_objetivo)
    defaults["oxygen_avg"] = float(df_train["oxygen_avg"].median())
    defaults["temperature_avg"] = float(df_train["temperature_avg"].median())
    defaults["feed_weekly"] = float(df_train["feed_weekly"].median())

# Defaults para extras
defaults["company_enc"] = int(enc_company.transform([df_target_cycle["company_id"].iloc[0]])[0]) if not df_target_cycle.empty else int(df["company_enc"].mode()[0])
defaults["pool_enc"] = int(enc_pool.transform([df_target_cycle["pool_id"].iloc[0]])[0]) if not df_target_cycle.empty else int(df["pool_enc"].mode()[0])
defaults["cycle_enc"] = int(enc_cycle.transform([ciclo_objetivo])[0])

# Construir controles dinámicamente según features
inputs = {}
cols = st.columns(min(len(features),4) or 1)
i = 0
for feat in features:
    col = cols[i % len(cols)]
    if feat == "week_number":
        inputs[feat] = col.number_input("Semana", min_value=1, max_value=60, value=int(defaults["week_number"]), step=1)
    elif feat in ["oxygen_avg","temperature_avg"]:
        inputs[feat] = col.number_input(feat, value=float(defaults[feat]), step=0.1, format="%.2f")
    elif feat == "feed_weekly":
        inputs[feat] = col.number_input("feed_weekly (kg)", min_value=0.0, value=float(defaults["feed_weekly"]), step=1.0, format="%.2f")
    elif feat in ["company_enc","pool_enc","cycle_enc"]:
        # Mostrar como select si provienen de encoders
        if feat == "company_enc":
            labels = list(enc_company.classes_)
            sel = col.selectbox("company_id", labels, index=int(defaults["company_enc"]))
            inputs[feat] = int(enc_company.transform([sel])[0])
        elif feat == "pool_enc":
            labels = list(enc_pool.classes_)
            sel = col.selectbox("pool_id", labels, index=int(defaults["pool_enc"]))
            inputs[feat] = int(enc_pool.transform([sel])[0])
        elif feat == "cycle_enc":
            labels = list(enc_cycle.classes_)
            # mostrar ciclo texto original
            sel = col.selectbox("cycle_number", labels, index=int(defaults["cycle_enc"]))
            inputs[feat] = int(enc_cycle.transform([sel])[0])
    else:
        inputs[feat] = col.number_input(feat, value=float(df_train[feat].median()))
    i += 1

# Predicción
if st.button("🔮 Predecir peso (g)"):
    X_new = pd.DataFrame([inputs], columns=features)
    pred = float(model.predict(X_new)[0])
    st.success(f"Predicción de **weight_avg** para la semana {int(inputs.get('week_number', semana_objetivo))}: **{pred:.3f} g**")

    # Curva del ciclo objetivo con el punto predicho
    hist = df_target_cycle.sort_values("week_number")
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(hist["week_number"], hist["weight_avg"], marker="o", label="Peso real ciclo objetivo", linewidth=2)
    ax.scatter([inputs.get("week_number", semana_objetivo)], [pred], marker="x", s=120, label="Predicción", linewidth=2)
    ax.set_xlabel("Semana")
    ax.set_ylabel("Peso promedio (g)")
    ax.set_title(f"Crecimiento – Ciclo {ciclo_objetivo}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

st.caption("Sugerencia: ajusta las variables (oxígeno, temperatura, alimento, semana) y observa el efecto en la predicción. Añade 'company_enc', 'pool_enc' o 'cycle_enc' si hay patrones específicos por empresa/piscina/ciclo.")
