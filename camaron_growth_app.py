import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Crecimiento Semanal Camarón", layout="wide")
st.title("🦐 Predicción de peso semanal de camarón")
st.caption("Carga tu Excel, elige camaronera y piscina, revisa el último ciclo y proyecta una semana con tus propios valores.")

# ============ 1) Carga de datos ============
uploaded = st.file_uploader("Sube tu archivo Excel (hoja con columnas: company_id, pool_id, cycle_number, week_number, oxygen_avg, temperature_avg, feed_weekly, weight_avg)", type=["xlsx"])

if uploaded is None:
    st.info("Sube el Excel para comenzar.")
    st.stop()

try:
    df = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"No se pudo leer el Excel: {e}")
    st.stop()

# Validación básica
required_cols = ["company_id","pool_id","cycle_number","week_number","oxygen_avg","temperature_avg","feed_weekly","weight_avg"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Faltan columnas obligatorias: {missing}")
    st.stop()

# Limpieza mínima
df = df.copy()
df = df.dropna(subset=["company_id","pool_id","cycle_number","week_number"])
for col in ["week_number","oxygen_avg","temperature_avg","feed_weekly","weight_avg"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df_clean = df.dropna(subset=["weight_avg"]).copy()

# ============ 2) Filtros: solo camaronera y piscina ============
col_f1, col_f2 = st.columns(2)
with col_f1:
    company_sel = st.selectbox("Camaronera (company_id)", sorted(df["company_id"].astype(str).unique().tolist()))
with col_f2:
    pool_options = sorted(df[df["company_id"].astype(str) == str(company_sel)]["pool_id"].astype(str).unique().tolist())
    if not pool_options:
        st.error("No hay piscinas para la camaronera seleccionada.")
        st.stop()
    pool_sel = st.selectbox("Piscina (pool_id)", pool_options)

df_cp = df[(df["company_id"].astype(str) == str(company_sel)) &
           (df["pool_id"].astype(str) == str(pool_sel))].copy()

if df_cp.empty:
    st.warning("No hay datos para esa combinación de camaronera y piscina.")
    st.stop()

# ============ 3) Detectar último ciclo y vista previa ============
cycles_series = df_cp["cycle_number"].astype(str)
# Intento de convertir a numérico para ordenar por valor real si procede
cycles_num = pd.to_numeric(cycles_series, errors="coerce")
if cycles_num.notna().any():
    # tomar el valor original cuyo numérico sea máximo
    idx = cycles_num.idxmax()
    last_cycle_val = df_cp.loc[idx, "cycle_number"]
else:
    last_cycle_val = sorted(cycles_series)[-1]

df_last = df_cp[df_cp["cycle_number"].astype(str) == str(last_cycle_val)].copy().sort_values("week_number")

st.subheader(f"📄 Vista previa del **último ciclo**: {last_cycle_val}")
st.dataframe(df_last, use_container_width=True)

# ============ 4) Seleccionar semana a proyectar y valores what-if ============
max_week_observed = int(df_last["week_number"].max()) if not df_last.empty else 1
suggested_week = max(1, max_week_observed + 1)
st.subheader("🎯 Semana a proyectar y variables")
colw1, colw2, colw3, colw4 = st.columns(4)

with colw1:
    week_to_predict = st.number_input("Semana a predecir", min_value=1, max_value=60, value=suggested_week, step=1)

if not df_last.empty:
    last_row = df_last.sort_values("week_number").iloc[-1]
    def_ox = float(last_row["oxygen_avg"]) if pd.notna(last_row["oxygen_avg"]) else float(df_clean["oxygen_avg"].median())
    def_te = float(last_row["temperature_avg"]) if pd.notna(last_row["temperature_avg"]) else float(df_clean["temperature_avg"].median())
    def_fe = float(last_row["feed_weekly"]) if pd.notna(last_row["feed_weekly"]) else float(df_clean["feed_weekly"].median())
else:
    def_ox = float(df_clean["oxygen_avg"].median())
    def_te = float(df_clean["temperature_avg"].median())
    def_fe = float(df_clean["feed_weekly"].median())

with colw2:
    oxygen_in = st.number_input("oxygen_avg (mg/L)", value=def_ox, step=0.1, format="%.2f")
with colw3:
    temperature_in = st.number_input("temperature_avg (°C)", value=def_te, step=0.1, format="%.2f")
with colw4:
    feed_in = st.number_input("feed_weekly (kg)", min_value=0.0, value=def_fe, step=1.0, format="%.2f")

# ============ 5) Entrenamiento del modelo ============
# Entrenar con históricos de esa camaronera+piscina, excluyendo su último ciclo
df_train_pool = df_clean[(df_clean["company_id"].astype(str) == str(company_sel)) &
                         (df_clean["pool_id"].astype(str) == str(pool_sel)) &
                         (df_clean["cycle_number"].astype(str) != str(last_cycle_val))].copy()

# Backoff si hay pocos datos
if len(df_train_pool) < 30:
    df_train_pool = df_clean[(df_clean["company_id"].astype(str) == str(company_sel)) &
                             ~((df_clean["pool_id"].astype(str) == str(pool_sel)) & (df_clean["cycle_number"].astype(str) == str(last_cycle_val)))].copy()
if len(df_train_pool) < 30:
    df_train_pool = df_clean[~((df_clean["company_id"].astype(str) == str(company_sel)) &
                               (df_clean["pool_id"].astype(str) == str(pool_sel)) &
                               (df_clean["cycle_number"].astype(str) == str(last_cycle_val)))].copy()

features = ["week_number","oxygen_avg","temperature_avg","feed_weekly"]
target = "weight_avg"

st.subheader("🧠 Entrenamiento del modelo")
if len(df_train_pool) < 10:
    st.error("No hay suficientes datos históricos para entrenar un modelo confiable. Agrega más ciclos.")
    st.stop()

X = df_train_pool[features].dropna()
y = df_train_pool.loc[X.index, target]

test_size = 0.2 if len(X) >= 10 else 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

m1, m2, m3 = st.columns(3)
m1.metric("R² (test)", f"{r2:0.3f}")
m2.metric("MAE (g)", f"{mae:0.3f}")
m3.metric("Observaciones", f"{len(X):,}")

# ============ 6) Predicción ============
st.subheader("🔮 Predicción de peso para la semana seleccionada")
X_new = pd.DataFrame([{
    "week_number": week_to_predict,
    "oxygen_avg": oxygen_in,
    "temperature_avg": temperature_in,
    "feed_weekly": feed_in
}], columns=features)

pred = float(model.predict(X_new)[0])
st.success(f"**Predicción de weight_avg (g) para la semana {int(week_to_predict)}:** {pred:.3f} g")

# ============ 7) Visualización: último ciclo + punto proyectado ============
st.subheader(f"📈 Curva del último ciclo ({last_cycle_val}) con proyección")
fig, ax = plt.subplots(figsize=(7,4))
if not df_last.empty:
    ax.plot(df_last["week_number"], df_last["weight_avg"], marker="o", label="Peso real último ciclo", linewidth=2)
ax.scatter([week_to_predict], [pred], marker="x", s=120, label="Predicción", linewidth=2)
ax.set_xlabel("Semana")
ax.set_ylabel("Peso promedio (g)")
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)

st.caption("Tip: Ajusta las variables y observa el cambio en la predicción. El modelo se entrena con el historial (excluyendo el último ciclo de esa piscina).")
