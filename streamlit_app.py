# =========================================================
# DI Without Insulin Calculator
# Version 1.2
# Streamlit app
# ---------------------------------------------------------
# Input glucose unit: mg/dL
# Time unit: min
# Fitted parameters: sigma, si
# Raw mDI-woI: sigma_raw * si_raw
# Paper scale: 10^-4 mL/mU/min = raw mDI-woI * 10000 / 1440
# =========================================================

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# =========================================================
# App constants
# =========================================================

APP_VERSION = "1.2"
APP_TITLE = "DI Without Insulin Calculator"
DISCLAIMER = (
    "本アプリは研究・教育目的の計算補助ツールです。"
    "診断、治療方針決定、保険診療上の判断には使用しないでください。"
    "利用および結果の解釈は使用者の責任でお願いします。"
)

DEFAULT_ID_COL = "subject"
DEFAULT_GLUCOSE_COLS = ["G0", "G30", "G60", "G90", "G120"]
DEFAULT_TDATA = np.array([0, 30, 60, 90, 120], dtype=float)

MINUTES_PER_DAY = 1440.0
PAPER_SCALE_FACTOR = 10000.0 / MINUTES_PER_DAY

# =========================================================
# Constants copied from OGTT_fit_woI.m
# =========================================================

INIT = np.array([80.1842, 5.8462, 60.9341, 443.7764], dtype=float)
THETA0 = np.array([1.0, 0.8], dtype=float)  # [sigma, si]
LB = np.array([0.01, 0.01], dtype=float)
UB = np.array([10.0, 100.0], dtype=float)
BOUND_PENALTY = 1e6

ODEPARAMS_BASE = {
    "a1": 5.99,
    "a2": 2.14,
    "a3": 0.0013,
    "t1": 15.60,
    "t2": 137.2,
    "t3": 258.3,
    "r20": 0.006,
    "hepasi": 1.0,
    "gamma_bar": 1.0,
    "gamma": 2.5e-6 * 17500,
}

# =========================================================
# ODE model
# =========================================================

def gi_ode_universal(t: float, y: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    """Faithful Python translation of GI_ode_universal_new.m.

    y = [G, I, N5, N6]
    G is model glucose, internally treated as mg/dL.
    I is model-predicted insulin-like state variable, not measured insulin.
    """
    G, I, N5, N6 = y

    Eg0 = 0.0118
    k = 0.4861
    BV = 7200.0
    b = 1553.6

    # Metabolic rate M
    Mmax = 1.0
    alpha_M = 150.0
    kM = 2.0
    M = Mmax * (G ** kM) / (alpha_M ** kM + G ** kM)

    # Rate of appearance of glucose after OGTT
    h0 = float(t > 0)
    h1 = float(t > p["t1"])
    h2 = float(t > p["t2"])
    h3 = float(t > p["t3"])

    OGTT_rate = (
        (h0 - h1) * t * p["a1"] / p["t1"]
        + (h1 - h2)
        * (((t - p["t2"]) * (p["a2"] - p["a1"]) / (p["t2"] - p["t1"])) + p["a2"])
        + (h2 - h3)
        * (t - p["t3"]) * (p["a3"] - p["a2"]) / (p["t3"] - p["t2"])
    )

    # Hepatic glucose production
    hepa_bar = 15.443
    hepa_k = 0.27
    hepa_b = -3.54277
    hepa_max = hepa_bar / (hepa_k + p["si"]) + hepa_b

    alpha_max = 6.0
    alpha_k = 0.4
    alpha_b = -0.5
    alpha_HGP = alpha_max / (alpha_k + p["si"]) + alpha_b

    HGP_b = 0.104166
    HGP = hepa_max / (alpha_HGP + I * p["hepasi"]) + HGP_b

    # Glucose amplification factor
    GF_bar = 4.45
    kGF = 16.0
    alpha_GF = 260.0
    shGF = -89.0
    GF_b = 1.78
    GF = (GF_bar * ((G - shGF) ** kGF) / (alpha_GF ** kGF + ((G - shGF) ** kGF))) + GF_b

    # Cytosolic calcium
    ca_bar = 2.0
    kca = 4.0
    alpha_ca = 0.62
    ca_b = 0.07
    ci = ca_bar * ((M + p["gamma_bar"] * p["gamma"]) ** kca) / (
        alpha_ca ** kca + ((M + p["gamma_bar"] * p["gamma"]) ** kca)
    ) + ca_b

    # Microdomain Ca2+
    cmd_factor = 150.0
    cmd_b = 0.0635
    cik = 4.0
    cialpha = 1.0
    cmd = cmd_factor * (ci ** cik) / (cialpha ** cik + ci ** cik) + cmd_b

    # Exocytosis model
    k1 = 20.0
    km1 = 100.0
    r1 = 0.6
    rm1 = 1.0
    rm2 = 0.001

    r30 = 1.205
    rm3 = 0.0001
    u1 = 2000.0
    u2 = 3.0
    u3 = 0.02
    Kp2 = 2.3

    r2 = p["r20"] * ci / (ci + Kp2)
    ts = 60.0
    unit_con = 0.00069444  # ≈ 1/1440
    r3 = p["sigma"] * GF * r30 * ci / (ci + Kp2)

    N1_C = km1 / (3.0 * k1 * cmd + rm1)
    N1_D = r1 / (3.0 * k1 * cmd + rm1)

    N2_E = 3.0 * k1 * cmd / (2.0 * k1 * cmd + km1)
    N2_F = 2.0 * km1 / (2.0 * k1 * cmd + km1)

    N3_L = 2.0 * k1 * cmd / (2.0 * km1 + k1 * cmd)
    N3_N = 3.0 * km1 / (2.0 * km1 + k1 * cmd)

    CN4 = k1 * cmd / (3.0 * km1 + u1)
    CN3 = N3_L / (1.0 - N3_N * CN4)
    CN2 = N2_E / (1.0 - N2_F * CN3)
    CN1 = N1_D / (1.0 - N1_C * CN2)

    N1 = CN1 * N5
    N2 = CN2 * N1
    N3 = CN3 * N2
    N4 = CN4 * N3
    NF = u1 * N4 / u2
    NR = (u2 / u3) * NF

    ISR = ts * 9.0 * (u3 * NR)

    dGdt = HGP + OGTT_rate - (Eg0 + unit_con * p["si"] * I) * G
    dIdt = b * ISR / BV - k * I
    dN5dt = ts * (rm1 * CN1 * N5 - (r1 + rm2) * N5 + r2 * N6)
    dN6dt = ts * (r3 + rm2 * N5 - (rm3 + r2) * N6)

    return np.array([dGdt, dIdt, dN5dt, dN6dt], dtype=float)

# =========================================================
# Solver and cost function
# =========================================================

def interp1_matlab_nan(x: np.ndarray, y: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Linear interpolation similar to MATLAB interp1 default."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xi = np.asarray(xi, dtype=float)
    yi = np.interp(xi, x, y)
    yi[(xi < x[0]) | (xi > x[-1])] = np.nan
    return yi


def solve_model(theta: np.ndarray, data: Dict[str, np.ndarray], init: np.ndarray, odeparams_base: Dict[str, float]):
    """Run-in [0,1440] and then simulate OGTT, following the MATLAB code."""
    p = dict(odeparams_base)
    p["sigma"] = float(theta[0])
    p["si"] = float(theta[1])

    sol1 = solve_ivp(
        fun=lambda t, y: gi_ode_universal(t, y, p),
        t_span=(0.0, 1440.0),
        y0=np.asarray(init, dtype=float),
        method="BDF",
        rtol=1e-5,
        atol=1e-8,
    )
    if not sol1.success:
        raise RuntimeError("run-in ODE failed: " + sol1.message)

    init2 = sol1.y[:, -1]
    tspan = (float(data["t"][0]), float(data["t"][-1]))

    sol2 = solve_ivp(
        fun=lambda t, y: gi_ode_universal(t, y, p),
        t_span=tspan,
        y0=init2,
        method="BDF",
        rtol=1e-5,
        atol=1e-8,
    )
    if not sol2.success:
        raise RuntimeError("OGTT ODE failed: " + sol2.message)

    t = sol2.t
    y = sol2.y.T

    Gpred = interp1_matlab_nan(t, y[:, 0], data["t"])
    Gsim = interp1_matlab_nan(t, y[:, 0], data["outtime"])
    Isim = interp1_matlab_nan(t, y[:, 1], data["outtime"])

    return Gpred, Gsim, Isim


def ogtt_cost_function_universal_woi(theta, data, costparams, odeparams_base):
    """Insulin-free cost function; only glucose residuals are used."""
    theta = np.asarray(theta, dtype=float)

    if np.any(theta < costparams["LB"]) or np.any(theta > costparams["UB"]):
        return BOUND_PENALTY, BOUND_PENALTY, None, None

    try:
        Gpred, Gsim, Isim = solve_model(theta, data, costparams["init"], odeparams_base)
        resid = Gpred - data["G"]
        S = np.sum(costparams["weights_G"] * (resid ** 2))
        true_err = np.sum(resid ** 2)

        if not np.isfinite(S) or not np.isfinite(true_err):
            return BOUND_PENALTY, BOUND_PENALTY, None, None

        return float(S), float(true_err), Gsim, Isim

    except Exception:
        return BOUND_PENALTY, BOUND_PENALTY, None, None

# =========================================================
# Fitting functions
# =========================================================

def convert_units(sigma_raw: float, si_raw: float) -> Dict[str, float]:
    """Return raw and unit-converted mDI-woI values."""
    mdi_raw = sigma_raw * si_raw
    return {
        "sigma_raw": sigma_raw,
        "si_raw": si_raw,
        "mDI_woI_raw_sigma_x_si": mdi_raw,
        "si_mL_per_mU_per_min": si_raw / MINUTES_PER_DAY,
        "mDI_woI_mL_per_mU_per_min": mdi_raw / MINUTES_PER_DAY,
        "si_10minus4_mL_per_mU_per_min": si_raw * PAPER_SCALE_FACTOR,
        "mDI_woI_10minus4_mL_per_mU_per_min": mdi_raw * PAPER_SCALE_FACTOR,
    }


def fit_one_subject_woi(
    glucose_values,
    weights_G,
    tdata=DEFAULT_TDATA,
    outtime=None,
    odeparams_base=ODEPARAMS_BASE,
):
    """Fit one subject using glucose values only."""
    if outtime is None:
        outtime = np.asarray(tdata, dtype=float).copy()

    glucose_values = np.asarray(glucose_values, dtype=float)
    weights_G = np.asarray(weights_G, dtype=float)

    data = {
        "t": np.asarray(tdata, dtype=float).copy(),
        "G": glucose_values.copy(),
        "outtime": np.asarray(outtime, dtype=float).copy(),
    }
    costparams = {
        "init": INIT.copy(),
        "LB": LB.copy(),
        "UB": UB.copy(),
        "weights_G": weights_G.copy(),
    }

    missing = np.isnan(data["G"])
    data["t"] = data["t"][~missing]
    data["G"] = data["G"][~missing]
    costparams["weights_G"] = costparams["weights_G"][~missing]

    if len(data["t"]) == 0:
        return {"fit_status": "missing_all"}

    def objective(theta):
        S, _, _, _ = ogtt_cost_function_universal_woi(theta, data, costparams, odeparams_base)
        return S

    result = minimize(
        objective,
        THETA0.copy(),
        method="Nelder-Mead",
        options={
            "xatol": 1e-2,
            "fatol": 1e2,
            "maxfev": int(1e4),
            "maxiter": int(1e4),
            "disp": False,
        },
    )

    paramin = np.asarray(result.x, dtype=float)
    WRSS, true_error, Gsim, Isim = ogtt_cost_function_universal_woi(paramin, data, costparams, odeparams_base)

    sigma_raw = float(paramin[0])
    si_raw = float(paramin[1])
    unit_values = convert_units(sigma_raw, si_raw)

    SS = np.sum(data["G"] ** 2)
    RSQ = 1.0 - true_error / SS if SS > 0 else np.nan

    numpar = 2
    numpts = len(tdata)  # MATLAB uses length(tdata), not number of non-missing points.
    AIC = 2 * numpar + numpts * np.log(numpts * WRSS) if WRSS > 0 else np.nan

    out = {
        **unit_values,
        "WRSS": WRSS,
        "true_error": true_error,
        "RSQ_matlab_style": RSQ,
        "AIC_matlab_style": AIC,
        "fit_status": "success" if result.success else "optimizer_not_converged",
        "optimizer_message": str(result.message),
        "nfev": result.nfev,
    }

    if Gsim is None or Isim is None:
        for t in tdata:
            out[f"G{int(t)}_sim"] = np.nan
            out[f"I{int(t)}_sim"] = np.nan
    else:
        for t, g, i in zip(tdata, Gsim, Isim):
            out[f"G{int(t)}_sim"] = g
            out[f"I{int(t)}_sim"] = i

    return out


def compute_weights_from_dataframe(df: pd.DataFrame, glucose_cols: List[str]) -> np.ndarray:
    """MATLAB-style weights: 1 / std(GDATA, omitnan)."""
    GDATA = df[glucose_cols].to_numpy(dtype=float)
    sd_G = np.nanstd(GDATA, axis=0, ddof=1)

    # If only one subject is provided, sd becomes NaN. In that case, use equal weights.
    if np.any(~np.isfinite(sd_G)) or np.any(sd_G <= 0):
        return np.ones(len(glucose_cols), dtype=float)

    return 1.0 / sd_G


def fit_dataframe_woi(df: pd.DataFrame, id_col: str, glucose_cols: List[str], tdata: np.ndarray, progress_callback=None):
    """Fit all subjects in a dataframe."""
    work = df.copy()
    for col in glucose_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    weights_G = compute_weights_from_dataframe(work, glucose_cols)

    rows = []
    n = len(work)
    for idx, row in work.iterrows():
        subj = row[id_col] if id_col in work.columns else idx + 1
        glucose_values = row[glucose_cols].to_numpy(dtype=float)

        res = fit_one_subject_woi(glucose_values, weights_G, tdata=tdata, outtime=tdata.copy())

        base = {id_col: subj}
        for col, val in zip(glucose_cols, glucose_values):
            base[col] = val
        base.update(res)
        rows.append(base)

        if progress_callback is not None:
            progress_callback(idx + 1, n, subj, res)

    result_df = pd.DataFrame(rows)
    return result_df, weights_G

# =========================================================
# Streamlit helper functions
# =========================================================

def make_template_excel() -> bytes:
    """Create an Excel template for batch input."""
    example = pd.DataFrame(
        {
            "subject": ["Sample_001", "Sample_002", "Sample_003"],
            "G0": [90, 95, 85],
            "G30": [140, 155, 130],
            "G60": [160, 180, 145],
            "G90": [135, 150, 125],
            "G120": [110, 125, 100],
        }
    )

    readme = pd.DataFrame(
        {
            "Item": [
                "Version",
                "Required columns",
                "Glucose unit",
                "Time unit",
                "Missing values",
                "Raw output",
                "Paper-scale output",
                "Disclaimer",
            ],
            "Description": [
                APP_VERSION,
                "subject, G0, G30, G60, G90, G120",
                "mg/dL",
                "min",
                "Blank cells are treated as missing time points.",
                "mDI_woI_raw_sigma_x_si = sigma_raw × si_raw",
                "mDI_woI_10minus4_mL_per_mU_per_min = raw × 10000 / 1440",
                DISCLAIMER,
            ],
        }
    )

    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        example.to_excel(writer, index=False, sheet_name="Template")
        readme.to_excel(writer, index=False, sheet_name="Readme")
    bio.seek(0)
    return bio.getvalue()


def dataframe_to_excel_bytes(df: pd.DataFrame, weights_G: np.ndarray | None = None) -> bytes:
    """Convert result dataframe to Excel bytes."""
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
        if weights_G is not None:
            pd.DataFrame({"time": DEFAULT_TDATA[: len(weights_G)], "weights_G": weights_G}).to_excel(
                writer, index=False, sheet_name="Weights"
            )
        info = pd.DataFrame(
            {
                "Item": ["Version", "Raw mDI-woI", "Paper-scale mDI-woI", "Disclaimer"],
                "Description": [
                    APP_VERSION,
                    "mDI_woI_raw_sigma_x_si = sigma_raw × si_raw",
                    "mDI_woI_10minus4_mL_per_mU_per_min = raw × 10000 / 1440",
                    DISCLAIMER,
                ],
            }
        )
        info.to_excel(writer, index=False, sheet_name="Info")
    bio.seek(0)
    return bio.getvalue()


def display_result_summary(res: Dict[str, float]):
    """Display main result metrics in Streamlit."""
    c1, c2, c3 = st.columns(3)
    c1.metric("mDI-woI raw", f"{res.get('mDI_woI_raw_sigma_x_si', np.nan):.6g}")
    c2.metric("mDI-woI, 10^-4 mL/mU/min", f"{res.get('mDI_woI_10minus4_mL_per_mU_per_min', np.nan):.6g}")
    c3.metric("R², MATLAB style", f"{res.get('RSQ_matlab_style', np.nan):.4f}")

# =========================================================
# Streamlit UI
# =========================================================

st.set_page_config(page_title=f"{APP_TITLE} v{APP_VERSION}", layout="wide")

st.title(APP_TITLE)
st.caption(f"Version {APP_VERSION}")

st.warning(DISCLAIMER)

with st.expander("単位と出力値について", expanded=True):
    st.markdown(
        """
- 入力血糖単位：**mg/dL**
- 時間単位：**min**
- 推定パラメータ：**sigma**, **si**
- raw mDI-woI：`sigma_raw × si_raw`
- 論文表記に合わせた値：`mDI_woI_raw × 10000 / 1440`
- 論文表記列：**`mDI_woI_10minus4_mL_per_mU_per_min`**

Excel一括計算では、MATLAB版と同様に各時点の血糖標準偏差から重み `weights_G = 1 / SD` を計算します。  
単独計算ではコホート内標準偏差を計算できないため、デフォルトでは等重みを用います。
        """
    )

single_tab, batch_tab, template_tab, about_tab = st.tabs(
    ["単独計算", "Excel一括計算", "入力テンプレート", "About"]
)

# ---------------------------------------------------------
# Single-subject calculation
# ---------------------------------------------------------
with single_tab:
    st.subheader("単独計算")
    st.info("単独計算ではコホート由来の重みを計算できないため、等重みを使用します。論文/MATLAB形式に近い一括計算にはExcel入力を推奨します。")

    subject_id = st.text_input("Subject ID", value="Single_001")

    cols = st.columns(5)
    g0 = cols[0].number_input("G0", min_value=0.0, value=90.0, step=1.0)
    g30 = cols[1].number_input("G30", min_value=0.0, value=140.0, step=1.0)
    g60 = cols[2].number_input("G60", min_value=0.0, value=160.0, step=1.0)
    g90 = cols[3].number_input("G90", min_value=0.0, value=135.0, step=1.0)
    g120 = cols[4].number_input("G120", min_value=0.0, value=110.0, step=1.0)

    use_g90 = st.checkbox("90分値を使用する", value=True)

    if st.button("単独計算を実行", type="primary"):
        glucose = np.array([g0, g30, g60, g90 if use_g90 else np.nan, g120], dtype=float)
        weights = np.ones(len(DEFAULT_GLUCOSE_COLS), dtype=float)

        with st.spinner("計算中です..."):
            res = fit_one_subject_woi(glucose, weights, tdata=DEFAULT_TDATA, outtime=DEFAULT_TDATA.copy())

        st.success("計算が完了しました。")
        display_result_summary(res)

        result_df = pd.DataFrame([{DEFAULT_ID_COL: subject_id, **dict(zip(DEFAULT_GLUCOSE_COLS, glucose)), **res}])
        st.dataframe(result_df, use_container_width=True)

        st.download_button(
            label="結果Excelをダウンロード",
            data=dataframe_to_excel_bytes(result_df, weights),
            file_name=f"DIwoI_single_result_v{APP_VERSION}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# ---------------------------------------------------------
# Batch calculation
# ---------------------------------------------------------
with batch_tab:
    st.subheader("Excel一括計算")
    st.markdown(
        "Excelファイルには `subject`, `G0`, `G30`, `G60`, `G90`, `G120` の列を含めてください。"
        "列名は下で変更できます。"
    )

    uploaded = st.file_uploader("ExcelまたはCSVファイルをアップロード", type=["xlsx", "xls", "csv"])

    with st.expander("列名設定", expanded=False):
        id_col = st.text_input("ID列名", value=DEFAULT_ID_COL)
        g_cols_text = st.text_input("血糖列名：カンマ区切り", value=", ".join(DEFAULT_GLUCOSE_COLS))
        glucose_cols = [c.strip() for c in g_cols_text.split(",") if c.strip()]

    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                input_df = pd.read_csv(uploaded)
            else:
                input_df = pd.read_excel(uploaded)

            st.write("読み込んだデータの先頭")
            st.dataframe(input_df.head(), use_container_width=True)

            missing_cols = [c for c in glucose_cols if c not in input_df.columns]
            if missing_cols:
                st.error(f"必要な血糖列が見つかりません: {missing_cols}")
            elif st.button("Excel一括計算を実行", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                tdata = np.array([int(c.replace("G", "")) for c in glucose_cols], dtype=float)

                def progress_callback(i, n, subj, res):
                    progress_bar.progress(i / n)
                    status_text.text(
                        f"{i}/{n}: {subj} | "
                        f"mDI raw={res.get('mDI_woI_raw_sigma_x_si', np.nan):.6g} | "
                        f"mDI 10^-4={res.get('mDI_woI_10minus4_mL_per_mU_per_min', np.nan):.6g}"
                    )

                with st.spinner("一括計算中です。症例数が多い場合は時間がかかります..."):
                    results, weights_G = fit_dataframe_woi(
                        input_df,
                        id_col=id_col,
                        glucose_cols=glucose_cols,
                        tdata=tdata,
                        progress_callback=progress_callback,
                    )

                st.success("一括計算が完了しました。")
                st.write("結果")
                st.dataframe(results, use_container_width=True)

                key_cols = [
                    id_col,
                    "sigma_raw",
                    "si_raw",
                    "mDI_woI_raw_sigma_x_si",
                    "mDI_woI_10minus4_mL_per_mU_per_min",
                    "WRSS",
                    "RSQ_matlab_style",
                    "fit_status",
                ]
                key_cols = [c for c in key_cols if c in results.columns]
                st.write("主要結果")
                st.dataframe(results[key_cols], use_container_width=True)

                st.download_button(
                    label="結果Excelをダウンロード",
                    data=dataframe_to_excel_bytes(results, weights_G),
                    file_name=f"DIwoI_batch_results_v{APP_VERSION}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        except Exception as e:
            st.exception(e)

# ---------------------------------------------------------
# Template download
# ---------------------------------------------------------
with template_tab:
    st.subheader("Excel入力テンプレート")
    st.markdown(
        "下のボタンから、Excel一括計算用テンプレートをダウンロードできます。"
        "血糖値は mg/dL で入力してください。"
    )

    st.download_button(
        label="入力テンプレートをダウンロード",
        data=make_template_excel(),
        file_name=f"DIwoI_input_template_v{APP_VERSION}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown(
        """
テンプレートの必須列：

| 列名 | 内容 |
|---|---|
| subject | 症例ID |
| G0 | 0分血糖 mg/dL |
| G30 | 30分血糖 mg/dL |
| G60 | 60分血糖 mg/dL |
| G90 | 90分血糖 mg/dL |
| G120 | 120分血糖 mg/dL |
        """
    )

# ---------------------------------------------------------
# About
# ---------------------------------------------------------
with about_tab:
    st.subheader("About")
    st.markdown(
        f"""
**{APP_TITLE}**  
Version **{APP_VERSION}**

このアプリは、OGTT中の血糖値のみを用いて、インスリンを使わない model-derived disposition index、すなわち **mDI-woI** を推定する研究用計算ツールです。

主な出力：

- `sigma_raw`
- `si_raw`
- `mDI_woI_raw_sigma_x_si`
- `mDI_woI_mL_per_mU_per_min`
- `mDI_woI_10minus4_mL_per_mU_per_min`
- `G*_sim`, `I*_sim`
- `WRSS`, `RSQ_matlab_style`, `AIC_matlab_style`

**注意事項**  
{DISCLAIMER}
        """
    )
