import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# =========================================================
# 1. 設定
# =========================================================

TDATA = np.array([0, 30, 60, 90, 120], dtype=float)
OUTTIME = TDATA.copy()

# Matlab側の初期状態
INIT_STATE = np.array([80.1842, 5.8462, 60.9341, 443.7764], dtype=float)

# Matlab側の初期推定値
THETA0 = np.array([1.0, 0.8], dtype=float)   # [sigma, si]

# Matlab側 bounds
LB = np.array([0.01, 0.01], dtype=float)
UB = np.array([10.0, 100.0], dtype=float)

# Matlab側に合わせたパラメータ
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

BIG_PENALTY = 1e12

# Clamp DI換算式
LINEAR_INTERCEPT = 9.197
LINEAR_SLOPE = 95.877

LOG_INTERCEPT = 4.222
LOG_SLOPE = 0.576


# =========================================================
# 2. 補助関数
# =========================================================

def safe_clip_state(y):
    """
    状態変数の暴走を防ぐ
    """
    y = np.asarray(y, dtype=float).copy()
    if y.shape[0] != 4:
        return np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)

    if not np.all(np.isfinite(y)):
        return np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)

    y[0] = max(y[0], 0.0)  # G
    y[1] = max(y[1], 0.0)  # I
    y[2] = max(y[2], 0.0)  # N5
    y[3] = max(y[3], 0.0)  # N6

    upper = np.array([1e4, 1e4, 1e7, 1e7], dtype=float)
    y = np.minimum(y, upper)

    return y


def ogtt_rate_piecewise(t, p):
    """
    Matlabの
    ((t>0)-(t>t1))*...
    をPython用に安全に書き換えたもの
    """
    t1, t2, t3 = p["t1"], p["t2"], p["t3"]
    a1, a2, a3 = p["a1"], p["a2"], p["a3"]

    if t <= 0:
        return 0.0
    elif t <= t1:
        return t * a1 / t1
    elif t <= t2:
        return ((t - t2) * (a2 - a1) / (t2 - t1)) + a2
    elif t <= t3:
        return (t - t3) * (a3 - a2) / (t3 - t2)
    else:
        return 0.0


def compute_weights_from_dataframe(
    df,
    glucose_cols=("G0", "G30", "G60", "G90", "G120")
):
    """
    Jupyter版に合わせて、データ全体から weights_G = 1 / SD を作る
    """
    work = df.copy()
    for c in glucose_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    GDATA = work[list(glucose_cols)].to_numpy(dtype=float)
    var_G = np.nanstd(GDATA, axis=0, ddof=1)

    var_G = np.where(~np.isfinite(var_G), 1.0, var_G)
    var_G = np.where(var_G <= 0, 1.0, var_G)

    weights_G = 1.0 / var_G
    return weights_G


def validate_glucose_inputs(glucose_obs):
    msgs = []
    if np.any(~np.isfinite(glucose_obs)):
        msgs.append("血糖値に欠損または非数値が含まれています。")
    if np.any(glucose_obs <= 0):
        msgs.append("血糖値は正の値で入力してください。")

    labels = ["0分", "30分", "60分", "90分", "120分"]
    for lab, val in zip(labels, glucose_obs):
        if np.isfinite(val) and (val < 40 or val > 400):
            msgs.append(f"{lab}血糖 {val:.1f} mg/dL は通常範囲から外れています。")
    return msgs


def fit_quality_label(rsq):
    if not np.isfinite(rsq):
        return "判定不能"
    if rsq >= 0.8:
        return "良好"
    if rsq >= 0.5:
        return "中等度"
    return "参考"


# =========================================================
# 3. ODE本体
# =========================================================

def gi_ode_universal(t, y, p):
    """
    Matlab GI_ode_universal_new.m をできるだけ忠実に、
    かつ Python で落ちにくいようにした版
    y = [G, I, N5, N6]
    """

    y = safe_clip_state(y)
    G, I, N5, N6 = y

    if not np.all(np.isfinite([G, I, N5, N6])):
        return np.array([BIG_PENALTY, BIG_PENALTY, BIG_PENALTY, BIG_PENALTY], dtype=float)

    Eg0 = 0.0118
    k = 0.4861
    BV = 7200.0
    b = 1553.6

    Mmax = 1.0
    alpha_M = 150.0
    kM = 2.0

    hepa_bar = 15.443
    hepa_k = 0.27
    hepa_b = -3.54277

    alpha_max = 6.0
    alpha_k = 0.4
    alpha_b = -0.5

    HGP_b = 0.104166

    GF_bar = 4.45
    kGF = 16.0
    alpha_GF = 260.0
    shGF = -89.0
    GF_b = 1.78

    ca_bar = 2.0
    kca = 4.0
    alpha_ca = 0.62
    ca_b = 0.07

    cmd_factor = 150.0
    cmd_b = 0.0635
    cik = 4.0
    cialpha = 1.0

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

    ts = 60.0
    unit_con = 0.00069444

    sigma = float(p["sigma"])
    si = float(p["si"])

    if (not np.isfinite(sigma)) or (not np.isfinite(si)) or sigma <= 0 or si <= 0:
        return np.array([BIG_PENALTY, BIG_PENALTY, BIG_PENALTY, BIG_PENALTY], dtype=float)

    try:
        M = Mmax * (G ** kM) / (alpha_M ** kM + G ** kM)
        OGTT_rate = ogtt_rate_piecewise(t, p)

        hepa_max = hepa_bar / (hepa_k + si) + hepa_b
        alpha_HGP = alpha_max / (alpha_k + si) + alpha_b
        HGP = hepa_max / (alpha_HGP + I * p["hepasi"]) + HGP_b

        base_gf = max(G - shGF, 0.0)
        GF = (GF_bar * (base_gf ** kGF) / (alpha_GF ** kGF + base_gf ** kGF)) + GF_b

        mterm = max(M + p["gamma_bar"] * p["gamma"], 0.0)
        ci = ca_bar * (mterm ** kca) / (alpha_ca ** kca + mterm ** kca) + ca_b

        cmd = cmd_factor * (ci ** cik) / (cialpha ** cik + ci ** cik) + cmd_b

        r2 = p["r20"] * ci / (ci + Kp2)
        r3 = sigma * GF * r30 * ci / (ci + Kp2)

        denom1 = (3.0 * k1 * cmd + rm1)
        denom2 = (2.0 * k1 * cmd + km1)
        denom3 = (2.0 * km1 + k1 * cmd)
        denom4 = (3.0 * km1 + u1)

        if min(denom1, denom2, denom3, denom4) <= 0:
            return np.array([BIG_PENALTY, BIG_PENALTY, BIG_PENALTY, BIG_PENALTY], dtype=float)

        N1_C = km1 / denom1
        N1_D = r1 / denom1

        N2_E = 3.0 * k1 * cmd / denom2
        N2_F = 2.0 * km1 / denom2

        N3_L = 2.0 * k1 * cmd / denom3
        N3_N = 3.0 * km1 / denom3

        CN4 = (k1 * cmd / denom4)

        denom_CN3 = (1.0 - N3_N * CN4)
        if abs(denom_CN3) < 1e-12:
            return np.array([BIG_PENALTY, BIG_PENALTY, BIG_PENALTY, BIG_PENALTY], dtype=float)

        CN3 = N3_L / denom_CN3

        denom_CN2 = (1.0 - N2_F * CN3)
        if abs(denom_CN2) < 1e-12:
            return np.array([BIG_PENALTY, BIG_PENALTY, BIG_PENALTY, BIG_PENALTY], dtype=float)

        CN2 = N2_E / denom_CN2

        denom_CN1 = (1.0 - N1_C * CN2)
        if abs(denom_CN1) < 1e-12:
            return np.array([BIG_PENALTY, BIG_PENALTY, BIG_PENALTY, BIG_PENALTY], dtype=float)

        CN1 = N1_D / denom_CN1

        N1 = CN1 * N5
        N2 = CN2 * N1
        N3 = CN3 * N2
        N4 = CN4 * N3
        NF = u1 * N4 / u2
        NR = (u2 / u3) * NF

        ISR = ts * 9.0 * (u3 * NR)

        dGdt = HGP + OGTT_rate - (Eg0 + unit_con * si * I) * G
        dIdt = b * ISR / BV - k * I
        dN5dt = ts * (rm1 * CN1 * N5 - (r1 + rm2) * N5 + r2 * N6)
        dN6dt = ts * (r3 + rm2 * N5 - (rm3 + r2) * N6)

        out = np.array([dGdt, dIdt, dN5dt, dN6dt], dtype=float)

        if not np.all(np.isfinite(out)):
            return np.array([BIG_PENALTY, BIG_PENALTY, BIG_PENALTY, BIG_PENALTY], dtype=float)

        return out

    except Exception:
        return np.array([BIG_PENALTY, BIG_PENALTY, BIG_PENALTY, BIG_PENALTY], dtype=float)


# =========================================================
# 4. シミュレーション
# =========================================================

def solve_model_with_runin(theta, data_t, outtime, init_state, odeparams_base):
    theta = np.asarray(theta, dtype=float)
    if theta.shape[0] != 2:
        return None

    sigma, si = theta[0], theta[1]

    if (not np.isfinite(sigma)) or (not np.isfinite(si)):
        return None
    if sigma <= 0 or si <= 0:
        return None

    p = dict(odeparams_base)
    p["sigma"] = float(sigma)
    p["si"] = float(si)

    y0 = np.asarray(init_state, dtype=float)
    if not np.all(np.isfinite(y0)):
        return None

    try:
        sol1 = solve_ivp(
            fun=lambda t, y: gi_ode_universal(t, y, p),
            t_span=(0.0, 1440.0),
            y0=y0,
            method="BDF",
            rtol=1e-5,
            atol=1e-8,
            max_step=10.0
        )
        if (not sol1.success) or (sol1.y.shape[1] == 0):
            return None

        init2 = sol1.y[:, -1]
        init2 = safe_clip_state(init2)
        if not np.all(np.isfinite(init2)):
            return None

        t0 = float(np.min(data_t))
        t1 = float(np.max(data_t))

        sol2 = solve_ivp(
            fun=lambda t, y: gi_ode_universal(t, y, p),
            t_span=(t0, t1),
            y0=init2,
            method="BDF",
            dense_output=True,
            rtol=1e-5,
            atol=1e-8,
            max_step=5.0
        )
        if not sol2.success:
            return None

        y_data = sol2.sol(data_t)
        y_out = sol2.sol(outtime)

        if (y_data is None) or (y_out is None):
            return None

        if not np.all(np.isfinite(y_data)) or not np.all(np.isfinite(y_out)):
            return None

        Gpred = y_data[0, :]
        Ipred = y_data[1, :]
        Gsim = y_out[0, :]
        Isim = y_out[1, :]

        return {
            "Gpred": Gpred,
            "Ipred": Ipred,
            "Gsim": Gsim,
            "Isim": Isim,
        }

    except Exception:
        return None


# =========================================================
# 5. cost function
# =========================================================

def ogtt_cost_function_universal_woi(theta, data, costparams, odeparams_base):
    theta = np.asarray(theta, dtype=float)

    if theta.shape[0] != 2:
        return BIG_PENALTY, BIG_PENALTY, None, None

    if np.any(~np.isfinite(theta)):
        return BIG_PENALTY, BIG_PENALTY, None, None

    if np.any(theta < costparams["LB"]) or np.any(theta > costparams["UB"]):
        return BIG_PENALTY, BIG_PENALTY, None, None

    sim = solve_model_with_runin(
        theta=theta,
        data_t=data["t"],
        outtime=data["outtime"],
        init_state=costparams["init"],
        odeparams_base=odeparams_base
    )

    if sim is None:
        return BIG_PENALTY, BIG_PENALTY, None, None

    Gpred = sim["Gpred"]
    Gsim = sim["Gsim"]
    Isim = sim["Isim"]

    resid = Gpred - data["G"]

    wrss = np.sum(costparams["weights_G"] * resid ** 2)
    true_err = np.sum(resid ** 2)

    if (not np.isfinite(wrss)) or (not np.isfinite(true_err)):
        return BIG_PENALTY, BIG_PENALTY, None, None

    return float(wrss), float(true_err), Gsim, Isim


# =========================================================
# 6. 1症例フィット
# =========================================================

def nan_result(status):
    return {
        "sigma": np.nan,
        "si": np.nan,
        "WRSS": np.nan,
        "true_error": np.nan,
        "RSQ": np.nan,
        "AIC": np.nan,
        "mDI_woI": np.nan,
        "G_pred_0": np.nan, "G_pred_30": np.nan, "G_pred_60": np.nan, "G_pred_90": np.nan, "G_pred_120": np.nan,
        "I_pred_0": np.nan, "I_pred_30": np.nan, "I_pred_60": np.nan, "I_pred_90": np.nan, "I_pred_120": np.nan,
        "fit_status": status
    }


def fit_one_subject_matlab_like(glucose_obs, weights_G=None, odeparams_base=None):
    if weights_G is None:
        raise ValueError("weights_G を指定してください。研究で使った実際の重み、または compute_weights_from_dataframe() の出力を渡してください。")
    if odeparams_base is None:
        odeparams_base = ODEPARAMS_BASE

    glucose_obs = np.asarray(glucose_obs, dtype=float)

    if np.all(~np.isfinite(glucose_obs)):
        return nan_result("missing_all")

    missing = ~np.isfinite(glucose_obs)
    data_t = TDATA[~missing]
    data_G = glucose_obs[~missing]
    weights_sub = np.asarray(weights_G, dtype=float)[~missing]

    if len(data_t) < 2:
        return nan_result("too_many_missing")

    data = {
        "t": data_t,
        "G": data_G,
        "outtime": OUTTIME.copy()
    }

    costparams = {
        "init": INIT_STATE.copy(),
        "LB": LB.copy(),
        "UB": UB.copy(),
        "weights_G": weights_sub.copy(),
    }

    def objective(theta):
        wrss, _, _, _ = ogtt_cost_function_universal_woi(theta, data, costparams, odeparams_base)
        if not np.isfinite(wrss):
            return BIG_PENALTY
        return float(wrss)

    try:
        result = minimize(
            fun=objective,
            x0=THETA0.copy(),
            method="Nelder-Mead",
            options={
                "xatol": 1e-2,
                "fatol": 1e2,
                "maxfev": 10000,
                "maxiter": 10000,
                "disp": False
            }
        )
    except Exception:
        return nan_result("optimizer_error")

    theta_hat = np.asarray(result.x, dtype=float)

    wrss, true_err, Gsim, Isim = ogtt_cost_function_universal_woi(
        theta_hat, data, costparams, odeparams_base
    )

    if Gsim is None or Isim is None:
        return nan_result("simulation_failed")

    sigma_hat = float(theta_hat[0])
    si_hat = float(theta_hat[1])

    SS = np.sum(data_G ** 2)
    RSQ = 1.0 - true_err / SS if SS > 0 else np.nan

    numpar = 2
    numpts = len(TDATA)
    AIC = 2 * numpar + numpts * np.log(numpts * wrss) if wrss > 0 else np.nan

    status = "success" if result.success else "optimizer_not_converged"

    return {
        "sigma": sigma_hat,
        "si": si_hat,
        "WRSS": wrss,
        "true_error": true_err,
        "RSQ": RSQ,
        "AIC": AIC,
        "mDI_woI": sigma_hat * si_hat,
        "G_pred_0": Gsim[0], "G_pred_30": Gsim[1], "G_pred_60": Gsim[2], "G_pred_90": Gsim[3], "G_pred_120": Gsim[4],
        "I_pred_0": Isim[0], "I_pred_30": Isim[1], "I_pred_60": Isim[2], "I_pred_90": Isim[3], "I_pred_120": Isim[4],
        "fit_status": status
    }


# =========================================================
# 7. 単例・一括推定
# =========================================================

def estimate_clamp_di_from_glucose(g0, g30, g60, g90, g120, weights_G, odeparams_base=None):
    if odeparams_base is None:
        odeparams_base = ODEPARAMS_BASE

    glucose_obs = np.array([g0, g30, g60, g90, g120], dtype=float)
    messages = validate_glucose_inputs(glucose_obs)

    fit_res = fit_one_subject_matlab_like(
        glucose_obs=glucose_obs,
        weights_G=weights_G,
        odeparams_base=odeparams_base
    )

    if fit_res["fit_status"] not in ["success", "optimizer_not_converged"]:
        return {
            "ok": False,
            "messages": messages,
            "mDI_woI": np.nan,
            "Clamp_DI_pred_linear": np.nan,
            "Clamp_DI_pred_loglog": np.nan,
            "fit_status": fit_res["fit_status"],
            "fit_res": fit_res,
        }

    mdi_woi = float(fit_res["mDI_woI"])
    if (not np.isfinite(mdi_woi)) or (mdi_woi <= 0):
        return {
            "ok": False,
            "messages": messages,
            "mDI_woI": np.nan,
            "Clamp_DI_pred_linear": np.nan,
            "Clamp_DI_pred_loglog": np.nan,
            "fit_status": fit_res["fit_status"],
            "fit_res": fit_res,
        }

    clamp_di_linear = LINEAR_INTERCEPT + LINEAR_SLOPE * mdi_woi
    log_clamp_di = LOG_INTERCEPT + LOG_SLOPE * np.log(mdi_woi)
    clamp_di_loglog = np.exp(log_clamp_di)

    return {
        "ok": True,
        "messages": messages,
        "mDI_woI": mdi_woi,
        "Clamp_DI_pred_linear": clamp_di_linear,
        "Clamp_DI_pred_loglog": clamp_di_loglog,
        "fit_status": fit_res["fit_status"],
        "fit_res": fit_res,
    }


def batch_estimate_from_dataframe(
    df,
    id_col="CaseID",
    col_g0="G0",
    col_g30="G30",
    col_g60="G60",
    col_g90="G90",
    col_g120="G120",
    weights_G=None,
    odeparams_base=None
):
    if odeparams_base is None:
        odeparams_base = ODEPARAMS_BASE

    if weights_G is None:
        weights_G = compute_weights_from_dataframe(
            df,
            glucose_cols=(col_g0, col_g30, col_g60, col_g90, col_g120)
        )

    out_rows = []

    work = df.copy()
    required_cols = [id_col, col_g0, col_g30, col_g60, col_g90, col_g120]
    missing_cols = [c for c in required_cols if c not in work.columns]
    if missing_cols:
        raise ValueError(f"CSVに必要な列がありません: {missing_cols}")

    for _, row in work.iterrows():
        case_id = row[id_col]
        try:
            g0 = float(row[col_g0])
            g30 = float(row[col_g30])
            g60 = float(row[col_g60])
            g90 = float(row[col_g90])
            g120 = float(row[col_g120])

            res = estimate_clamp_di_from_glucose(
                g0, g30, g60, g90, g120,
                weights_G=weights_G,
                odeparams_base=odeparams_base
            )
            fit_res = res["fit_res"]

            out_rows.append({
                id_col: case_id,
                "G0": g0,
                "G30": g30,
                "G60": g60,
                "G90": g90,
                "G120": g120,
                "mDI_woI": res["mDI_woI"],
                "Clamp_DI_pred_linear": res["Clamp_DI_pred_linear"],
                "Clamp_DI_pred_loglog": res["Clamp_DI_pred_loglog"],
                "sigma": fit_res["sigma"],
                "si": fit_res["si"],
                "WRSS": fit_res["WRSS"],
                "RSQ": fit_res["RSQ"],
                "AIC": fit_res["AIC"],
                "fit_status": res["fit_status"],
                "messages": " | ".join(res["messages"]) if res["messages"] else "",
            })
        except Exception as e:
            out_rows.append({
                id_col: case_id,
                "G0": row[col_g0],
                "G30": row[col_g30],
                "G60": row[col_g60],
                "G90": row[col_g90],
                "G120": row[col_g120],
                "mDI_woI": np.nan,
                "Clamp_DI_pred_linear": np.nan,
                "Clamp_DI_pred_loglog": np.nan,
                "sigma": np.nan,
                "si": np.nan,
                "WRSS": np.nan,
                "RSQ": np.nan,
                "AIC": np.nan,
                "fit_status": "error",
                "messages": str(e),
            })

    return out_rows
