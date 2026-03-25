import pandas as pd
import streamlit as st

from model_core import (
    estimate_clamp_di_from_glucose,
    fit_quality_label,
    batch_estimate_from_dataframe,
    compute_weights_from_dataframe,
)

st.set_page_config(
    page_title="OGTT血糖のみからDI推定",
    page_icon="🧪",
    layout="wide"
)

# =========================================================
# 単例入力で使う固定 weights_G
# notebookで実際に使った5個の値に置き換えてください
# 例: [0.081, 0.064, 0.057, 0.055, 0.061]
# =========================================================
SINGLE_WEIGHTS_G = [0.06411335, 0.02705089, 0.01643665, 0.01263463, 0.011494]

PRESETS = {
    "手入力": {"g0": 90.0, "g30": 150.0, "g60": 170.0, "g90": 140.0, "g120": 110.0},
    "正常型の例": {"g0": 88.0, "g30": 135.0, "g60": 122.0, "g90": 103.0, "g120": 92.0},
    "境界型の例": {"g0": 95.0, "g30": 162.0, "g60": 178.0, "g90": 154.0, "g120": 136.0},
    "糖尿病型の例": {"g0": 118.0, "g30": 212.0, "g60": 246.0, "g90": 228.0, "g120": 204.0},
}


def html_table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0, classes="simple-table")


st.markdown(
    """
    <style>
    .simple-table {
        border-collapse: collapse;
        width: 100%;
        font-size: 15px;
    }
    .simple-table th, .simple-table td {
        border: 1px solid #ddd;
        padding: 8px 10px;
        text-align: center;
    }
    .simple-table th {
        background-color: #f7f7f7;
        font-weight: 600;
    }
    .simple-table tr:nth-child(even) {
        background-color: #fbfbfb;
    }
    .brand {
        font-size: 14px;
        color: #666;
        margin-bottom: 0.2rem;
    }
    .version {
        font-size: 13px;
        color: #888;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="brand">神戸大学臨床糖尿病グループ</div>', unsafe_allow_html=True)
st.title("OGTT血糖のみから Clamp DI を推定")
st.markdown('<div class="version">Version 1.1</div>', unsafe_allow_html=True)
st.caption("75g OGTT の 0, 30, 60, 90, 120分血糖値から、mDI_woI と推定 Clamp DI を算出します。")


with st.sidebar:
    st.header("入力")
    mode = st.radio("モード", ["1例入力", "CSV一括入力"])

    with st.expander("重み設定"):
        st.write("単例入力で使用する固定 weights_G")
        st.code(str(SINGLE_WEIGHTS_G))
        st.write("※ notebookで実際に使った5個の値に置き換えてください。")
        st.write("※ CSV一括入力では、アップロードCSV全体から 1 / SD を自動計算します。")

    if mode == "1例入力":
        case_id = st.text_input("症例ID", value="Case-001")
        preset_name = st.selectbox("プリセット", list(PRESETS.keys()))
        preset = PRESETS[preset_name]

        g0 = st.number_input("0分血糖 (mg/dL)", min_value=1.0, value=float(preset["g0"]), step=1.0)
        g30 = st.number_input("30分血糖 (mg/dL)", min_value=1.0, value=float(preset["g30"]), step=1.0)
        g60 = st.number_input("60分血糖 (mg/dL)", min_value=1.0, value=float(preset["g60"]), step=1.0)
        g90 = st.number_input("90分血糖 (mg/dL)", min_value=1.0, value=float(preset["g90"]), step=1.0)
        g120 = st.number_input("120分血糖 (mg/dL)", min_value=1.0, value=float(preset["g120"]), step=1.0)

        run_btn = st.button("推定する", type="primary", use_container_width=True)

    else:
        st.write("必要列: `CaseID, G0, G30, G60, G90, G120`")
        uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])
        run_btn = st.button("一括推定する", type="primary", use_container_width=True)

    with st.expander("前提・注意事項"):
        st.write("・対象は 75g OGTT、血糖単位は mg/dL を想定しています。")
        st.write("・主結果は log-log 回帰式による推定 Clamp DI です。")
        st.write("・研究用の補助ツールであり、単独での診療判断用途は想定していません。")


if "last_single_result" not in st.session_state:
    st.session_state["last_single_result"] = None

if "last_batch_result" not in st.session_state:
    st.session_state["last_batch_result"] = None

if "last_batch_weights" not in st.session_state:
    st.session_state["last_batch_weights"] = None


# =========================================================
# 1例入力モード
# =========================================================
if mode == "1例入力":
    if run_btn:
        with st.spinner("計算中です..."):
            st.session_state["last_single_result"] = estimate_clamp_di_from_glucose(
                g0, g30, g60, g90, g120,
                weights_G=SINGLE_WEIGHTS_G
            )

    res = st.session_state["last_single_result"]

    if res is None:
        st.info("左側で血糖値を入力して「推定する」を押してください。")
    else:
        for msg in res["messages"]:
            st.warning(msg)

        if not res["ok"]:
            st.error(f"推定に失敗しました。fit_status: {res['fit_status']}")
        else:
            fit_res = res["fit_res"]
            quality = fit_quality_label(fit_res["RSQ"])

            c1, c2, c3 = st.columns(3)
            c1.metric("mDI_woI", f"{res['mDI_woI']:.4f}")
            c2.metric("推定 Clamp DI", f"{res['Clamp_DI_pred_loglog']:.3f}")
            c3.metric("フィット判定", quality)

            tab1, tab2, tab3, tab4 = st.tabs(["結果", "フィット", "詳細", "ダウンロード"])

            with tab1:
                st.subheader("主結果")
                st.write(f"**症例ID**: {case_id}")
                st.write(f"**主推定値（log-log）**: {res['Clamp_DI_pred_loglog']:.3f}")
                st.write(f"参考値（線形回帰）: {res['Clamp_DI_pred_linear']:.3f}")
                st.write(f"fit_status: {res['fit_status']}")

                if quality == "良好":
                    st.success("モデル適合は良好です。")
                elif quality == "中等度":
                    st.warning("モデル適合は中等度です。結果解釈は慎重に行ってください。")
                else:
                    st.warning("モデル適合は十分ではありません。参考値として扱ってください。")

            with tab2:
                st.subheader("入力血糖とモデル予測血糖")

                plot_df = pd.DataFrame({
                    "Time": [0, 30, 60, 90, 120],
                    "Observed": [g0, g30, g60, g90, g120],
                    "Predicted": [
                        fit_res["G_pred_0"],
                        fit_res["G_pred_30"],
                        fit_res["G_pred_60"],
                        fit_res["G_pred_90"],
                        fit_res["G_pred_120"],
                    ]
                })
                plot_df["Residual"] = plot_df["Predicted"] - plot_df["Observed"]

                st.markdown(html_table(plot_df), unsafe_allow_html=True)

            with tab3:
                st.subheader("詳細パラメータ")
                detail_df = pd.DataFrame([{
                    "sigma": fit_res["sigma"],
                    "si": fit_res["si"],
                    "WRSS": fit_res["WRSS"],
                    "RSQ": fit_res["RSQ"],
                    "AIC": fit_res["AIC"],
                    "fit_status": res["fit_status"],
                    "weights_G": str(SINGLE_WEIGHTS_G),
                }])
                st.markdown(html_table(detail_df), unsafe_allow_html=True)

            with tab4:
                st.subheader("結果のダウンロード")
                csv_df = pd.DataFrame([{
                    "CaseID": case_id,
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
                    "weights_G": str(SINGLE_WEIGHTS_G),
                }])

                st.markdown(html_table(csv_df), unsafe_allow_html=True)

                csv_data = csv_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="CSVをダウンロード",
                    data=csv_data,
                    file_name=f"{case_id}_ClampDI_prediction.csv",
                    mime="text/csv",
                    use_container_width=True
                )


# =========================================================
# CSV一括入力モード
# =========================================================
else:
    st.subheader("CSV一括入力")
    st.write("必要列: `CaseID, G0, G30, G60, G90, G120`")

    sample_df = pd.DataFrame([
        {"CaseID": "Case-001", "G0": 90, "G30": 150, "G60": 170, "G90": 140, "G120": 110},
        {"CaseID": "Case-002", "G0": 95, "G30": 162, "G60": 178, "G90": 154, "G120": 136},
    ])
    st.markdown("#### CSV形式の例")
    st.markdown(html_table(sample_df), unsafe_allow_html=True)

    if run_btn:
        if uploaded_file is None:
            st.error("CSVファイルをアップロードしてください。")
        else:
            with st.spinner("一括推定中です..."):
                try:
                    df_in = pd.read_csv(uploaded_file)

                    batch_weights = compute_weights_from_dataframe(
                        df_in,
                        glucose_cols=("G0", "G30", "G60", "G90", "G120")
                    )
                    st.session_state["last_batch_weights"] = batch_weights.tolist()

                    out_rows = batch_estimate_from_dataframe(
                        df_in,
                        id_col="CaseID",
                        col_g0="G0",
                        col_g30="G30",
                        col_g60="G60",
                        col_g90="G90",
                        col_g120="G120",
                        weights_G=batch_weights
                    )
                    st.session_state["last_batch_result"] = pd.DataFrame(out_rows)

                except Exception as e:
                    st.session_state["last_batch_result"] = None
                    st.error(str(e))

    batch_df = st.session_state["last_batch_result"]

    if st.session_state["last_batch_weights"] is not None:
        st.markdown("#### このCSVから自動計算された weights_G")
        st.code(str(st.session_state["last_batch_weights"]))

    if batch_df is not None:
        st.markdown("#### 推定結果")
        st.markdown(html_table(batch_df), unsafe_allow_html=True)

        csv_data = batch_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="一括結果CSVをダウンロード",
            data=csv_data,
            file_name="batch_ClampDI_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )