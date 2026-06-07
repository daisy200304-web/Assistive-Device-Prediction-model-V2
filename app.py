import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# ==========================================
# 頁面設定
# ==========================================
st.set_page_config(
    page_title="輔具需求預測系統",
    page_icon="🦽",
    layout="wide"
)

# ==========================================
# 美化 CSS
# ==========================================
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px;
}

body {
    background-color: #f5f7fb;
}

.hero {
    background: linear-gradient(135deg, #1f6feb 0%, #3b82f6 45%, #60a5fa 100%);
    padding: 28px 32px;
    border-radius: 20px;
    color: white;
    margin-bottom: 22px;
    box-shadow: 0 10px 25px rgba(31, 111, 235, 0.22);
}

.hero-title {
    font-size: 30px;
    font-weight: 800;
    margin-bottom: 6px;
}

.hero-subtitle {
    font-size: 16px;
    opacity: 0.95;
    margin-bottom: 0px;
}

.section-title {
    font-size: 18px;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 10px;
}

.result-success {
    background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
    border: 1px solid #bbf7d0;
    border-left: 8px solid #22c55e;
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 4px 14px rgba(34, 197, 94, 0.10);
}

.result-warning {
    background: linear-gradient(135deg, #fffbeb 0%, #fefce8 100%);
    border: 1px solid #fde68a;
    border-left: 8px solid #f59e0b;
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 4px 14px rgba(245, 158, 11, 0.10);
}

.result-danger {
    background: linear-gradient(135deg, #fff1f2 0%, #fef2f2 100%);
    border: 1px solid #fecdd3;
    border-left: 8px solid #ef4444;
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 4px 14px rgba(239, 68, 68, 0.10);
}

.result-title {
    font-size: 22px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 8px;
}

.result-text {
    font-size: 18px;
    font-weight: 700;
    color: #334155;
}

.disclaimer {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 12px 16px;
    color: #475569;
    font-size: 14px;
    margin-top: 12px;
}

div.stButton > button:first-child {
    height: 3rem;
    border-radius: 14px;
    font-weight: 700;
    font-size: 16px;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.22);
}

@media print {
    @page {
        size: A4 portrait;
        margin: 1cm;
    }
    body {
        zoom: 0.85;
    }
    button {
        display: none !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# Hero 區塊
# ==========================================
st.markdown("""
<div class="hero">
    <div class="hero-title">🦽輔具需求預測系統</div>
    <div class="hero-subtitle">
        依據個案基本資料、臨床特徵與診斷代碼，預測可能需要的行動輔具類別。
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 載入模型
# ==========================================
model_path = "assistive_device_v2.pkl"

if not os.path.exists(model_path):
    st.error(f"找不到模型檔案 `{model_path}`，請確認檔案已上傳到 GitHub repository。")
    st.stop()

try:
    pipeline = joblib.load(model_path)
except Exception as e:
    st.error(f"模型檔案載入失敗，請確認 `{model_path}` 是否為正確的 pkl 檔案。詳細錯誤：{e}")
    st.stop()

# ==========================================
# 讀取模型內容
# ==========================================
try:
    model = pipeline["model"]
    features = pipeline["features"]
except Exception as e:
    st.error(f"模型檔案格式不正確，pkl 內應包含 `model` 與 `features`。詳細錯誤：{e}")
    st.stop()

median_values = pipeline.get("median_values", None)
imputer = pipeline.get("imputer", None)

label_map = pipeline.get(
    "label_map",
    {
        0: "🟢 建議評估：類別 1（輕度輔具／單拐等）",
        1: "🟡 建議評估：類別 2（中度輔具／助行器等）",
        2: "🔴 建議評估：類別 3（重度輔具／輪椅等）"
    }
)

# ==========================================
# 系統提醒
# 已移除「目前模型」與「預測類別」資訊列
# ==========================================
st.markdown("""
<div class="disclaimer">
    ⚠️ 本系統為研究展示與初步風險評估用途，實際輔具處方仍需由臨床專業人員進行完整評估。
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 輸入區
# ==========================================
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<div class="section-title">① 基本生理數值</div>', unsafe_allow_html=True)

    age = st.number_input("年齡 age", value=75.0, format="%.1f", step=0.1)
    ht = st.number_input("身高 HT_y（cm）", value=160.0, format="%.1f", step=0.1)
    wt = st.number_input("體重 WT_y（kg）", value=60.0, format="%.1f", step=0.1)

    if ht > 0:
        bmi = wt / ((ht / 100) ** 2)
    else:
        bmi = 0.0

    st.number_input(
        "BMI（系統自動計算）",
        value=float(round(bmi, 1)),
        format="%.1f",
        disabled=True
    )

with col2:
    st.markdown('<div class="section-title">② 臨床評估與身分指標</div>', unsafe_allow_html=True)

    category_of_care = st.number_input(
        "照護類別 Category of care",
        value=1.0,
        format="%.1f",
        step=0.1
    )

    disability_level = st.number_input(
        "失能等級 Disability level",
        value=2.0,
        format="%.1f",
        step=0.1
    )

    func_type = st.number_input(
        "門診科別 FUNC_TYPE",
        value=1.0,
        format="%.1f",
        step=0.1
    )

    master_placement = st.number_input(
        "主要安置 master placement",
        value=0.0,
        format="%.1f",
        step=0.1
    )

    official_rank = st.number_input(
        "官方等級 official rank",
        value=1.0,
        format="%.1f",
        step=0.1
    )

    address = st.number_input(
        "居住地代碼 address",
        value=1.0,
        format="%.1f",
        step=0.1
    )

with col3:
    st.markdown('<div class="section-title">③ 診斷代碼 ICD-10</div>', unsafe_allow_html=True)
    st.caption("若無相關次要診斷，請維持 0.0")

    icd_main = st.number_input(
        "主診斷 ICD10CM_CODE",
        value=0.0,
        format="%.1f",
        step=0.1
    )

    icd_1 = st.number_input(
        "次診斷 1 ICD10CM_CODE_1",
        value=0.0,
        format="%.1f",
        step=0.1
    )

    icd_2 = st.number_input(
        "次診斷 2 ICD10CM_CODE_2",
        value=0.0,
        format="%.1f",
        step=0.1
    )

    icd_3 = st.number_input(
        "次診斷 3 ICD10CM_CODE_3",
        value=0.0,
        format="%.1f",
        step=0.1
    )

    icd_4 = st.number_input(
        "次診斷 4 ICD10CM_CODE_4",
        value=0.0,
        format="%.1f",
        step=0.1
    )

st.markdown("<hr style='margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)

# ==========================================
# 預測區
# ==========================================
btn_col, result_col = st.columns([1, 2], gap="large")

with btn_col:
    st.markdown('<div class="section-title">④ 執行預測</div>', unsafe_allow_html=True)

    predict_btn = st.button(
        "🚀 執行輔具需求預測",
        type="primary",
        use_container_width=True
    )

    st.caption("點擊後系統將依據輸入資料產生預測結果與各類別機率。")

with result_col:
    if not predict_btn:
        st.info("請點擊左側按鈕執行預測，結果將顯示於此。")

    if predict_btn:

        input_dict = {
            "Category of care": category_of_care,
            "Disability level": disability_level,
            "FUNC_TYPE": func_type,
            "HT_y": ht,
            "ICD10CM_CODE": icd_main,
            "ICD10CM_CODE_1": icd_1,
            "ICD10CM_CODE_2": icd_2,
            "ICD10CM_CODE_3": icd_3,
            "ICD10CM_CODE_4": icd_4,
            "WT_y": wt,
            "address": address,
            "age": age,
            "master placement": master_placement,
            "official rank": official_rank,
            "BMI": bmi,

            # assistive_device_v2.pkl 需要的額外特徵
            # 代表該診斷欄位是否有填寫
            "has_ICD10CM_CODE": 1 if icd_main != 0 else 0,
            "has_ICD10CM_CODE_1": 1 if icd_1 != 0 else 0,
            "has_ICD10CM_CODE_2": 1 if icd_2 != 0 else 0,
            "has_ICD10CM_CODE_3": 1 if icd_3 != 0 else 0,
            "has_ICD10CM_CODE_4": 1 if icd_4 != 0 else 0
        }

        try:
            input_df = pd.DataFrame([input_dict])

            # 若模型 features 裡有前端沒有產生的欄位，先補成 NaN
            for col in features:
                if col not in input_df.columns:
                    input_df[col] = np.nan

            # 確保欄位順序與訓練模型時一致
            input_df = input_df[features]

            # 全部轉成數值
            input_df = input_df.apply(pd.to_numeric, errors="coerce")

            # 補缺失值
            if median_values is not None:
                input_for_model = input_df.fillna(median_values)
            elif imputer is not None:
                input_for_model = imputer.transform(input_df)
            else:
                st.error("pkl 檔案內沒有 `median_values` 或 `imputer`，無法處理缺失值。請重新產生 pkl。")
                st.stop()

            # 模型預測
            prediction = int(model.predict(input_for_model)[0])
            result_text = label_map.get(prediction, "未知類別")

            if prediction == 0:
                result_class = "result-success"
            elif prediction == 1:
                result_class = "result-warning"
            else:
                result_class = "result-danger"

            st.markdown(f"""
            <div class="{result_class}">
                <div class="result-title">分析完成</div>
                <div class="result-text">{result_text}</div>
            </div>
            """, unsafe_allow_html=True)

            # 預測機率
            st.markdown("#### 各類別預測機率")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_for_model)[0]

                labels = [
                    "類別 1：輕度輔具／單拐等",
                    "類別 2：中度輔具／助行器等",
                    "類別 3：重度輔具／輪椅等"
                ]

                for label, p in zip(labels, proba):
                    st.write(f"**{label}**：{p * 100:.1f}%")
                    st.progress(float(p))

                prob_df = pd.DataFrame({
                    "輔具類別": labels,
                    "預測機率": [f"{p * 100:.1f}%" for p in proba]
                })

                with st.expander("查看機率表格"):
                    st.table(prob_df)

            with st.expander("查看本次輸入資料"):
                st.dataframe(input_df, use_container_width=True)

            st.markdown("""
            <div class="disclaimer">
                臨床提醒：模型結果應作為輔具需求初步篩檢與風險分層參考，
                不應取代專業人員的身體功能評估、環境評估與個案需求訪談。
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"預測時發生錯誤，請檢查輸入格式。詳細錯誤：{e}")
