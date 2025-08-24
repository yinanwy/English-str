import numpy as np
import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt


@st.cache_resource
def load_resources():
    # 加载4维预处理器和模型
    with open('xgb_model_4features.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('preprocessors_4features.pkl', 'rb') as f:
        preprocessors = pickle.load(f)
    explainer = shap.TreeExplainer(model)
    return model, explainer, preprocessors


model, explainer, preprocessors = load_resources()
scaler = preprocessors['scaler']  # 4维标准化器
normalizer = preprocessors['normalizer']  # 4维正则化器

# 用户输入界面
st.title('Credit Risk Assessment of Private Elderly-Care Institutions')
st.markdown("Please enter the indicators：")

# 输入字段
CW02 = st.number_input("Registered Capital (CW02)(unit: 10,000 CNY)")
CP05 = st.number_input("Financing history (CP05)(unit: times)")
CP02 = st.number_input("Number of Patents (CP02) (unit: times) ")
CS03 = st.number_input("Number of Tax-Paying Employees (CS03) (unit: persons) ")

if st.button('Creditworthiness Level'):
    input_data = pd.DataFrame([[CW02, CP05, CP02,CS03]],
                              columns=['CW02', 'CP05', 'CP02', 'CS03'])
    input_scaled = scaler.transform(input_data)
    input_processed = normalizer.transform(input_scaled)

    prob = model.predict_proba(input_processed)[0, 1]
    # 数值显示
    # st.success(f"**Creditworthiness Level：{prob:.4%}**")

    # 根据概率值划分层次
    if prob < 0.85:
        level = "Early Warning"
        color = "red"
    elif prob < 0.992:
        level = "Review Required"
        color = "orange"
    elif prob < 0.998:
        level = "Under Observation"
        color = "Yellow"
    else:
        level = "Prime Quality"
        color = "green"

    # 使用HTML标记和颜色显示结果
    # 只显示文字结果
    st.markdown(f"<p style='font-size:20px;'>Based on the feature values：<span style='color:{color};font-weight:bold;'>{level}</span></p>",
                unsafe_allow_html=True)

    with st.expander("Click to view data processing details"):
        st.write("Original input values：", input_data.values)
        st.write("After standardization：", input_scaled)
        st.write("After normalization：", input_processed)

    # 显示原始输入值作为参考
    st.markdown("**Current input values:**")
    st.dataframe(input_data.style.format("{:.1f}"))
