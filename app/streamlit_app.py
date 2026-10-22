"""Run with:  pip install -e ".[app]" && streamlit run app/streamlit_app.py"""
import json

import pandas as pd
import streamlit as st

from researchguard.reporting import build_report, to_markdown

st.set_page_config(page_title="ResearchGuard ML", layout="wide")
st.title("ResearchGuard ML")
st.caption("Auditable evaluation for binary-outcome tabular data. Predictive association only — "
           "not causal inference, not clinical validation. Do not upload identifiable patient data.")

file = st.file_uploader("CSV file", type="csv")
if file:
    df = pd.read_csv(file)
    st.dataframe(df.head(20))
    cols = list(df.columns)
    target = st.selectbox("Binary outcome column", cols)
    levels = sorted(df[target].dropna().astype(str).unique())
    positive = st.selectbox("Which outcome value counts as positive (the event)?", levels,
                            index=len(levels) - 1 if levels else 0)
    group = st.selectbox("Group column (repeated participants / sites), if any", [None] + cols)
    time = st.selectbox("Time column, if outcomes are time-ordered", [None] + cols)
    sub = st.multiselect("Subgroup columns to evaluate", cols)
    drop = st.multiselect("Columns to exclude from features (IDs, post-outcome variables)", cols)
    splits = st.slider("Folds", 2, 10, 5)
    if st.button("Run evaluation"):
        try:
            with st.spinner("Running..."):
                rep = build_report(df, target, group, time, sub, drop, splits,
                                   positive_class=positive)
            st.markdown(to_markdown(rep))
            st.download_button("Download JSON report", json.dumps(rep, indent=2, default=str),
                               "report.json")
        except (ValueError, NotImplementedError) as e:
            st.error(str(e))
