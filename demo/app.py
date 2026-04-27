"""Streamlit demo for LegalDrift."""

import streamlit as st
import pandas as pd
from pathlib import Path

from legaldrift import LegalDocument, EmbeddingEngine, DriftDetector, LegalConceptExtractor

st.set_page_config(page_title="LegalDrift Demo", page_icon="⚖️")

st.title("⚖️ LegalDrift")
st.markdown("**Detect semantic drift in legal documents**")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Select", ["Home", "Analyze", "Detect Drift", "Compare Methods"])

if page == "Home":
    st.markdown("""
    ## Welcome to LegalDrift

    LegalDrift detects **semantic drift** in legal documents—when the meaning changes even if the wording seems similar.

    ### Use Cases
    - Monitor contract changes over time
    - Detect regulatory compliance shifts
    - Track policy evolution
    - Identify problematic clause modifications

    ### How It Works
    1. **Embed** documents using Legal-BERT
    2. **Compare** distributions with multiple statistical tests
    3. **Combine** results via Fisher's method for robust detection
    """)

elif page == "Analyze":
    st.header("Document Analysis")

    uploaded_file = st.file_uploader("Upload a legal document (.txt)", type=["txt"])

    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
        doc = LegalDocument(text=text, document_id=uploaded_file.name)

        extractor = LegalConceptExtractor()
        concepts = extractor.extract_from_text(text)

        st.success(f"Loaded: {uploaded_file.name}")
        st.markdown(f"**Words:** {doc.word_count} | **Characters:** {doc.char_count}")

        st.subheader("Legal Concepts Detected")
        if concepts:
            for concept in sorted(concepts):
                st.write(f"- {concept}")
        else:
            st.info("No standard legal concepts detected")

elif page == "Detect Drift":
    st.header("Drift Detection")

    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader("Original Document", type=["txt"], key="v1")
    with col2:
        file2 = st.file_uploader("Updated Document", type=["txt"], key="v2")

    if file1 and file2:
        if st.button("Detect Drift"):
            text1 = file1.read().decode("utf-8")
            text2 = file2.read().decode("utf-8")

            doc1 = LegalDocument(text=text1, document_id=file1.name)
            doc2 = LegalDocument(text=text2, document_id=file2.name)

            engine = EmbeddingEngine()
            detector = DriftDetector()

            emb1 = engine.encode([text1])
            emb2 = engine.encode([text2])

            result = detector.detect(emb1, emb2)

            st.subheader("Results")

            col1, col2, col3 = st.columns(3)
            col1.metric("Drift Detected", "Yes" if result.drift_detected else "No")
            col2.metric("P-Value", f"{result.p_value:.4f}")
            col3.metric("Severity", f"{result.severity:.4f}")

            st.markdown("### Individual Test Results")
            df = pd.DataFrame([
                {"Test": name, "P-Value": data.get("p_value", data.get("statistic", 0))}
                for name, data in result.tests.items()
            ])
            st.dataframe(df)

elif page == "Compare Methods":
    st.header("Compare Detection Methods")

    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader("Baseline Document", type=["txt"], key="c1")
    with col2:
        file2 = st.file_uploader("Current Document", type=["txt"], key="c2")

    if file1 and file2:
        if st.button("Compare Methods"):
            text1 = file1.read().decode("utf-8")
            text2 = file2.read().decode("utf-8")

            from legaldrift.core.baselines import ADWIN, DDM, HDP

            engine = EmbeddingEngine()
            detector = DriftDetector()

            emb1 = engine.encode([text1])
            emb2 = engine.encode([text2])

            result = detector.detect(emb1, emb2)
            adwin = ADWIN().detect(emb1, emb2)
            ddm = DDM().detect(emb1, emb2)
            hdp = HDP().detect(emb1, emb2)

            st.subheader("Detection Results")

            data = {
                "Method": ["LegalDrift (Ours)", "ADWIN", "DDM", "HDP"],
                "Drift Detected": [result.drift_detected, adwin.drift_detected, ddm.drift_detected, hdp.drift_detected],
                "P-Value": [result.p_value, adwin.p_value, ddm.p_value, hdp.p_value],
                "Severity": [result.severity, adwin.severity, ddm.severity, hdp.severity],
            }
            df = pd.DataFrame(data)
            st.dataframe(df)