import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

def render_shap_explanation(rf_model, rf_features, input_df):
    """Renders SHAP local waterfall plot and feature contribution table for the current input."""
    st.markdown("---")
    st.subheader("Model Explainability (SHAP Analysis)")
    st.markdown("Understand **why** the model made this prediction and the exact contribution of each input feature.")

    if not SHAP_AVAILABLE:
        st.info("Install `shap` and `matplotlib` to view live waterfall plots (`pip install shap matplotlib`).")
        return

    if rf_model is None:
        st.info("Train the Random Forest model to unlock live SHAP waterfall diagrams.")
        return

    try:
        cols = rf_features or list(input_df.columns)
        aligned_df = input_df.reindex(columns=cols, fill_value=0.0)

        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer(aligned_df)

        tab1, tab2 = st.tabs(["Local Prediction Breakdown (Waterfall)", "Feature Contribution Table"])

        with tab1:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            shap.plots.waterfall(shap_values[0], show=False)
            plt.title("SHAP Local Explanation for Current Input", fontsize=12, pad=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with tab2:
            contrib_df = pd.DataFrame({
                "Feature": aligned_df.columns,
                "Feature Value": aligned_df.iloc[0].values,
                "SHAP Impact (Delta AQI)": shap_values[0].values
            }).sort_values("SHAP Impact (Delta AQI)", ascending=False).reset_index(drop=True)

            st.dataframe(
                contrib_df.style.format({
                    "Feature Value": "{:.2f}",
                    "SHAP Impact (Delta AQI)": "{:+.4f}"
                }).background_gradient(subset=["SHAP Impact (Delta AQI)"], cmap="coolwarm"),
                width="stretch"
            )

    except Exception as e:
        st.warning(f"Could not compute live SHAP explainability: {e}")
