import streamlit as st
import pandas as pd
from meridian import data_frame_input_data_builder, Meridian
from meridian.output import HTMLModelOutput

st.title("📈 Google Meridian MMM Demo")
st.markdown("Upload a cleaned CSV with media spend, conversions, and (optional) control variables.")

uploaded_file = st.file_uploader("Upload your marketing data CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:", df.head())

    with st.spinner("Setting up the model and running analysis..."):
        channels = [col.replace("_spend", "") for col in df.columns if col.endswith("_spend")]

        builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
            kpi_type='non_revenue',
            default_kpi_column='conversions',
            default_revenue_per_kpi_column='revenue_per_conversion',
        )

        builder = builder.with_kpi(df)

        if 'revenue_per_conversion' in df.columns:
            builder = builder.with_revenue_per_kpi(df)

        if 'population' in df.columns:
            builder = builder.with_population(df)

        control_cols = [col for col in df.columns if 'control' in col]
        if control_cols:
            builder = builder.with_controls(df, control_cols=control_cols)

        builder = builder.with_media(
            df,
            media_cols=[f"{c}_impression" for c in channels if f"{c}_impression" in df.columns],
            media_spend_cols=[f"{c}_spend" for c in channels],
            media_channels=channels,
        )

        input_data = builder.build()

        mmm = Meridian(input_data)
        mmm.sample_prior(300)
        model_fit = mmm.sample_posterior(n_chains=2, n_adapt=500, n_burnin=300, n_keep=500)

        output = HTMLModelOutput(input_data, mmm, model_fit)

        summary_path = "summary_output.html"
        output.write_html(summary_path)

        st.success("✅ Model run complete! Download your summary report below.")
        with open(summary_path, "rb") as file:
            st.download_button("📄 Download Summary Report", file, file_name="meridian_mmm_summary.html")
