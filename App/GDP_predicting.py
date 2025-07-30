import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px

# Set page config
st.set_page_config(page_title="GDP Prediction App", layout="centered")
st.header("🌍 Country GDP Prediction for 2025")
st.markdown("---")

# Load model and data
with open("App/predicting_model.pkl", "rb") as f:
    data = pickle.load(f)

models = data['models']
metrics = data['metrics']
df_gdp_2025 = data['predictions']
df_pd = pd.read_csv("App/cleaned_GDP_data.csv")

features = ['Exports', 'Imports', 'GDP_Per_Capita', 'GNI', 'Life_Expectancy', 'Population']
all_countries = sorted(df_pd['Country Name'].unique())

# --- User input via multiselect ---
selected_countries = st.multiselect(
    "Select countries to predict GDP for:",
    all_countries,
    default=["United States", "China", "Germany"]
)

# Run prediction if at least one country is selected
if selected_countries:
    # Historical data
    df_hist = df_pd[df_pd['Country Name'].isin(selected_countries)].copy()
    df_hist = df_hist[['Country Name', 'Year', 'GDP']]
    df_hist.dropna(subset=['GDP'], inplace=True)
    df_hist['Year'] = df_hist['Year'].astype(int)

    # Prediction
    predicted_rows = []
    st.subheader("📊 Prediction Results:")
    for country in selected_countries:
        if country not in models:
            st.warning(f"No model available for {country}")
            continue

        df_country = df_pd[df_pd['Country Name'] == country].dropna(subset=features)
        latest_year = df_country['Year'].max()
        latest_row = df_country[df_country['Year'] == latest_year]

        if latest_row.empty:
            st.warning(f"No recent data to predict for {country}")
            continue

        X_latest = latest_row[features].values
        model = models[country]
        pred_gdp = model.predict(X_latest)[0]

        rmse = metrics[country]['RMSE']
        r2 = metrics[country]['R2']

        st.write(f"**{country}** → 2025 GDP: **{pred_gdp:,.0f} USD** | RMSE: `{rmse:,.2f}` | R²: `{r2:.4f}`")

        predicted_rows.append({
            'Country Name': country,
            'Year': 2025,
            'GDP': pred_gdp
        })

    # Visualization
    if predicted_rows:
        df_pred = pd.DataFrame(predicted_rows)
        df_plot = pd.concat([df_hist, df_pred], ignore_index=True)
        df_plot.sort_values(by=['Country Name', 'Year'], inplace=True)

        fig = px.line(df_plot, x='Year', y='GDP', color='Country Name', markers=True,
                      title='GDP Trend with 2025 Prediction')

        fig.add_vline(x=2024.5, line_dash="dash", line_color="red", annotation_text="2025 Prediction")

        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="GDP (USD)",
            legend_title="Country",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)
