import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px

# page layout
st.set_page_config(page_title="GDP & Growth Prediction", layout="centered")
st.title("🌍 Country GDP and GDP Growth Prediction for 2025")
st.markdown("---")

#Load Model 1 (GDP Prediction)
with open('App/predicting_model.pkl', "rb") as f1:
    data1 = pickle.load(f1)
models1 = data1["models"]
metrics1 = data1["metrics"]
df_gdp_2025_1 = data1["predictions"]

#Load Model 2 (GDP Growth Prediction) 
with open("App/gdp_growth_predicting_model.pkl", "rb") as f2:
    data2 = pickle.load(f2)
models2 = data2["models"]
metrics2 = data2["metrics"]
df_gdp_2025_2 = data2["predictions"]

#Load cleaned data
df_pd = pd.read_csv("App/cleaned_GDP_data.csv")

# Define features for each model
features1 = ['Exports', 'Imports', 'GDP_Per_Capita', 'GNI', 'Life_Expectancy', 'Population']
features2 = ['Exports', 'Imports', 'GDP_Per_Capita', 'GNI', 'Life_Expectancy', 'Population']

#Country selection 
all_countries = sorted(df_pd['Country Name'].unique())
selected_countries = st.multiselect("Select countries:", all_countries, default=["United States", "China", "Germany"])

#Predict GDP and Growth 
if selected_countries:
    df_hist_gdp = df_pd[df_pd["Country Name"].isin(selected_countries)][["Country Name", "Year", "GDP"]].dropna()
    df_hist_growth = df_pd[df_pd["Country Name"].isin(selected_countries)][["Country Name", "Year", "GDP_Growth_Rate"]].dropna()

    df_hist_gdp["Year"] = df_hist_gdp["Year"].astype(int)
    df_hist_growth["Year"] = df_hist_growth["Year"].astype(int)

    pred_gdp_rows = []
    pred_growth_rows = []

     #--- GDP Prediction ---
    st.subheader("📈 GDP Predictions")

    for country in selected_countries:
        df_country = df_pd[df_pd["Country Name"] == country]
        
        # Get most recent row
        latest_year = df_country["Year"].max()
        latest_data = df_country[df_country["Year"] == latest_year]

        if latest_data.empty:
            st.warning(f"No recent data for {country}")
            continue

        if country in models1:
            try:
                model1 = models1[country]
                X1 = latest_data[features1].values
                pred_gdp = model1.predict(X1)[0]
                rmse1 = metrics1[country]["RMSE"]
                r2_1 = metrics1[country]["R2"]

                pred_gdp_rows.append({"Country Name": country, "Year": 2025, "GDP": pred_gdp})
                st.write(f"**{country}** – 2025 GDP: **{pred_gdp:,.0f} USD** | RMSE: `{rmse1:,.2f}` | R²: `{r2_1:.4f}`")
            except Exception as e:
                st.error(f"Error in GDP prediction for {country}: {e}")
        else:
            st.warning(f"No GDP model for {country}")
     # --- GDP Line Chart ---
    if pred_gdp_rows:
        df_pred_gdp = pd.DataFrame(pred_gdp_rows)
        df_plot_gdp = pd.concat([df_hist_gdp, df_pred_gdp], ignore_index=True)
        df_plot_gdp.sort_values(by=["Country Name", "Year"], inplace=True)

        st.markdown("#### 📊 GDP Trend with 2025 Prediction")
        fig_gdp = px.line(df_plot_gdp, x="Year", y="GDP", color="Country Name", markers=True)
        fig_gdp.add_vline(x=2024.5, line_dash="dash", line_color="red", annotation_text="2025 Prediction")
        st.plotly_chart(fig_gdp, use_container_width=True)

    # --- GDP Growth Rate(%) Prediction ---
    st.subheader("📈 GDP Growth Rate(%) Predictions")
    for country in selected_countries:
        df_country = df_pd[df_pd["Country Name"] == country]
        
        # Get most recent row
        latest_year = df_country["Year"].max()
        latest_data = df_country[df_country["Year"] == latest_year]

        if latest_data.empty:
            st.warning(f"No recent data for {country}")
            continue

        # --- GDP Growth Prediction ---
        if country in models2:
            try:
                model2 = models2[country]
                X2 = latest_data[features2].values
                pred_growth = model2.predict(X2)[0]
                rmse2 = metrics2[country]["RMSE"]
                r2_2 = metrics2[country]["R2"]

                pred_growth_rows.append({"Country Name": country, "Year": 2025, "GDP_Growth_Rate": pred_growth})
                st.write(f"**{country}** – 2025 GDP Growth Rate: **{pred_growth:.2f}%** | RMSE: `{rmse2:,.2f}` | R²: `{r2_2:.4f}`")
            except Exception as e:
                st.error(f"Error in GDP Growth Rate prediction for {country}: {e}")
        else:
            st.warning(f"No GDP Growth model for {country}")

    # --- GDP Growth Line Chart ---
    if pred_growth_rows:
        df_pred_growth = pd.DataFrame(pred_growth_rows)
        df_plot_growth = pd.concat([df_hist_growth, df_pred_growth], ignore_index=True)
        df_plot_growth.sort_values(by=["Country Name", "Year"], inplace=True)

        st.markdown("#### 📈 GDP Growth Trend with 2025 Prediction")
        fig_growth = px.line(df_plot_growth, x="Year", y="GDP_Growth_Rate", color="Country Name", markers=True)
        fig_growth.add_vline(x=2024.5, line_dash="dash", line_color="green", annotation_text="2025 Prediction")
        st.plotly_chart(fig_growth, use_container_width=True)
