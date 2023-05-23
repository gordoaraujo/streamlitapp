# Streamlit live coding script
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from urllib.request import urlopen
import json
from copy import deepcopy
import statsmodels.formula.api as smf

# First some MPG Data Exploration
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

@st.cache_data
def fit_and_predict(df, lm_formula):
    # fit a model explaining hwy fuel mileage through displacement
    lm = smf.ols(formula = lm_formula, data=df).fit()
    
    # find two points on the line represented by the model
    x_bounds = df['displ']
    y_bounds = df['hwy']

    preds_input = pd.DataFrame({'displ': x_bounds})
    predictions = lm.predict(preds_input)

    return lm, pd.DataFrame({'displ': x_bounds, "hwy" : y_bounds, 'hwy_hat': predictions})

mpg_df_raw = load_data(path="data/mpg.csv")
mpg_df = deepcopy(mpg_df_raw)

# Add title and header
st.title("MPG Data Exploration")

# Widgets: checkbox (you can replace st.xx with st.sidebar.xx)
if st.checkbox("Show Dataframe"):
    st.subheader("This is my dataset:")
    st.dataframe(data=mpg_df)
    # st.table(data=mpg_df)

# Setting up columns
left_column, middle_column, right_column, ext_right_column = st.columns([2, 2, 2, 2])

# Widgets: selectbox
years = ["All"]+sorted(pd.unique(mpg_df['year']))
year = left_column.selectbox("Choose a Year", years)

# Widgets: radio buttons
show_means = middle_column.radio(
    label='Show Class Means', options=['Yes', 'No'])

plot_types = ["Matplotlib", "Plotly"]
plot_type = right_column.radio("Choose Plot Type", plot_types)

# Flow control and plotting
if year == "All":
    reduced_df = mpg_df
else:
    reduced_df = mpg_df[mpg_df["year"] == year]

means = reduced_df.groupby('class').mean(numeric_only=True)

# Setting second-row columns
sr_left_column, sr_right_column = st.columns([2, 6])

# Add linear regression
show_lm = sr_left_column.radio(label='Show Linear Model', options=['Yes', 'No'], index = 1)
fit_lm = sr_right_column.text_input('Linear Model Formula', 'hwy ~ displ + I(displ**2)')

if show_lm == "Yes":
    lm, pred = fit_and_predict(reduced_df, lm_formula = fit_lm)
    pred.sort_values(by = "displ", ascending = True, inplace = True)

# In Matplotlib
m_fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(reduced_df['displ'], reduced_df['hwy'], alpha=0.7)

if show_means == "Yes":
    ax.scatter(means['displ'], means['hwy'], alpha=0.7, color="red")

if show_lm == "Yes":
    ax.plot(pred["displ"], pred["hwy_hat"], marker = ".", color = "red")
    ax.set_title(f"R2 adj. = {lm.rsquared_adj:.6f}", loc = "left")

plt.suptitle("Engine Size vs. Highway Fuel Mileage")
ax.set_xlabel('Displacement (Liters)')
ax.set_ylabel('MPG')

# In Plotly
p_fig = px.scatter(reduced_df, x='displ', y='hwy', opacity=0.5,
                   range_x=[1, 8], range_y=[10, 50],
                   width=750, height=600,
                   labels={"displ": "Displacement (Liters)",
                           "hwy": "MPG"},
                   title="Engine Size vs. Highway Fuel Mileage")
p_fig.update_layout(title_font_size=22)

if show_means == "Yes":
    p_fig.add_trace(go.Scatter(x=means['displ'], y=means['hwy'],
                               mode="markers", marker_color = "red"))
    p_fig.update_layout(showlegend=False)

if show_lm == "Yes":
    p_fig.add_trace(go.Scatter(x = pred["displ"], y = pred["hwy_hat"], line_color = "red", mode = "lines"))
    p_fig.update_layout(title = f"Engine Size vs. Highway Fuel Mileage <br><sup> R2 adj. = {lm.rsquared_adj:.6f} </sup>")

# Select which plot to show
if plot_type == "Matplotlib":
    st.pyplot(m_fig)
else:
    st.plotly_chart(p_fig)

# We can write stuff
url = "https://archive.ics.uci.edu/ml/datasets/auto+mpg"
st.write("Data Source:", url)
# "This works too:", url