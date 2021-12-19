
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Start of the AppWeb
st.title("Engine Prognosis Modelling streamlit")

# Training dataset
st.subheader("100 training engine units dataset")
df_stream = pd.read_csv('/home/dsc/git/TFM/TFM/df_train_1.csv', header=0, index_col=0)
st.write(df_stream)

# Max cycles per engine
st.subheader("Max cycles achieved by each engine unit")
df_max_cycles = df_stream.groupby("ID").max("Cycle").reset_index()["Cycle"]
st.bar_chart(df_max_cycles)

#ID engine
st.subheader("Operational settings and sensor variation per engine")
st.text("ID engine must be entered")

engine_option = df_stream["ID"].unique().tolist()
ids = st.selectbox("Which engine unit to display?", engine_option,0)
df_stream_id=df_stream[df_stream["ID"]==ids]

#Column filter
filtered = st.multiselect("Filter columns", options=list(df_stream.columns), default=list(df_stream.columns))
df_stream_id_fil = df_stream_id[filtered]

fig1 = px.line(df_stream_id_fil)
st.write(fig1)

# Sensor subplots

time = df_stream_id['Cycle']
df_stream_id.drop(labels=['Cycle'],axis=1,inplace=True)
cols = df_stream_id.columns[1:]
    
# Machine learning!

st.subheader("Set the training features of our model")
st.text("Define the parameters")

sel_col, disp_col = st.columns(2)
max_depth = sel_col.slider("Depth of our model?", min_value=10, max_value=200, value=20, step=10)
n_estimators = sel_col.selectbox("How many trees should there be?", options=[100,200,300,"No limit"], index=0)
#input_feature = sel_col.text_input("Which feature should be used as input feature?", "sensor_measurement_1")
#input_feature = sel_col.selectbox("Which feature should be used as input feature?", options=df_stream.columns[2:-1], index=0)

regr = RandomForestRegressor(max_depth=max_depth, n_estimators = n_estimators)
X_train = df_stream.drop(columns=["ID", "Cycle","RUL"])
y_train = df_stream["RUL"]

regr.fit(X_train,y_train)
preds = regr.predict(X_train)

disp_col.subheader("Mean absolute Error:")
disp_col.write(mean_absolute_error(y_train, preds))
disp_col.subheader("Mean Squared Error:")
disp_col.write(mean_squared_error(y_train, preds))
disp_col.subheader("R^2 Score:")
disp_col.write(r2_score(y_train, preds))
