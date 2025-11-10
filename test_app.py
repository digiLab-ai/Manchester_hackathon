import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import twinlab as tl
import pandas as pd
tl.set_api_key("tl_zATKGbFWx34DVRj9wIUryw")
import base64
import time
constant =  1E9

if "emulator_trained" not in st.session_state:
    st.session_state.emulator_trained = None


st.set_page_config(
    page_title="Wide Layout Example",
    layout="wide",          # 👈 makes the app use full width
    initial_sidebar_state="expanded"  # optional
)

st.title("Tritium Transport Emulator Dashboard")
col1,col2 = st.columns(2)
with col1:
    st.header("training")
    col11,col12 = st.columns(2)

    with col11:
        col111,col112 = st.columns(2)
    with col11:
        placeholder = st.empty()
        placeholder12 = st.empty()
    with col12:
        col121,col122 = st.columns(2)
        placeholder2 = st.empty()
        placeholder22 = st.empty()
with col2:
    col21,col22 = st.columns(2)
    with col21:
        st.header("inference")
        placeholder3 = st.empty()

    # placeholder4 = st.empty()
# Button to trigger plot
trained = False
if col1.button("Train Emulator"):
    if st.session_state.emulator_trained is None:
        file_ = open("3d.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()


        for i in range(10):
            placeholder.image("Figures_3d//output"+str(i)+".png",  use_container_width=True)
            placeholder2.image("Figures_validation//output"+str(i)+".png", use_container_width=True)
            # placeholder22.image("Figures_uncertainty//output"+str(i)+".png", use_container_width=True)
            placeholder12.image("Figures_mean//output"+str(i)+".png", use_container_width=True)
            time.sleep(1)
        st.text("emulator trained")
    st.session_state.emulator_trained = True
if st.session_state.emulator_trained:
            placeholder.image("Figures_3d//output9.png",  use_container_width=True)
            placeholder2.image("Figures_validation//output9.png", use_container_width=True)
            placeholder12.image("Figures_mean//output9.png", use_container_width=True)
            st.text("emulator already trained")
# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

with col111:
    st.text_input(
        "diffusivity range start",
        "1E19",
        key="diff1",
    )
    st.text_input(
        "solubility range start",
        "0.02",
        key="sol1",
    )
    st.text_input(
        "thickness range start",
        "1.9",
        key="thic1",
    )
with col112:
    st.text_input(
        "diffusivity range end",
        "1E20",
        key="diff2",
    )
    st.text_input(
        "solubility range end",
        "0.10",
        key="sol2",
    )
    st.text_input(
        "thickness range end",
        "2.0",
        key="thic2",
    )
with col121:
    st.text_input(
        "kernel",
        "RBF",
        key="ker",
    )
    st.text_input(
        "warp inputs",
        "True",
        key="length",
    )
    st.text_input(
        "dimensionality reduction",
        "True",
        key="dim",
    )
with col2: 
    st.text_input(
        "experimental permeation rate [pa s-1]",
        "8.0",
        key="exper",
    )
    param = st.text_input(
        "parameter of interest",
        "diffusivity",
        key="param",
    )
if col2.button("Run Inference"):
    time.sleep(2)
    # Create a Matplotlib figure

    placeholder3.image("Figures_pairplots//30000.png", caption="sampling strategy for training", use_container_width=True)
        # placeholder4.image("Figures_mean//output"+str(i)+".png", caption="sampling strategy for training", use_container_width=True)
    with col2: 
        if param == "diffusivity":
            st.text(
                "diffusivity 7E19 ± 2E19 ",
            )
        elif param == "solubility":
            st.text(
                "solubility: 0.05 ± 0.01",
            )
        elif param == "thickness ±0.4":
            st.text(
                "thickness: 1.4 ±0.4",
            )
    # Display plot in Streamlit
st.header("prediction")
st.text_input(
    "diffusivity",
    "7E19",
    key="pred1",
)
st.text_input(
    "solubility",
    "0.04",
    key="pred2",
)
st.text_input(
    "thickness",
    "1.95",
    key="pred3",
)

if st.button("predict emulator"):
        st.text(
                "permeation: 14 s-1",
        )
