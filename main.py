import time
import math
import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

st.title("Prediksi Penyakit Kucing")
"""
Aplikasi ini dibuat untuk memprediksi penyakit kucing berdasarkan gejala yang dimasukkan. 
aplikasi ini dibuat menggunakan metode klasifikasi dan menggunakan algoritma **K-Nearest Neighbors** (KNN).
"""

tab1, tab2 = st.tabs(["Prediksi Data", "DataFrame"])

with tab1:
    """
    ## Prediksi Data
    ketahui apa yang dialami kucing anda dengan memasukkan gejala yang dialami kucing anda.
    """
    # load data
    df = pd.read_csv("data.csv")
    # symptoms get all column except the last column
    symptoms = df.columns[:-1]
    # get list of diseases
    diseases = df["penyakit"].unique()

    # get user input
    user_input = []

    symptoms_half = math.ceil(len(symptoms)/2)
    col1, col2 = st.columns(2)
    for i in range(len(symptoms)):
        if i < symptoms_half:
            # create checkbox
            user_input.append(col1.checkbox(symptoms[i]))
        else:
            # create checkbox
            user_input.append(col2.checkbox(symptoms[i]))

    # create button
    btn = st.button("Prediksi")

    if btn:
        with st.spinner("Memproses..."):
            time.sleep(5)
            
            # create dataframe from user input
            user_input_df = pd.DataFrame([user_input], columns=symptoms)
            # create model
            model = KNeighborsClassifier(n_neighbors=3)
            # df[symptoms] to number
            symptoms_converted = df[symptoms].replace({"Yes": 1, "No": 0})
            # fit model
            model.fit(symptoms_converted, df["penyakit"])
            """---"""
            # accuracy
            accuracy = model.score(symptoms_converted, df["penyakit"])
            precision = model.predict_proba(user_input_df[symptoms])
            st.text(
                f"akurasi model: {round(accuracy*100, 2)}% \n"
                f"presisi model: {round(precision.max()*100, 2)}%"
            )     
            """---"""
            # check user_input_df if there is no input
            if user_input_df.sum().sum() == 0:
                st.warning("Tidak ada gejala yang dipilih")
            
            else:
                # predict
                prediction = model.predict(user_input_df)

                # show prediction
                st.header("Hasil Prediksi")
                st.write("Berdasarkan gejala yang anda masukkan, kucing anda mungkin mengalami penyakit")
                st.subheader(prediction[0].capitalize())


with tab2:
    """
    # Data Frame
    """
    df = pd.read_csv("./data.csv")
    st.dataframe(df)

    # btn 
    btn = st.button("Lihat Sebaran Data")
    if btn:
        with st.spinner("Memproses..."):
            time.sleep(5)
            
            # chart penyakit
            st.subheader("Sebaran Penyakit")
            st.bar_chart(df["penyakit"].value_counts())

