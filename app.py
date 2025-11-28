import streamlit as st
import pandas as pd
import joblib

# 1. Load Model
model = joblib.load('random_forest_model.pkl')

# 2. Judul & Deskripsi
st.title("Aplikasi Prediksi Gagal Jantung")
st.write("Masukkan data klinis pasien di bawah ini untuk memprediksi risiko kematian.")

# 3. Form Input Data
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Umur', min_value=1, max_value=100, value=50)
    creatinine_phosphokinase = st.number_input('Creatinine Phosphokinase (mcg/L)', value=582)
    ejection_fraction = st.number_input('Ejection Fraction (%)', min_value=0, max_value=100, value=38)
    platelets = st.number_input('Platelets (kiloplatelets/mL)', value=263358.0)
    serum_creatinine = st.number_input('Serum Creatinine (mg/dL)', value=1.1)

with col2:
    serum_sodium = st.number_input('Serum Sodium (mEq/L)', value=136)
    sex = st.selectbox('Jenis Kelamin', [0, 1], format_func=lambda x: 'Perempuan' if x==0 else 'Laki-laki')
    smoking = st.selectbox('Perokok?', [0, 1], format_func=lambda x: 'Tidak' if x==0 else 'Ya')
    anaemia = st.selectbox('Anaemia?', [0, 1], format_func=lambda x: 'Tidak' if x==0 else 'Ya')
    diabetes = st.selectbox('Diabetes?', [0, 1], format_func=lambda x: 'Tidak' if x==0 else 'Ya')
    high_blood_pressure = st.selectbox('Darah Tinggi?', [0, 1], format_func=lambda x: 'Tidak' if x==0 else 'Ya')

# 4. Tombol Prediksi
if st.button('Prediksi Risiko'):
    input_data = pd.DataFrame({
        'age': [age],
        'anaemia': [anaemia],
        'creatinine_phosphokinase': [creatinine_phosphokinase],
        'diabetes': [diabetes],
        'ejection_fraction': [ejection_fraction],
        'high_blood_pressure': [high_blood_pressure],
        'platelets': [platelets],
        'serum_creatinine': [serum_creatinine],
        'serum_sodium': [serum_sodium],
        'sex': [sex],
        'smoking': [smoking]
    })

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    # Tampilkan Hasil
    st.subheader('Hasil Prediksi:')
    if prediction[0] == 1:
        st.error(f'Pasien Berisiko TINGGI (Probabilitas Kematian: {probability:.1%})')
    else:
        st.success(f'Pasien Berisiko RENDAH (Probabilitas Kematian: {probability:.1%})')