import os, streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.write("CWD:", os.getcwd())
st.write("Files:", os.listdir('.'))
try:
    import joblib
except Exception as e:
    st.error("Import joblib gagal: " + str(e))
    st.stop()

# Definisi Ulang Fungsi
def log_transform(X):
    return np.log1p(X)

# Muat model
try:
    model = joblib.load('random_forest_model.pkl')
    st.success("Model berhasil dimuat!")
except FileNotFoundError:
    st.error("File model 'random_forest_model.pkl' tidak ditemukan. Pastikan sudah disimpan dari training.")
    st.stop()
except Exception as e:
    st.error(f"Error saat memuat model: {str(e)}")
    st.stop()

st.title("Prediksi Kelangsungan Hidup Pasien Gagal Jantung")
st.write("Masukkan data pasien untuk memprediksi risiko kematian (DEATH_EVENT).")

# Form input fitur
with st.form(key='input_form'):
    age = st.number_input("Usia (age)", min_value=0.0, max_value=120.0, value=50.0)
    anaemia = st.selectbox("Anemia (anaemia)", [0, 1], index=0)
    creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (CPK)", min_value=0, value=582)
    diabetes = st.selectbox("Diabetes", [0, 1], index=0)
    ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=0, max_value=100, value=40)
    high_blood_pressure = st.selectbox("Tekanan Darah Tinggi", [0, 1], index=0)
    platelets = st.number_input("Platelets", min_value=0.0, value=265000.0)
    serum_creatinine = st.number_input("Serum Creatinine", min_value=0.0, value=1.0)
    serum_sodium = st.number_input("Serum Sodium", min_value=0, value=135)
    sex = st.selectbox("Jenis Kelamin (sex: 1=Pria, 0=Wanita)", [1, 0], index=0)
    smoking = st.selectbox("Merokok (smoking)", [0, 1], index=0)
    
    submit_button = st.form_submit_button(label='Prediksi')

# Proses prediksi jika submit
if submit_button:
    # Buat DataFrame dari input
    input_data = {
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
    }
    input_df = pd.DataFrame(input_data)
    
    # Prediksi
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]  # Probabilitas kelas 1 (Meninggal)
        
        # Tampilkan hasil
        st.subheader("Hasil Prediksi:")
        if prediction == 0:
            st.success(f"Prediksi: Bertahan (0)")
        else:
            st.error(f"Prediksi: Meninggal (1)")
        st.write(f"Probabilitas Kematian: {probability:.4f}")
        
        # Visualisasi sederhana probabilitas
        st.subheader("Visualisasi Probabilitas")
        fig, ax = plt.subplots()
        ax.bar(['Bertahan', 'Meninggal'], [1 - probability, probability], color=['green', 'red'])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probabilitas')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error saat prediksi: {str(e)}")