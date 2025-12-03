import os, streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Gagal Jantung",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    h1 {
        color: #2d3748;
        text-align: center;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem;
    }
    h2, h3 {
        color: #4a5568;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    .success-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .info-card {
        background: white;
        padding: 1rem;
        border-left: 4px solid #667eea;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Buat Debugging
def log_transform(X):
    return np.log1p(X)

# Muat model
try:
    model = joblib.load('random_forest_model.pkl')
    success_placeholder = st.empty()
    success_placeholder.success("Model berhasil dimuat!")
    import time
    time.sleep(2)
    success_placeholder.empty()
except FileNotFoundError:
    st.error("File model 'random_forest_model.pkl' tidak ditemukan. Pastikan sudah disimpan dari training.")
    st.stop()
except Exception as e:
    st.error(f"Error saat memuat model: {str(e)}")
    st.stop()

# Header
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h1>Prediksi Kelangsungan Hidup Pasien Gagal Jantung</h1>
    <p style='font-size: 1.2rem; color: #718096;'>
        Sistem prediksi untuk analisis risiko kematian
    </p>
</div>
""", unsafe_allow_html=True)

# Navigasi
tab1, tab2, tab3 = st.tabs(["Data Dictionary", "Perbandingan Model", "Prediksi"])

with tab1:
    st.markdown("### Data Dictionary & Rentang Normal")
    
    data_dict = {
        'Variabel': ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 
                     'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 
                     'sex', 'smoking', 'time', 'DEATH_EVENT'],
        'Tipe Data': ['float64', 'int64', 'int64', 'int64', 'int64', 'int64', 'float64', 
                      'float64', 'int64', 'int64', 'int64', 'int64', 'int64'],
        'Deskripsi': [
            'Usia pasien dalam tahun',
            'Anemia (0=Tidak, 1=Ya)',
            'Level enzim CPK dalam darah (mcg/L)',
            'Diabetes (0=Tidak, 1=Ya)',
            'Persentase darah yang dipompa keluar dari jantung setiap kontraksi (%)',
            'Tekanan darah tinggi (0=Tidak, 1=Ya)',
            'Jumlah trombosit dalam darah (kiloplatelets/mL)',
            'Level kreatinin dalam serum darah (mg/dL)',
            'Level natrium dalam serum darah (mEq/L)',
            'Jenis kelamin (1=Pria, 0=Wanita)',
            'Merokok (0=Tidak, 1=Ya)',
            'Lama waktu observasi atau tindak lanjut (hari)',
            'Status kematian pasien (Target: 0=Bertahan, 1=Meninggal)'
        ],
        'Rentang Normal/Keterangan': [
            'N/A',
            'N/A',
            '< 200 mcg/L',
            'N/A',
            '50%-70%',
            'N/A',
            '150.000 - 450.000',
            '0.6 - 1.3 mg/dL',
            '135 - 145 mEq/L',
            'N/A',
            'N/A',
            'Tidak digunakan untuk prediksi input',
            'Target prediksi'
        ]
    }
    df_dict = pd.DataFrame(data_dict)
    st.dataframe(df_dict, use_container_width=True)

with tab2:
    st.markdown("### Perbandingan Kinerja Model Klasifikasi")
    
    data_comp = {
        'Model': ['Random Forest Classifier', 'Logistic Regression', 'Decision Tree'], 
        'Accuracy': [0.75, 0.6833, 0.7], 
        'Precision': [0.625, 0.5, 0.5217], 
        'Recall': [0.5263, 0.5789, 0.6316], 
        'F1-Score': [0.5714, 0.5366, 0.5714], 
        'ROC-AUC': [0.8023, 0.7407, 0.7279], 
        'PR-AUC': [0.5881, 0.5849, 0.503], 
        'MAE': [0.3442, 0.4261, 0.2881], 
        'RMSE': [0.4117, 0.4564, 0.4872], 
        'R²': [0.2167, 0.0373, -0.0971]
    }
    
    df_comp = pd.DataFrame(data_comp).set_index('Model').T.round(4)
    df_comp.index.name = "Metric"
    
    st.dataframe(df_comp, use_container_width=True)
    
    # Visualisasi perbandingan
    st.markdown("#### Visualisasi Perbandingan Metrik")
    
    col_comp1, col_comp2 = st.columns(2)
    
    with col_comp1:
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(data_comp['Model']))
        width = 0.2
        
        for i, metric in enumerate(metrics_to_plot):
            values = [data_comp[metric][j] for j in range(len(data_comp['Model']))]
            ax.bar(x + i*width, values, width, label=metric)
        
        ax.set_xlabel('Model', fontsize=11, weight='bold')
        ax.set_ylabel('Score', fontsize=11, weight='bold')
        ax.set_title('Perbandingan Metrik Klasifikasi', fontsize=13, weight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(data_comp['Model'], rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col_comp2:
        fig, ax = plt.subplots(figsize=(10, 6))
        models = data_comp['Model']
        roc_auc = data_comp['ROC-AUC']
        colors_bar = ['#667eea', '#764ba2', '#f093fb']
        
        bars = ax.barh(models, roc_auc, color=colors_bar)
        ax.set_xlabel('ROC-AUC Score', fontsize=11, weight='bold')
        ax.set_title('Perbandingan ROC-AUC', fontsize=13, weight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{width:.4f}', 
                   ha='left', va='center', fontsize=10, weight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)

with tab3:
    st.markdown("### Input Data Pasien")
    
    with st.form(key='input_form'):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Demographic Information")
            age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=50.0, help="Patient age")
            sex_label = st.selectbox("Sex", ['Male', 'Female'], index=0)
            smoking_label = st.selectbox("Smoking", ['No', 'Yes'], index=0)
            
            st.markdown("#### Medical Conditions")
            anaemia_label = st.selectbox("Anaemia", ['No', 'Yes'], index=0)
            diabetes_label = st.selectbox("Diabetes", ['No', 'Yes'], index=0)
            high_blood_pressure_label = st.selectbox("High Blood Pressure", ['No', 'Yes'], index=0)
        
        with col2:
            st.markdown("#### Laboratory Results")
            creatinine_phosphokinase = st.number_input(
                "CPK (mcg/L)", 
                min_value=0, 
                value=582, 
                help="CPK enzyme level in blood. Normal < 200 mcg/L"
            )
            ejection_fraction = st.number_input(
                "Ejection Fraction (%)", 
                min_value=0, 
                max_value=100, 
                value=40, 
                help="Percentage of blood pumped out. Normal 50% to 70%"
            )
            platelets = st.number_input(
                "Platelets (kiloplatelets/mL)", 
                min_value=0.0, 
                value=265000.0, 
                help="Platelet count. Normal 150000 to 450000"
            )
            serum_creatinine = st.number_input(
                "Serum Creatinine (mg/dL)", 
                min_value=0.0, 
                value=1.0, 
                help="Creatinine level. Normal 0.6 to 1.3 mg/dL"
            )
            serum_sodium = st.number_input(
                "Serum Sodium (mEq/L)", 
                min_value=0, 
                value=135, 
                help="Sodium level. Normal 135 to 145 mEq/L"
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            submit_button = st.form_submit_button(label='Prediksi', use_container_width=True)

    # Konversi label menjadi numerik
    anaemia = 1 if anaemia_label == 'Yes' else 0
    diabetes = 1 if diabetes_label == 'Yes' else 0
    high_blood_pressure = 1 if high_blood_pressure_label == 'Yes' else 0
    smoking = 1 if smoking_label == 'Yes' else 0
    sex = 1 if sex_label == 'Male' else 0

    # Proses prediksi
    if submit_button:
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
        
        try:
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            
            st.markdown("---")
            st.markdown("### Hasil Prediksi")
            
            col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
            
            with col_res2:
                if prediction == 0:
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                                padding: 2rem; border-radius: 15px; text-align: center;
                                box-shadow: 0 10px 20px rgba(0,0,0,0.1);'>
                        <h2 style='color: white; margin: 0;'>BERTAHAN</h2>
                        <p style='color: white; font-size: 1.1rem; margin-top: 0.5rem;'>
                            Pasien diprediksi akan bertahan hidup
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
                                padding: 2rem; border-radius: 15px; text-align: center;
                                box-shadow: 0 10px 20px rgba(0,0,0,0.1);'>
                        <h2 style='color: white; margin: 0;'>RISIKO TINGGI</h2>
                        <p style='color: white; font-size: 1.1rem; margin-top: 0.5rem;'>
                            Pasien memiliki risiko kematian yang tinggi
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Metrics
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("Probabilitas Bertahan", f"{(1-probability)*100:.1f}%", 
                             delta=None, delta_color="normal")
                with col_m2:
                    st.metric("Probabilitas Meninggal", f"{probability*100:.1f}%", 
                             delta=None, delta_color="inverse")
            
            # Visualisasi
            st.markdown("---")
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                st.markdown("#### Distribusi Probabilitas")
                probabilities = [1 - probability, probability]
                labels = ['Bertahan', 'Meninggal']
                colors = ['#38ef7d', '#ff6a00']
                
                fig, ax = plt.subplots(figsize=(6, 4.5))
                wedges, texts, autotexts = ax.pie(
                    probabilities, 
                    labels=labels, 
                    colors=colors, 
                    autopct='%1.1f%%', 
                    startangle=90,
                    textprops={'fontsize': 12, 'weight': 'bold'},
                    explode=(0.05, 0.05)
                )
                for autotext in autotexts:
                    autotext.set_color('white')
                ax.set_title("Distribusi Probabilitas", fontsize=14, weight='bold', pad=20)
                st.pyplot(fig)
            
            with col_viz2:
                st.markdown("#### Tingkat Risiko")
                risk_level = "Rendah" if probability < 0.3 else "Sedang" if probability < 0.7 else "Tinggi"
                risk_color = "#38ef7d" if probability < 0.3 else "#ffa500" if probability < 0.7 else "#ff6a00"
                
                fig, ax = plt.subplots(figsize=(6, 4.5))
                categories = ['Risiko\nKematian']
                values = [probability * 100]
                
                bars = ax.barh(categories, values, color=risk_color, height=0.3)
                ax.set_xlim(0, 100)
                ax.set_xlabel('Persentase (%)', fontsize=12, weight='bold')
                ax.set_title('Tingkat Risiko Kematian', fontsize=14, weight='bold', pad=20)
                ax.grid(axis='x', alpha=0.3)
                
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 2, bar.get_y() + bar.get_height()/2, 
                           f'{width:.1f}%', 
                           ha='left', va='center', fontsize=12, weight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error saat prediksi: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #718096; padding: 1rem;'>
    <p>💡 Sistem ini menggunakan Random Forest Classifier dengan akurasi 75%</p>
</div>
""", unsafe_allow_html=True)