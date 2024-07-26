import pickle
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Membaca model
karir_model = pickle.load(open('karir_prediksi.sav', 'rb'))

# Judul web
st.title('Prediksi karir')

# Input data dengan validasi
st.sidebar.header('Input Data')
def user_input_features():
    GamesPlayer = st.sidebar.text_input('GamesPlayer', '0.0')
    MinutesPlayer = st.sidebar.text_input('MinutesPlayer', '0.0')
    PointPerGame = st.sidebar.text_input('PointPerGame', '0.0')
    FieldGoalsMade = st.sidebar.text_input('FieldGoalsMade', '0.0')
    data = {
        'GamesPlayer': float(GamesPlayer),
        'MinutesPlayer': float(MinutesPlayer),
        'PointPerGame': float(PointPerGame),
        'FieldGoalsMade': float(FieldGoalsMade)
    }
    return data

data = user_input_features()

# Menampilkan data input
st.subheader('Input Data')
st.write(data)

if st.button('Prediksi'):
    try:
        # Konversi input menjadi numerik
        inputs = np.array([[data['GamesPlayer'], data['MinutesPlayer'], data['PointPerGame'], data['FieldGoalsMade']]])
        
        # Lakukan prediksi
        karir_prediksi = karir_model.predict(inputs)
        probabilities = karir_model.predict_proba(inputs)  # Mendapatkan probabilitas
        
        # Menampilkan hasil prediksi
        st.subheader('Hasil Prediksi')
        if karir_prediksi[0] == 0:
            st.write('karier seorang pemain kurang dari 5 tahun')
        else:
            st.write('karier seorang pemain memiliki karier selama 5 tahun atau lebih')
        
        
        # Visualisasi probabilitas
        fig, ax = plt.subplots()
        classes = ['karier seorang pemain kurang dari 5 tahun', 'karier seorang pemain memiliki karier selama 5 tahun atau lebih']
        ax.bar(classes, probabilities[0])
        ax.set_ylabel('Probabilitas')
        ax.set_title('Probabilitas Prediksi')
        st.pyplot(fig)
    
    except ValueError:
        st.error("Pastikan semua input diisi dengan angka yang valid.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
