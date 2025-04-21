import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import base64
from stegano_utils import embed_message_dct, extract_message_dct, calculate_psnr, calculate_mse, calculate_ssim

# Set page configuration
st.set_page_config(
    page_title="PixelHide",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
def local_css():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 1rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid #f0f2f6;
        }
        .sub-header {
            font-size: 1.8rem;
            color: #0277BD;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .card {
            border-radius: 5px;
            background-color: #f9f9f9;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .metric-card {
            background-color: #e3f2fd;
            border-radius: 5px;
            padding: 10px;
            margin: 5px 0;
            text-align: center;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #0277BD;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #555;
        }
        .stButton button {
            background-color: #1E88E5;
            color: white;
            font-weight: bold;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            transition: all 0.3s;
        }
        .stButton button:hover {
            background-color: #0277BD;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .download-btn {
            background-color: #43a047;
            color: white;
            font-weight: bold;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
        }
        .info-box {
            background-color: #e8f5e9;
            border-left: 5px solid #43a047;
            padding: 10px 15px;
            margin: 10px 0;
            border-radius: 0 5px 5px 0;
        }
        .warning-box {
            background-color: #fff8e1;
            border-left: 5px solid #ffb300;
            padding: 10px 15px;
            margin: 10px 0;
            border-radius: 0 5px 5px 0;
        }
        .tab-content {
            padding: 20px 0;
        }
        .footer {
            text-align: center;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #f0f2f6;
            color: #666;
            font-size: 0.8rem;
        }
        /* Popup styling */
        .popup-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(0, 0, 0, 0.5);
            z-index: 1000;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .popup-content {
            background-color: white;
            padding: 25px;
            border-radius: 10px;
            max-width: 600px;
            width: 90%;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            position: relative;
        }
        .popup-header {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
            border-bottom: 1px solid #eaeaea;
            padding-bottom: 10px;
        }
        .popup-title {
            font-size: 1.5rem;
            color: #000000;
            margin: 0;
            padding-left: 10px;
        }
        .popup-body {
            margin-bottom: 20px;
            font-size: 1.1rem;
            color: #000000;
        }
        .popup-footer {
            display: flex;
            justify-content: flex-end;
        }
        .popup-icon {
            font-size: 1.8rem;
            color: #000000;
        }
        .popup-button {
            padding: 8px 16px;
            background-color: #1E88E5;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
        }
        .popup-button:hover {
            background-color: #0277BD;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# App header with logo
def header():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown('<div class="main-header">🔐 PixelHide</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            Sembunyikan pesan rahasia Anda di dalam gambar menggunakan Discrete Cosine Transform
        </div>
        """, unsafe_allow_html=True)

header()

# Sidebar with information
with st.sidebar:
    st.image("https://raw.githubusercontent.com/streamlit/streamlit/master/examples/data/logo.png", width=200)
    st.markdown("## Tentang")
    st.markdown("""
    Aplikasi ini memungkinkan Anda untuk:
    - Menyembunyikan pesan rahasia di dalam gambar
    - Mengekstrak pesan tersembunyi dari gambar
    - Mengevaluasi kualitas steganografi
    """)
    
    st.markdown("## Cara Kerja")
    st.markdown("""
    1. **Transformasi DCT**: Mengubah blok gambar ke domain frekuensi
    2. **Penyisipan Pesan**: Memodifikasi koefisien frekuensi tertentu
    3. **Transformasi Balik**: Mengembalikan ke domain spasial
    """)
    
    st.markdown("## Tips")
    st.markdown("""
    - Gunakan gambar PNG untuk hasil yang lebih baik
    - Gambar yang lebih besar dapat menyembunyikan pesan yang lebih panjang
    - Hindari mengedit gambar stego setelah dibuat
    """)
    
    st.markdown("---")
    st.markdown("### Dibuat dengan Oleh Kelompok Keamanan Informasi")
    st.markdown("### Universitas Siliwangi")

# Tambahkan ikon pada tab
tab1, tab2, tab3 = st.tabs(["📝 Sisipkan Pesan", "🔍 Ekstrak Pesan", "📊 Evaluasi"])

with tab1:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Sisipkan Pesan Rahasia</div>', unsafe_allow_html=True)
    
    # Instructions card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    ### Petunjuk
    1. Rekomendasi Unggah gambar (PNG atau JPG)
    2. Masukkan pesan rahasia Anda
    3. Klik "Sisipkan Pesan" untuk menyembunyikan pesan di dalam gambar
    4. Unduh gambar stego yang dihasilkan
    """)
    st.markdown('<div class="warning-box">Kapasitas tergantung pada ukuran gambar. Gambar yang lebih besar dapat menyembunyikan pesan yang lebih panjang.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Upload and message input
    uploaded_file = st.file_uploader("Pilih gambar sampul...", type=["jpg", "jpeg", "png", "gif"])
    message = st.text_area("Masukkan pesan untuk disembunyikan:", height=100, placeholder="Ketik pesan rahasia Anda di sini...")
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Gambar Asli")
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar Sampul", use_column_width=True)
            
            # Image info
            img_width, img_height = image.size
            img_format = image.format
            img_size = uploaded_file.size / 1024  # KB
            
            st.markdown(f"""
            **Detail Gambar:**
            - Dimensi: {img_width} × {img_height} piksel
            - Format: {img_format}
            - Ukuran: {img_size:.1f} KB
            - Panjang pesan maksimal: ~{(img_width * img_height) // 64 // 8} karakter
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Center the button
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            embed_button = st.button("🔒 Sisipkan Pesan", use_container_width=True, help="Klik untuk menyembunyikan pesan")
        
        if embed_button and message:
            # Convert PIL Image to OpenCV format
            img_array = np.array(image)
            if len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            # Embed message
            with st.spinner("Proses menyembunyikan pesan..."):
                stego_img, success = embed_message_dct(img_array, message)
            
            if success:
                with col2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("### Gambar Stego")
                    stego_pil = Image.fromarray(stego_img)
                    st.image(stego_pil, caption="Gambar dengan pesan tersembunyi", use_column_width=True)
                    
                    # Save button
                    buf = io.BytesIO()
                    stego_pil.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="💾 Unduh Gambar Stego",
                        data=byte_im,
                        file_name="stego_image.png",
                        mime="image/png",
                        use_container_width=True
                    )
                    
                    # Calculate metrics
                    psnr = calculate_psnr(img_array, stego_img)
                    mse = calculate_mse(img_array, stego_img)
                    ssim = calculate_ssim(img_array, stego_img)
                    
                    st.markdown("### Metrik Kualitas Gambar")
                    
                    metric_cols = st.columns(3)
                    with metric_cols[0]:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown(f'<div class="metric-value">{psnr:.2f} dB</div>', unsafe_allow_html=True)
                        st.markdown('<div class="metric-label">PSNR</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with metric_cols[1]:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown(f'<div class="metric-value">{mse:.6f}</div>', unsafe_allow_html=True)
                        st.markdown('<div class="metric-label">MSE</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with metric_cols[2]:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown(f'<div class="metric-value">{ssim:.4f}</div>', unsafe_allow_html=True)
                        st.markdown('<div class="metric-label">SSIM</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="info-box">Nilai PSNR dan SSIM yang lebih tinggi menunjukkan kualitas yang lebih baik. Nilai MSE yang lebih rendah lebih baik.</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Pesan terlalu besar untuk gambar ini. Gunakan gambar yang lebih besar atau pesan yang lebih pendek.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ekstrak Pesan Tersembunyi</div>', unsafe_allow_html=True)
    
    # Instructions card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    ### Petunjuk
    1. Unggah gambar stego (gambar dengan pesan tersembunyi)
    2. Klik "Ekstrak Pesan" untuk mengungkapkan konten tersembunyi
    3. Pesan yang diekstrak akan muncul di bawah
    """)
    st.markdown('<div class="warning-box">Hanya gambar yang dibuat dengan aplikasi ini yang dapat didekodekan dengan benar.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Upload stego image
    stego_file = st.file_uploader("Unggah gambar stego...", type=["jpg", "jpeg", "png", "gif"], key="extract_uploader")
    
    if stego_file is not None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        stego_image = Image.open(stego_file)
        st.image(stego_image, caption="Gambar Stego", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Center the button
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            extract_button = st.button("🔍 Ekstrak Pesan", use_container_width=True, help="Klik untuk mengekstrak pesan")
        
        if extract_button:
            # Convert PIL Image to OpenCV format
            stego_array = np.array(stego_image)
            if len(stego_array.shape) == 3 and stego_array.shape[2] == 4:  # RGBA
                stego_array = cv2.cvtColor(stego_array, cv2.COLOR_RGBA2RGB)
            
            # Extract message
            with st.spinner("Proses mengekstrak pesan..."):
                extracted_message = extract_message_dct(stego_array)
            
            if extracted_message:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.success("Pesan berhasil diekstrak!")
                st.markdown("### Pesan yang Diekstrak:")
                st.markdown('<div style="background-color: #f0f0f0; padding: 15px; border-radius: 5px; font-family: monospace;">', unsafe_allow_html=True)
                st.markdown(f"{extracted_message}")
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Tidak dapat mengekstrak pesan dari gambar ini.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluasi Steganografi</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Apa itu Steganografi DCT?")
        
        st.markdown("""
        Steganografi DCT (Discrete Cosine Transform) bekerja dengan cara:
        
        1. **Membagi** gambar menjadi blok piksel 8×8
        2. **Mengubah** setiap blok menggunakan DCT untuk mendapatkan koefisien frekuensi
        3. **Memodifikasi** koefisien DCT tertentu untuk menyisipkan bit pesan
        4. **Membalikkan** DCT untuk mendapatkan gambar yang dimodifikasi
        
        Metode ini tahan terhadap beberapa operasi pemrosesan gambar dan memberikan keseimbangan yang baik antara kapasitas dan imperseptibilitas.
        """)
        
        st.markdown("### Metrik Evaluasi")
        
        st.markdown("""
        - **PSNR (Peak Signal-to-Noise Ratio)**: Mengukur kualitas gambar. Nilai lebih tinggi (>30dB) menunjukkan kualitas yang lebih baik.
        - **MSE (Mean Squared Error)**: Mengukur perbedaan antara gambar asli dan stego. Semakin rendah semakin baik.
        - **SSIM (Structural Similarity Index)**: Mengukur kesamaan yang dirasakan. Nilai mendekati 1 menunjukkan kesamaan yang lebih tinggi.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Tips untuk Hasil yang Lebih Baik")
        
        st.markdown("""
        - **Gunakan gambar yang lebih besar** untuk menyembunyikan pesan yang lebih panjang
        - **Format PNG** lebih baik dalam menjaga data tersembunyi dibandingkan JPEG
        - **Hindari kompresi** atau pengeditan gambar stego
        - **Gambar yang kompleks** (dengan tekstur) umumnya lebih baik untuk menyembunyikan pesan
        - **Hindari area yang halus** seperti langit atau warna solid
        """)
        
        st.markdown("### Teknik Lanjutan")
        
        st.markdown("""
        - **Perlindungan kata sandi**: Enkripsi pesan sebelum menyisipkan
        - **Penyisipan adaptif**: Memodifikasi koefisien berdasarkan konten gambar
        - **Kode koreksi kesalahan**: Menambahkan redundansi untuk memulihkan dari perubahan kecil
        - **Saluran warna ganda**: Menyebarkan pesan di seluruh saluran RGB
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add a visual explanation
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Penjelasan Visual DCT")
        
        # Create a simple visual explanation
        fig_width, fig_height = 400, 200
        fig_data = np.zeros((fig_height, fig_width, 3), dtype=np.uint8)
        
        # Draw original image section
        fig_data[50:150, 50:150] = [200, 200, 200]
        
        # Draw arrow
        for i in range(150, 250):
            fig_data[100, i] = [0, 120, 255]
        for i in range(95, 106):
            fig_data[i, 240] = [0, 120, 255]
        fig_data[95:101, 235:241] = [0, 120, 255]
        fig_data[100:106, 235:241] = [0, 120, 255]
        
        # Draw DCT coefficients
        for i in range(8):
            for j in range(8):
                intensity = 255 - min(255, int(30 * (i + j)))
                fig_data[50+i*12:50+(i+1)*12, 250+j*12:250+(j+1)*12] = [intensity, intensity, intensity]
        
        # Add text
        pil_img = Image.fromarray(fig_data)
        st.image(pil_img, caption="DCT mengubah blok gambar menjadi koefisien frekuensi", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Tambahkan footer dengan gaya baru
st.markdown("""
<div style="text-align: center; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #f0f2f6; color: #666; font-size: 0.8rem;">
    Aplikasi Steganografi DCT © 2025 | Dibuat dengan ❤️ menggunakan Streamlit
</div>
""", unsafe_allow_html=True)
