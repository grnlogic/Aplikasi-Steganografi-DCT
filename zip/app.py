from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import cv2
from PIL import Image
import io
import base64
import os
from stegano_utils import embed_message_dct, extract_message_dct, calculate_psnr, calculate_mse, calculate_ssim

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['UPLOAD_FOLDER'] = 'temp'

# Pastikan folder temp ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    """Render halaman utama"""
    return render_template('index.html')

@app.route('/embed', methods=['POST'])
def embed():
    """API endpoint untuk menyembunyikan pesan dalam gambar"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    message = request.form.get('message', '')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Baca gambar
        img = Image.open(file.stream)
        img_array = np.array(img)
        
        # Konversi ke RGB jika RGBA
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Embed pesan
        stego_img, success = embed_message_dct(img_array, message)
        
        if not success:
            return jsonify({'error': 'Message too large for this image'}), 400
        
        # Hitung metrik
        psnr = calculate_psnr(img_array, stego_img)
        mse = calculate_mse(img_array, stego_img)
        ssim = calculate_ssim(img_array, stego_img)
        
        # Konversi gambar hasil ke base64 untuk ditampilkan
        stego_pil = Image.fromarray(stego_img)
        buffered = io.BytesIO()
        stego_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Simpan gambar sementara untuk download
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'stego_image.png')
        stego_pil.save(temp_path)
        
        return jsonify({
            'success': True,
            'image': img_str,
            'metrics': {
                'psnr': round(psnr, 2),
                'mse': round(mse, 6),
                'ssim': round(ssim, 4)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/extract', methods=['POST'])
def extract():
    """API endpoint untuk mengekstrak pesan dari gambar"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Baca gambar
        img = Image.open(file.stream)
        img_array = np.array(img)
        
        # Konversi ke RGB jika RGBA
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Ekstrak pesan
        message = extract_message_dct(img_array)
        
        if not message:
            return jsonify({'error': 'No message found in image'}), 400
        
        return jsonify({
            'success': True,
            'message': message
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download')
def download():
    """Endpoint untuk mengunduh gambar stego"""
    try:
        return send_file(
            os.path.join(app.config['UPLOAD_FOLDER'], 'stego_image.png'),
            as_attachment=True,
            download_name='stego_image.png',
            mimetype='image/png'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
