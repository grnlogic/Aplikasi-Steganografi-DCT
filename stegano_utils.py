import numpy as np
import cv2
from scipy.fftpack import dct, idct
from skimage.metrics import structural_similarity as ssim
import os

def get_image_type(file_path):
    """Get image type from file extension."""
    if not file_path:
        return "unknown"
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.jpg', '.jpeg', '.jpe', '.jfif']:
        return "jpeg"
    elif ext in ['.png']:
        return "png"
    elif ext in ['.bmp']:
        return "bmp"
    elif ext in ['.tiff', '.tif']:
        return "tiff"
    elif ext in ['.webp']:
        return "webp"
    elif ext in ['.gif']:
        return "gif"
    else:
        return "unknown"

def is_jpeg(file_path):
    """Check if file is JPEG by extension."""
    return get_image_type(file_path) == "jpeg"

def is_lossy_format(file_path):
    """Check if image format uses lossy compression."""
    img_type = get_image_type(file_path)
    return img_type in ["jpeg", "webp"]

def embed_message_dct(image, message, file_path=None):
    """Embed a message in an image using DCT."""
    # Konversi pesan ke binary
    binary_message = ''.join(format(ord(char), '08b') for char in message)
    binary_message += '00000000'  # Add null terminator
    
    # Check if the image is large enough for the message
    height, width = image.shape[:2]
    max_bytes = (height * width) // 64  # Each 8x8 block can store 1 bit
    
    if len(binary_message) > max_bytes:
        # For backward compatibility, return only two values if called with old signature
        if 'file_path' not in locals() or file_path is None:
            return image, False
        return image, False, "Pesan terlalu panjang untuk gambar ini"
    
    # Determine if we're working with a lossy format
    is_lossy = file_path and is_lossy_format(file_path)
    
    # Make a copy of the image
    stego_img = image.copy()
    
    # Convert to YCrCb color space (if RGB)
    if len(image.shape) == 3:
        ycrcb_img = cv2.cvtColor(stego_img, cv2.COLOR_RGB2YCrCb)
        y_channel = ycrcb_img[:, :, 0].astype(float)
    else:
        y_channel = stego_img.astype(float)
    
    # Embed the message
    msg_index = 0
    block_size = 8
    
    # Use different DCT coefficients based on image type
    # For lossy formats (JPEG, WebP), use more robust coefficients
    dct_row, dct_col = (3, 4) if is_lossy else (4, 5)
    
    # Embedding strength factor - higher for lossy formats
    strength = 3 if is_lossy else 1
    
    for i in range(0, height - block_size + 1, block_size):
        for j in range(0, width - block_size + 1, block_size):
            if msg_index < len(binary_message):
                # Extract 8x8 block
                block = y_channel[i:i+block_size, j:j+block_size]
                
                # Apply DCT
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                
                # Modify coefficient based on image type
                if binary_message[msg_index] == '1':
                    # Make coefficient odd
                    if int(abs(dct_block[dct_row, dct_col])) % 2 == 0:
                        dct_block[dct_row, dct_col] += strength if dct_block[dct_row, dct_col] >= 0 else -strength
                else:
                    # Make coefficient even
                    if int(abs(dct_block[dct_row, dct_col])) % 2 == 1:
                        dct_block[dct_row, dct_col] += strength if dct_block[dct_row, dct_col] >= 0 else -strength
                
                # Apply inverse DCT
                block = idct(idct(dct_block, norm='ortho').T, norm='ortho').T
                
                # Update the image
                y_channel[i:i+block_size, j:j+block_size] = block
                
                msg_index += 1
    
    # Convert back to RGB if needed
    if len(image.shape) == 3:
        # Ensure values are within 0-255 range
        y_channel = np.clip(y_channel, 0, 255)
        ycrcb_img[:, :, 0] = y_channel
        stego_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2RGB)
    else:
        # Ensure values are within 0-255 range
        stego_img = np.clip(y_channel, 0, 255).astype(np.uint8)
    
    # Success message based on image type
    success_message = "Berhasil menyisipkan pesan"
    if is_lossy:
        success_message += " (Format gambar lossy: Gunakan kualitas tinggi saat menyimpan)"
    
    # For backward compatibility, return only two values if called with old signature
    if 'file_path' not in locals() or file_path is None:
        return stego_img, True
    
    return stego_img, True, success_message

def extract_message_dct(stego_img, file_path=None):
    """Extract a message from a stego image using DCT."""
    # Determine if this is a lossy format image
    is_lossy = file_path and is_lossy_format(file_path)
    
    # Convert to YCrCb color space (if RGB)
    if len(stego_img.shape) == 3:
        ycrcb_img = cv2.cvtColor(stego_img, cv2.COLOR_RGB2YCrCb)
        y_channel = ycrcb_img[:, :, 0].astype(float)
    else:
        y_channel = stego_img.astype(float)
    
    height, width = y_channel.shape
    block_size = 8
    
    # Extract bits
    extracted_bits = []
    
    # Use different DCT coefficients based on image type
    dct_row, dct_col = (3, 4) if is_lossy else (4, 5)
    
    for i in range(0, height - block_size + 1, block_size):
        for j in range(0, width - block_size + 1, block_size):
            # Extract 8x8 block
            block = y_channel[i:i+block_size, j:j+block_size]
            
            # Apply DCT
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            
            # Check the selected coefficient
            if int(abs(dct_block[dct_row, dct_col])) % 2 == 1:
                extracted_bits.append('1')
            else:
                extracted_bits.append('0')
    
    # Convert bits to characters
    extracted_message = ""
    for i in range(0, len(extracted_bits), 8):
        if i + 8 <= len(extracted_bits):
            byte = ''.join(extracted_bits[i:i+8])
            if byte == '00000000':  # Null terminator
                break
            try:
                char_code = int(byte, 2)
                # Check if character is valid (printable ASCII)
                if 32 <= char_code <= 126:
                    extracted_message += chr(char_code)
                else:
                    # Invalid character, likely the end of the message
                    break
            except:
                # Decoding error, stop extraction
                break
    
    return extracted_message

def calculate_psnr(original, modified):
    """Calculate Peak Signal-to-Noise Ratio between two images."""
    mse = np.mean((original - modified) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_mse(original, modified):
    """Calculate Mean Squared Error between two images."""
    return np.mean((original - modified) ** 2)

def calculate_ssim(original, modified):
    """Calculate Structural Similarity Index between two images."""
    if len(original.shape) == 3:
        # Convert to grayscale for SSIM
        original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        modified_gray = cv2.cvtColor(modified, cv2.COLOR_RGB2GRAY)
        return ssim(original_gray, modified_gray)
    else:
        return ssim(original, modified)
