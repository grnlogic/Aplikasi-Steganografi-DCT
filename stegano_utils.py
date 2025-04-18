import numpy as np
import cv2
from scipy.fftpack import dct, idct
from skimage.metrics import structural_similarity as ssim

def embed_message_dct(image, message):
    """Embed a message in an image using DCT."""
    # Konversi pesan ke binary
    binary_message = ''.join(format(ord(char), '08b') for char in message)
    binary_message += '00000000'  # Add null terminator
    
    # Check if the image is large enough for the message
    height, width = image.shape[:2]
    max_bytes = (height * width) // 64  # Each 8x8 block can store 1 bit
    
    if len(binary_message) > max_bytes:
        return image, False
    
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
    
    for i in range(0, height - block_size + 1, block_size):
        for j in range(0, width - block_size + 1, block_size):
            if msg_index < len(binary_message):
                # Extract 8x8 block
                block = y_channel[i:i+block_size, j:j+block_size]
                
                # Apply DCT
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                
                # Modify the mid-frequency coefficient (4,5)
                # This position is chosen as it's less perceptible to human eye
                if binary_message[msg_index] == '1':
                    # Make coefficient odd
                    if int(abs(dct_block[4, 5])) % 2 == 0:
                        dct_block[4, 5] += 1 if dct_block[4, 5] >= 0 else -1
                else:
                    # Make coefficient even
                    if int(abs(dct_block[4, 5])) % 2 == 1:
                        dct_block[4, 5] += 1 if dct_block[4, 5] >= 0 else -1
                
                # Apply inverse DCT
                block = idct(idct(dct_block, norm='ortho').T, norm='ortho').T
                
                # Update the image
                y_channel[i:i+block_size, j:j+block_size] = block
                
                msg_index += 1
    
    # Convert back to RGB if needed
    if len(image.shape) == 3:
        ycrcb_img[:, :, 0] = y_channel
        stego_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2RGB)
    else:
        stego_img = y_channel.astype(np.uint8)
    
    return stego_img, True

def extract_message_dct(stego_img):
    """Extract a message from a stego image using DCT."""
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
    
    for i in range(0, height - block_size + 1, block_size):
        for j in range(0, width - block_size + 1, block_size):
            # Extract 8x8 block
            block = y_channel[i:i+block_size, j:j+block_size]
            
            # Apply DCT
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            
            # Check the mid-frequency coefficient (4,5)
            if int(abs(dct_block[4, 5])) % 2 == 1:
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
            extracted_message += chr(int(byte, 2))
    
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
