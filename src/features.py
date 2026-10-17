import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.stats import skew, kurtosis

def extract_histogram_features(image):
    """
    Rút trích các đặc trưng thống kê bậc 1 từ phân bố mức xám:
    Mean, Std/Variance, Skewness, Kurtosis.
    """
    pixels = image.ravel().astype(np.float64)
    mean_val = np.mean(pixels)
    std_val = np.std(pixels)
    skew_val = float(skew(pixels))
    kurt_val = float(kurtosis(pixels))
    return [mean_val, std_val, skew_val, kurt_val]

def extract_glcm_features(image):
    """
    Rút trích 5 đặc trưng kết cấu GLCM trung bình qua 4 hướng (0, 45, 90, 135 độ).
    """
    # Lượng tử hóa về 16 mức xám để tăng tốc độ tính toán
    quantized_img = (image // 16).astype(np.uint8)
    
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    glcm = graycomatrix(quantized_img, distances=distances, angles=angles, levels=16, symmetric=True, normed=True)
    
    contrast = float(graycoprops(glcm, 'contrast').mean())
    dissimilarity = float(graycoprops(glcm, 'dissimilarity').mean())
    homogeneity = float(graycoprops(glcm, 'homogeneity').mean())
    energy = float(graycoprops(glcm, 'energy').mean())
    correlation = float(graycoprops(glcm, 'correlation').mean())
    
    return [contrast, dissimilarity, homogeneity, energy, correlation]

def extract_lbp_features(image, num_points=24, radius=3):
    """
    Rút trích đặc trưng kết cấu cục bộ LBP (Uniform Pattern).
    Số bins = num_points + 2 = 26 giá trị.
    """
    lbp = local_binary_pattern(image, P=num_points, R=radius, method="uniform")
    n_bins = num_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.tolist()

def extract_single_image_features(image):
    """
    Tổng hợp toàn bộ đặc trưng của 1 ảnh thành 1 vector 1D (35 giá trị).
    """
    hist_feats = extract_histogram_features(image)
    glcm_feats = extract_glcm_features(image)
    lbp_feats = extract_lbp_features(image)
    return np.array(hist_feats + glcm_feats + lbp_feats, dtype=np.float32)

def extract_features_dataset(images, verbose=True):
    """
    Rút trích đặc trưng cho toàn bộ tập ảnh.
    Trả về ma trận X có kích thước: (N_samples, N_features).
    """
    feature_list = []
    total = len(images)
    
    if verbose:
        print(f"[*] Bắt đầu trích xuất đặc trưng cho {total} ảnh...")
        
    for i, img in enumerate(images):
        feats = extract_single_image_features(img)
        feature_list.append(feats)
        if verbose and (i + 1) % 200 == 0:
            print(f" -> Đã trích xuất {i + 1}/{total} ảnh")
            
    X = np.array(feature_list, dtype=np.float32)
    if verbose:
        print(f"[+] Hoàn tất! Ma trận đặc trưng có kích thước: {X.shape}")
    return X

if __name__ == "__main__":
    from dataset import load_neu_dataset_recursive
    
    DATA_PATH = r"D:\Tài Liệu\DIP_Project\Surface_Defect_Project\Dataset\NEU-DET\train"
    imgs, lbls = load_neu_dataset_recursive(DATA_PATH)
    
    if len(imgs) > 0:
        # Chạy thử nghiệm trích xuất trên 10 ảnh đầu tiên để kiểm tra kích thước
        print("\n[*] Chạy kiểm tra mẫu trích xuất đặc trưng...")
        sample_X = extract_features_dataset(imgs[:10], verbose=False)
        print(f"[+] Kích thước vector 1 ảnh: {sample_X.shape[1]} đặc trưng (Kỳ vọng: 35)")
        print(f"[+] Ma trận 10 mẫu thử nghiệm: {sample_X.shape}")