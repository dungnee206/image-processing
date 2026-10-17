import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_histogram_equalization(image):
    """
    Cân bằng lược đồ mức xám toàn cục (Global Histogram Equalization).
    Kéo giãn dải động độ xám trên toàn bộ khoảng [0, 255].
    """
    return cv2.equalizeHist(image)

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Cân bằng lược đồ mức xám cục bộ thích nghi giới hạn độ tương phản (CLAHE).
    Làm rõ vân bề mặt cục bộ mà không gây hiện tượng bão hòa nhiễu.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def apply_median_filter(image, ksize=3):
    """
    Lọc trung vị với kích thước mặt nạ ksize x ksize.
    Triệt tiêu nhiễu muối tiêu tốt và bảo toàn cạnh viền của vết nứt, vết xước.
    """
    return cv2.medianBlur(image, ksize)

def apply_gaussian_filter(image, ksize=3, sigma=1.0):
    """
    Lọc tuyến tính Gaussian để làm mịn nền kim loại trước khi bắt biên.
    """
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)

def apply_sobel(image, ksize=3):
    """
    Toán tử Sobel tính xấp xỉ đạo hàm bậc nhất theo 2 hướng x và y.
    Độ lớn biên: G = sqrt(Gx^2 + Gy^2) hoặc |Gx| + |Gy|.
    """
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

def apply_laplacian(image, ksize=3):
    """
    Toán tử đạo hàm bậc hai Laplace phát hiện điểm chuyển mức xám đột ngột.
    """
    lap = cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)
    return cv2.convertScaleAbs(lap)

def preprocess_image(image, method="raw"):
    """
    Hàm chuẩn hóa tiền xử lý áp dụng cho từng mẫu dữ liệu.
    Hỗ trợ: 'raw', 'hist_eq', 'clahe', 'median', 'gaussian', 'sobel', 'laplacian'
    """
    if method == "raw":
        return image
    elif method == "hist_eq":
        return apply_histogram_equalization(image)
    elif method == "clahe":
        return apply_clahe(image)
    elif method == "median":
        return apply_median_filter(image, ksize=3)
    elif method == "gaussian":
        return apply_gaussian_filter(image, ksize=3)
    elif method == "sobel":
        # Lọc mịn nhẹ trước khi bắt biên để tránh bắt nhầm nhiễu hạt kim loại
        smoothed = apply_median_filter(image, ksize=3)
        return apply_sobel(smoothed, ksize=3)
    elif method == "laplacian":
        smoothed = apply_median_filter(image, ksize=3)
        return apply_laplacian(smoothed, ksize=3)
    else:
        raise ValueError(f"Phương thức '{method}' không được hỗ trợ!")

def preprocess_dataset(images, method="raw"):
    """
    Áp dụng tiền xử lý đồng loạt cho toàn bộ tập ảnh nén vào mảng numpy.
    """
    print(f"[*] Đang áp dụng tiền xử lý: '{method}' cho {len(images)} ảnh...")
    processed_images = np.array([preprocess_image(img, method=method) for img in images], dtype=np.uint8)
    return processed_images

if __name__ == "__main__":
    from dataset import load_neu_dataset_recursive
    
    # Định nghĩa trực tiếp đường dẫn tại đây
    DATA_PATH = r"D:\Tài Liệu\DIP_Project\Surface_Defect_Project\Dataset\NEU-DET\train"
    
    print("[*] Chạy thử nghiệm trực quan hóa các bộ lọc...")
    imgs, lbls = load_neu_dataset_recursive(DATA_PATH)
    
    if len(imgs) > 0:
        sample_img = imgs[0]  # Lấy 1 ảnh mẫu
        
        methods = ["raw", "hist_eq", "clahe", "median", "sobel", "laplacian"]
        titles = ["Ảnh gốc (Raw)", "Hist Equalization", "CLAHE", "Median Filter", "Sobel Edge", "Laplacian Edge"]
        
        plt.figure(figsize=(15, 6))
        for i, (m, t) in enumerate(zip(methods, titles)):
            res = preprocess_image(sample_img, method=m)
            plt.subplot(2, 3, i + 1)
            plt.imshow(res, cmap="gray")
            plt.title(t, fontsize=11, fontweight="bold")
            plt.axis("off")
            
        plt.tight_layout()
        plt.savefig("preprocessing_comparison.png", dpi=300)
        plt.close()
        print("[+] Đã xuất file ảnh so sánh các phương pháp: preprocessing_comparison.png")