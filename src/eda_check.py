import os
import numpy as np
import matplotlib.pyplot as plt
from dataset import load_neu_dataset_recursive, CLASSES

def run_eda_and_check(images, labels, output_plot="eda_samples.png"):
    print("=" * 60)
    print("       BÁO CÁO KIỂM TRA CHẤT LƯỢNG & KHÁM PHÁ DỮ LIỆU (EDA)       ")
    print("=" * 60)
    
    # 1. Kiểm tra cấu trúc & kiểu dữ liệu
    print(f"1. Tổng số ảnh nạp được: {len(images)}")
    print(f"   - Kích thước mảng (Shape): {images.shape}")
    print(f"   - Kiểu dữ liệu pixel (dtype): {images.dtype}")
    print(f"   - Mức xám thực tế: Min = {images.min()}, Max = {images.max()}")
    
    # 2. Kiểm tra ảnh rỗng / lỗi
    corrupted_count = 0
    blank_count = 0
    for img in images:
        if img is None:
            corrupted_count += 1
        elif np.all(img == 0) or np.all(img == 255):
            blank_count += 1
            
    print("\n2. Kiểm tra tính toàn vẹn:")
    print(f"   - Số ảnh bị hỏng/rỗng: {corrupted_count}")
    print(f"   - Số ảnh bất thường (đồng nhất hoàn toàn): {blank_count}")
    
    # 3. Phân bố tần suất các lớp (Class Distribution)
    print("\n3. Phân bố 6 lớp khuyết tật:")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        pct = (c / len(labels)) * 100
        print(f"   - Lớp [{u}] {CLASSES[u]:<16}: {c:4d} ảnh ({pct:.1f}%)")
    print("=" * 60)
    
    # 4. Trực quan hóa hình ảnh mẫu đại diện (Dành cho Slide & Báo cáo)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    fig.suptitle("Visual Samples of NEU Defect Classes", fontsize=14, fontweight='bold')
    
    for i, class_name in enumerate(CLASSES):
        sample_idx = np.where(labels == i)[0][0]
        ax = axes[i // 3, i % 3]
        ax.imshow(images[sample_idx], cmap='gray')
        ax.set_title(f"Class {i}: {class_name}\n({counts[i]} images)", fontsize=11)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"[+] Đã xuất file ảnh trực quan hóa: {output_plot}")
    plt.close()

if __name__ == "__main__":
    DATA_PATH = r"D:\Tài Liệu\DIP_Project\Surface_Defect_Project\Dataset\NEU-DET\train"
    
    imgs, lbls = load_neu_dataset_recursive(DATA_PATH)
    if len(imgs) > 0:
        run_eda_and_check(imgs, lbls)