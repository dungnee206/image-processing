import sys
import os
import numpy as np
import matplotlib.pyplot as plt

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from dataset import load_neu_dataset_recursive, CLASSES

DATA_PATH = r"D:\Tài Liệu\DIP_Project\Surface_Defect_Project\Dataset\NEU-DET\train"

def inspect_dataset_integrity(images, labels):
    print("=" * 60)
    print("           BÁO CÁO KIỂM TRA CHẤT LƯỢNG DỮ LIỆU           ")
    print("=" * 60)
    
    # 1. Kích thước & dải mức xám
    print(f"1. Tổng số ảnh: {len(images)}")
    print(f"   - Kích thước mảng: {images.shape}")
    print(f"   - Mức xám: Min = {images.min()}, Max = {images.max()}")
    
    # 2. Kiểm tra ảnh rỗng
    corrupted_count = sum(1 for img in images if img is None)
    blank_count = sum(1 for img in images if np.all(img == 0) or np.all(img == 255))
    print(f"\n2. Tính toàn vẹn:")
    print(f"   - Số ảnh bị hỏng/rỗng: {corrupted_count}")
    print(f"   - Số ảnh đồng nhất hoàn toàn: {blank_count}")
    
    # 3. Phân bố 6 lớp
    print("\n3. Thống kê phân bố 6 lớp:")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        pct = (c / len(labels)) * 100
        print(f"   - Lớp [{u}] {CLASSES[u]:<16}: {c:4d} ảnh ({pct:.1f}%)")
    print("=" * 60)

def export_eda_visualizations(images, labels):
    # 1. Xuất lưới ảnh mẫu đại diện 6 lớp (eda_samples.png)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    fig.suptitle("Visual Samples of NEU Steel Surface Defects", fontsize=14, fontweight='bold')
    
    for i, class_name in enumerate(CLASSES):
        sample_idx = np.where(labels == i)[0][0]
        ax = axes[i // 3, i % 3]
        ax.imshow(images[sample_idx], cmap='gray')
        ax.set_title(f"Class {i}: {class_name}", fontsize=11)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig("eda_samples.png", dpi=300)
    plt.close()
    print("[+] Đã xuất file: eda_samples.png")

    # 2. Xuất biểu đồ cột phân bố số lượng ảnh (eda_distribution.png)
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(9, 4.5))
    bars = plt.bar([CLASSES[u] for u in unique], counts, color='teal', edgecolor='black', alpha=0.75)
    plt.title("Phan Bo So Luong Mau 6 Lop Khuyet Tat", fontsize=12, fontweight='bold')
    plt.xlabel("Lop Khuyet Tat")
    plt.ylabel("So Luong Anh")
    plt.ylim(0, max(counts) + 50)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 5, int(yval), ha='center', va='bottom', fontsize=10)
        
    plt.tight_layout()
    plt.savefig("eda_distribution.png", dpi=300)
    plt.close()
    print("[+] Đã xuất file: eda_distribution.png")

if __name__ == "__main__":
    imgs, lbls = load_neu_dataset_recursive(DATA_PATH)
    if len(imgs) > 0:
        inspect_dataset_integrity(imgs, lbls)
        export_eda_visualizations(imgs, lbls)
