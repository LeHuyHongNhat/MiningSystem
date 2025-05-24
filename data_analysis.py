import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Đọc dữ liệu
file_path = Path("D:/Python Projects/SYSTEM/MiningSystem/online_retail_II.xlsx")
df = pd.read_excel(file_path)

# 1. Thông tin cơ bản về dataset
print("\n=== THÔNG TIN CƠ BẢN VỀ DATASET ===")
print(f"Số lượng dòng: {df.shape[0]}")
print(f"Số lượng cột: {df.shape[1]}")
print("\nTên các cột:")
print(df.columns.tolist())

# 2. Kiểm tra kiểu dữ liệu
print("\n=== KIỂU DỮ LIỆU ===")
print(df.dtypes)

# 3. Kiểm tra giá trị null
print("\n=== KIỂM TRA GIÁ TRỊ NULL ===")
null_counts = df.isnull().sum()
print("Số lượng giá trị null trong mỗi cột:")
print(null_counts)
print(f"\nTổng số giá trị null: {df.isnull().sum().sum()}")

# 4. Thống kê mô tả
print("\n=== THỐNG KÊ MÔ TẢ ===")
print(df.describe())

# 5. Kiểm tra giá trị duy nhất trong mỗi cột
print("\n=== SỐ LƯỢNG GIÁ TRỊ DUY NHẤT TRONG MỖI CỘT ===")
for column in df.columns:
    unique_count = df[column].nunique()
    print(f"{column}: {unique_count} giá trị duy nhất")

# 6. Visualize dữ liệu
plt.figure(figsize=(15, 10))

# 6.1. Biểu đồ phân phối giá trị null
plt.subplot(2, 2, 1)
null_counts.plot(kind='bar')
plt.title('Phân phối giá trị null theo cột')
plt.xticks(rotation=45)
plt.tight_layout()

# 6.2. Biểu đồ box plot cho các cột số
plt.subplot(2, 2, 2)
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns].boxplot()
plt.title('Box plot cho các cột số')
plt.xticks(rotation=45)
plt.tight_layout()

# 6.3. Biểu đồ histogram cho các cột số
plt.subplot(2, 2, 3)
for column in numeric_columns:
    plt.hist(df[column].dropna(), bins=30, alpha=0.5, label=column)
plt.title('Histogram cho các cột số')
plt.legend()
plt.tight_layout()

# 6.4. Heatmap tương quan
plt.subplot(2, 2, 4)
correlation = df[numeric_columns].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Ma trận tương quan')
plt.tight_layout()

plt.show()

# 7. Kiểm tra giá trị trùng lặp
print("\n=== KIỂM TRA GIÁ TRỊ TRÙNG LẶP ===")
duplicate_rows = df.duplicated().sum()
print(f"Số lượng dòng trùng lặp: {duplicate_rows}")

# 8. Kiểm tra giá trị ngoại lai (outliers)
print("\n=== KIỂM TRA GIÁ TRỊ NGOẠI LAI ===")
for column in numeric_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    print(f"\n{column}:")
    print(f"Số lượng giá trị ngoại lai: {len(outliers)}")
    print(f"Giá trị ngoại lai nhỏ nhất: {outliers.min() if len(outliers) > 0 else 'Không có'}")
    print(f"Giá trị ngoại lai lớn nhất: {outliers.max() if len(outliers) > 0 else 'Không có'}") 