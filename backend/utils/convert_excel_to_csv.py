import pandas as pd

# Read the Excel file
excel_file = r"D:\Python Projects\SYSTEM\MiningSystem\online_retail_II.xlsx"
df = pd.read_excel(excel_file)

# Save as CSV
csv_file = r"D:\Python Projects\SYSTEM\MiningSystem\online_retail_II.csv"
df.to_csv(csv_file, index=False)

print(f"Conversion completed. CSV file saved at: {csv_file}") 