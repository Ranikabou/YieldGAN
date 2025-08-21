# 📊 CSV Data Source Setup Guide

This guide shows you how to modify your Treasury GAN project to use real CSV data files instead of API data.

## 🔄 **What Changes:**

### **1. New Files Created:**
- `data/csv_collector.py` - CSV data loading and processing
- `config/csv_config.yaml` - Configuration for CSV data sources
- `train_gan_csv.py` - Modified training script for CSV data
- `CSV_SETUP_GUIDE.md` - This guide

### **2. Modified Components:**
- Data collection pipeline now reads from CSV files
- Configuration supports CSV file structure definitions
- Training script adapted for CSV data processing

## 📁 **CSV File Structure Requirements:**

### **Treasury Yields CSV (`treasury_yields.csv`):**
```csv
date,2Y,5Y,10Y,30Y,SOFR
2022-01-01,4.25,4.15,4.05,3.85,5.25
2022-01-02,4.30,4.20,4.10,3.90,5.30
2022-01-03,4.28,4.18,4.08,3.88,5.28
...
```

**Required Columns:**
- `date` - Date in YYYY-MM-DD format
- `2Y` - 2-year treasury yield
- `5Y` - 5-year treasury yield  
- `10Y` - 10-year treasury yield
- `30Y` - 30-year treasury yield
- `SOFR` - Secured Overnight Financing Rate

### **Order Book Data CSV (`order_book_data.csv`):**
```csv
timestamp,instrument,level,bid_price,bid_size,ask_price,ask_size
2022-01-01 09:30:00,2Y,1,99.85,1000000,99.87,1000000
2022-01-01 09:30:00,2Y,2,99.84,2000000,99.88,2000000
2022-01-01 09:30:00,2Y,3,99.83,3000000,99.89,3000000
...
```

**Required Columns:**
- `timestamp` - Timestamp in YYYY-MM-DD HH:MM:SS format
- `instrument` - Treasury instrument (2Y, 5Y, 10Y, 30Y, SOFR)
- `level` - Order book level (1-5)
- `bid_price` - Bid price at this level
- `bid_size` - Bid size at this level
- `ask_price` - Ask price at this level
- `ask_size` - Ask size at this level

### **Features CSV (`features.csv`):**
```csv
date,feature_1,feature_2,feature_3
2022-01-01,0.85,0.92,0.78
2022-01-02,0.87,0.89,0.81
2022-01-03,0.83,0.94,0.79
...
```

**Required Columns:**
- `date` - Date in YYYY-MM-DD format
- `feature_1`, `feature_2`, `feature_3` - Your custom features

## 🚀 **Setup Steps:**

### **Step 1: Create CSV Directory**
```bash
mkdir -p data/csv
```

### **Step 2: Add Your CSV Files**
Place your CSV files in the `data/csv/` directory:
```bash
data/csv/
├── treasury_yields.csv
├── order_book_data.csv
└── features.csv
```

### **Step 3: Update Configuration**
Edit `config/csv_config.yaml` to match your CSV structure:
```yaml
data_source:
  csv_directory: "data/csv"
  csv_structure:
    treasury:
      date_column: "date"  # Your actual column name
      yield_columns: ["2Y", "5Y", "10Y", "30Y", "SOFR"]  # Your actual column names
```

### **Step 4: Run CSV Training**
```bash
# Use the CSV-specific training script
python train_gan_csv.py --config config/csv_config.yaml

# Or specify custom sequence length
python train_gan_csv.py --config config/csv_config.yaml --sequence-length 200
```

## ⚙️ **Configuration Options:**

### **Data Processing:**
```yaml
data_processing:
  sequence_length: 100        # Length of sequences for GAN
  handle_missing: "forward_fill"  # How to handle missing data
  normalize_data: true        # Whether to normalize data
  normalization_method: "standard"  # Standard, minmax, or robust
```

### **Feature Engineering:**
```yaml
data_processing:
  calculate_returns: true     # Calculate daily returns
  calculate_volatility: true  # Calculate rolling volatility
  volatility_window: 20       # Window for volatility calculation
  add_technical_indicators: true  # Add technical indicators
```

## 🔍 **CSV File Validation:**

### **Check Your CSV Files:**
```bash
# Check if files exist
ls -la data/csv/

# Validate CSV structure
python -c "
import pandas as pd
df = pd.read_csv('data/csv/treasury_yields.csv')
print('Columns:', df.columns.tolist())
print('Shape:', df.shape)
print('Sample data:')
print(df.head())
"
```

### **Common Issues & Solutions:**

| Issue | Solution |
|-------|----------|
| **Missing columns** | Add required columns to your CSV |
| **Wrong date format** | Ensure dates are YYYY-MM-DD |
| **Missing values** | Use forward_fill or interpolation |
| **Wrong data types** | Convert to numeric where needed |

## 📊 **Dashboard Integration:**

The dashboard will automatically detect and display your CSV data:

1. **Data Analysis Tab**: Shows your CSV files with sizes and timestamps
2. **Training Progress Tab**: Monitors CSV-based training
3. **Results Tab**: Displays results from CSV training

## 🎯 **Example CSV Files:**

### **Minimal Treasury Yields:**
```csv
date,2Y,5Y,10Y,30Y,SOFR
2022-01-01,4.25,4.15,4.05,3.85,5.25
2022-01-02,4.30,4.20,4.10,3.90,5.30
```

### **Minimal Order Book:**
```csv
timestamp,instrument,level,bid_price,bid_size,ask_price,ask_size
2022-01-01 09:30:00,2Y,1,99.85,1000000,99.87,1000000
2022-01-01 09:30:00,2Y,2,99.84,2000000,99.88,2000000
```

## 🚨 **Important Notes:**

1. **Column Names**: Must match exactly what's in your CSV files
2. **Date Format**: Use YYYY-MM-DD for dates, YYYY-MM-DD HH:MM:SS for timestamps
3. **Data Types**: Ensure numeric columns contain only numbers
4. **Missing Data**: Handle missing values appropriately
5. **File Encoding**: Use UTF-8 encoding for CSV files

## 🔄 **Switching Back to API:**

If you want to switch back to API data:
```bash
# Use the original training script
python train_gan.py --config config/gan_config.yaml
```

## 📈 **Performance Tips:**

- **Large CSV files**: Consider chunking for memory efficiency
- **Real-time data**: Update CSV files before each training run
- **Data quality**: Validate CSV data before training
- **Backup**: Keep original CSV files as backup

---

**🎯 Ready to use your own CSV data? Follow the setup steps above and start training with real data!** 