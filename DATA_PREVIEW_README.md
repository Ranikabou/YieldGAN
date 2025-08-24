# Data Preview Functionality - Treasury GAN Dashboard

## Overview
The Treasury GAN Dashboard now includes comprehensive data preview functionality that allows users to visualize and analyze training data before starting GAN training. This feature automatically detects data types and provides appropriate visualizations.

## Features

### 🔍 **Automatic Data Type Detection**
- **Multi-Level Order Book Data**: Automatically detects bid/ask structures, price levels, and size information
- **Time Series Data**: Identifies sequential numeric data suitable for time series analysis
- **Smart Column Analysis**: Categorizes columns by type (bid, ask, price, size, level, depth)

### 📊 **Data Structure Analysis**
- **Shape Information**: Shows rows × columns dimensions
- **Column Types**: Displays data types for each column
- **Missing Values**: Reports any missing data
- **Summary Statistics**: Provides descriptive statistics for numeric columns

### 📈 **Time Series Visualization**
- **Interactive Charts**: Uses Chart.js for responsive time series plots
- **Multi-Series Support**: Shows up to 5 numeric columns simultaneously
- **Color-Coded Lines**: Different colors for each data series
- **Responsive Design**: Adapts to different screen sizes

### 📋 **Data Preview Table**
- **Sample Data**: Shows first 10 rows of data
- **Formatted Display**: Numbers are formatted to 4 decimal places
- **Scrollable Interface**: Handles wide datasets gracefully

## How to Use

### 1. **Access the Dashboard**
```
http://localhost:8081
```

### 2. **Select Training Data Source**
- Click the "📁 Upload CSV Dataset" button
- Choose a CSV file from your computer
- The system will automatically analyze and preview the data

### 3. **View Data Preview**
The preview section will show:
- **Data Structure**: File info, dimensions, data type
- **Time Series Plot**: Interactive chart of numeric columns
- **Data Table**: Sample rows with formatted values

### 4. **Start Training**
Once you're satisfied with the data preview:
- Click "▶️ Start Training" to begin GAN training
- The system will use the selected dataset

## Sample Data Files

### Multi-Level Order Book Data
- `sample_orderbook.csv`: Basic order book structure
- `treasury_orderbook_sample.csv`: Enhanced order book with spreads

### Time Series Data
- `sample_timeseries.csv`: Treasury yield curves and spreads

## Technical Implementation

### Backend Endpoints
- `POST /api/upload_csv`: Handles file uploads
- `GET /api/preview_csv?filename=<name>`: Returns data analysis and preview

### Data Analysis Features
- **Pandas Integration**: Uses pandas for CSV parsing and analysis
- **Smart Detection**: Identifies order book patterns automatically
- **Plot Data Generation**: Prepares time series data for visualization

### Frontend Components
- **File Input**: Hidden file input with custom button styling
- **Preview Section**: Collapsible section showing data analysis
- **Chart.js Integration**: Professional time series visualization
- **Responsive Tables**: Handles wide datasets with horizontal scrolling

## Data Type Detection Logic

### Order Book Indicators
The system looks for these keywords in column names:
- `bid`, `ask`, `price`, `size`, `level`, `depth`

### Multi-Level Detection
- **Bid Columns**: Columns containing "bid" in the name
- **Ask Columns**: Columns containing "ask" in the name
- **Price Columns**: Columns containing "price" in the name
- **Size Columns**: Columns containing "size" in the name
- **Level Columns**: Columns containing "level" in the name

### Time Series Detection
- **Numeric Columns**: Automatically identifies numeric data
- **Sequential Data**: Assumes time series if no order book indicators found
- **Plot Generation**: Creates time series charts for numeric columns

## Benefits

### 🎯 **Better Training Preparation**
- **Data Validation**: Verify data structure before training
- **Quality Assessment**: Identify potential issues early
- **Feature Understanding**: Understand what the model will learn

### 📊 **Professional Visualization**
- **Time Series Plots**: See data patterns and trends
- **Order Book Analysis**: Understand market microstructure
- **Interactive Charts**: Zoom, pan, and explore data

### 🚀 **Improved Workflow**
- **Data Selection**: Make informed decisions about training data
- **Error Prevention**: Catch data issues before training starts
- **Documentation**: Built-in data analysis and reporting

## Example Output

### Order Book Data
```json
{
  "data_type": "multi_level_order_book",
  "order_book_info": {
    "bid_columns": ["bid_price_1", "bid_size_1", "bid_price_2", "bid_size_2"],
    "ask_columns": ["ask_price_1", "ask_size_1", "ask_price_2", "ask_size_2"],
    "price_columns": ["bid_price_1", "ask_price_1", "bid_price_2", "ask_price_2"],
    "size_columns": ["bid_size_1", "ask_size_1", "bid_size_2", "ask_size_2"]
  }
}
```

### Time Series Data
```json
{
  "data_type": "time_series",
  "numeric_columns": ["2y_yield", "5y_yield", "10y_yield", "30y_yield"],
  "plot_data": {
    "2y_yield": [2.5, 2.51, 2.52, ...],
    "5y_yield": [3.0, 3.01, 3.02, ...],
    "index": [0, 1, 2, ...]
  }
}
```

## Future Enhancements

### 🔮 **Planned Features**
- **Data Quality Metrics**: Automated data quality scoring
- **Statistical Analysis**: Advanced statistical summaries
- **Data Cleaning Tools**: Built-in data preprocessing
- **Export Functionality**: Save analysis reports
- **Batch Processing**: Handle multiple files simultaneously

### 🎨 **UI Improvements**
- **Dark Mode**: Alternative color scheme
- **Customizable Charts**: User-defined chart options
- **Data Filtering**: Interactive data filtering tools
- **Comparison Views**: Side-by-side data comparison

## Troubleshooting

### Common Issues
1. **File Not Found**: Ensure CSV files are in the `data/csv/` directory
2. **Large Files**: Very large files may take time to process
3. **Encoding Issues**: Use UTF-8 encoding for best compatibility
4. **Chart Display**: Ensure JavaScript is enabled in your browser

### Performance Notes
- **Sample Size**: Charts show first 100 rows for performance
- **Memory Usage**: Large files are processed in chunks
- **Response Time**: Analysis typically completes in <1 second

---

**🎉 The data preview functionality is now fully integrated into your Treasury GAN Dashboard!**

You can now:
- Upload CSV files and see immediate data analysis
- View time series plots of your training data
- Automatically detect order book vs. time series data
- Make informed decisions about your training datasets
- Start training with confidence in your data quality 