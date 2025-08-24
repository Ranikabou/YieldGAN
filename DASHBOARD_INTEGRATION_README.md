# Dashboard Integration for GAN Training

This document explains how to use the modified `train_gan_csv.py` script that now sends real-time training data and progress updates to the GAN dashboard via SSE channels.

## Overview

The `train_gan_csv.py` script has been enhanced with dashboard channel integration, allowing it to send:
- **Training Metrics**: Generator loss, discriminator loss, real scores, fake scores
- **Progress Updates**: Epoch-by-epoch progress (0%, 25%, 50%, 75%, 100%)
- **Real-time Updates**: Data is sent to the dashboard as training progresses

## How It Works

### 1. Dashboard Channel Sender
The script now includes a `DashboardChannelSender` class that:
- Connects to the dashboard at `http://localhost:8081` (configurable)
- Sends training data to `/training_data` endpoint
- Sends progress updates to `/progress_data` endpoint
- Handles connection errors gracefully

### 2. Custom Training Loop
When a dashboard sender is provided, the script uses a custom training loop that:
- Sends progress updates at the start and end of each epoch
- Sends training metrics after each epoch completion
- Maintains the same training logic but with dashboard integration
- Falls back to default training if no dashboard sender is provided

## Usage

### 1. Start the Dashboard
First, make sure the GAN dashboard is running:
```bash
python gan_dashboard.py
```

### 2. Run Training with Dashboard Integration
```bash
python train_gan_csv.py --config config/gan_config.yaml --data data/treasury_orderbook_sample.csv
```

The script will automatically:
- Detect if the dashboard is running
- Send real-time updates during training
- Display progress in the dashboard interface

### 3. Test Dashboard Integration
You can test the dashboard integration independently:
```bash
python test_dashboard_integration.py
```

This will send test data to verify the channels are working.

## Configuration

### Dashboard URL
The dashboard URL can be configured in your config file:
```yaml
dashboard:
  url: "http://localhost:8081"
```

Or it defaults to `http://localhost:8081`.

### Training Parameters
All existing training parameters remain the same:
- `--config`: Path to configuration file
- `--sequence-length`: Length of training sequences
- `--skip-training`: Skip training and only evaluate
- `--checkpoint`: Load from checkpoint

## What You'll See in the Dashboard

### Training Channel
- Real-time generator and discriminator loss
- Real and fake scores for each epoch
- Training completion status

### Progress Channel
- Epoch-by-epoch progress updates
- Visual progress bars and indicators
- Training timeline

### Log Channel
- Training logs and error messages
- Real-time log file monitoring

## Troubleshooting

### Dashboard Not Accessible
If you see warnings about dashboard communication:
1. Ensure the dashboard is running on the correct port
2. Check firewall settings
3. Verify the dashboard URL in your config

### Training Continues Without Dashboard
The script will continue training even if the dashboard is unavailable:
- It falls back to default training mode
- Logs are still saved locally
- Training metrics are still displayed in the console

### Performance Impact
The dashboard integration adds minimal overhead:
- Small delays between epochs (0.1 seconds)
- HTTP requests are non-blocking
- Training performance is not affected

## Example Output

When running with dashboard integration, you'll see:
```
🎯 Training data sent to dashboard: Epoch 1, Gen Loss: 0.8234, Disc Loss: 0.7123
📊 Progress 100% sent to dashboard for epoch 1
🎯 Training data sent to dashboard: Epoch 2, Gen Loss: 0.7654, Disc Loss: 0.6891
📊 Progress 100% sent to dashboard for epoch 2
```

## Benefits

1. **Real-time Monitoring**: See training progress as it happens
2. **Visual Feedback**: Interactive charts and progress indicators
3. **Remote Monitoring**: Monitor training from anywhere via web browser
4. **Historical Data**: Track training metrics over time
5. **Debugging**: Identify issues early in the training process

## Next Steps

1. Start the dashboard: `python gan_dashboard.py`
2. Run training: `python train_gan_csv.py`
3. Open your browser to `http://localhost:8081`
4. Watch real-time training progress!

The dashboard will now show live training data just like `test_dashboard_channels.py` did, but with actual GAN training instead of simulated data. 