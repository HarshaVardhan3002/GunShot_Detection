# Gunshot Localization System

AI-assisted acoustic localization system for detecting and triangulating gunshot events using 8 spatially distributed microphones.

## Features

- Real-time multi-channel audio capture (8 microphones)
- Gunshot detection using amplitude and frequency analysis
- Time Difference of Arrival (TDoA) calculation
- 2D/3D triangulation using multilateration
- Real-time coordinate output with confidence scoring

## Installation

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Configure microphone positions in `config/mic_positions.json`

3. Run the system:

```bash
python main.py --mic_config config/mic_positions.json --sampling_rate 48000
```

## Configuration

Edit `config/mic_positions.json` to set:

- Microphone positions (x, y, z coordinates)
- System parameters (sampling rate, detection thresholds, etc.)

## Usage

```bash
python main.py [options]

Options:
  --mic_config PATH     Path to microphone configuration file
  --sampling_rate RATE  Audio sampling rate in Hz (default: 48000)
  --log_level LEVEL     Logging level (DEBUG, INFO, WARNING, ERROR)
  --log_file PATH       Optional log file path
```

## Project Structure

```
gunshot-localizer/
├── audio_capture.py      # Multi-channel audio capture
├── gunshot_detector.py   # Gunshot detection algorithms
├── tdoa_localizer.py     # TDoA calculation and triangulation
├── intensity_filter.py   # Signal quality filtering
├── config_manager.py     # Configuration management
├── utils.py              # Utility functions
├── main.py               # Main application entry point
├── config/
│   └── mic_positions.json # Microphone configuration
├── logs/                 # Log files
└── requirements.txt      # Python dependencies
```
