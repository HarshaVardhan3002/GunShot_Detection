#!/bin/bash
# Gunshot Localization System Startup Script

# Set working directory
cd "C:\Users\Von3002\Desktop\Gunshot Detection\gunshot-localizer"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the system with default configuration
python main.py --config config/default_config.json --verbose

# Keep script running
while true; do
    echo "System stopped. Restarting in 5 seconds..."
    sleep 5
    python main.py --config config/default_config.json --verbose
done
