#!/bin/bash

echo "Starting dental implant debug script..."
PYTHONPATH=/home/mswr/Automated\ Dental\ Implant\ Recognition\ in\ Radiographic\ Images\ Transfer\ Learning\ Implementation\ with\ EfficientNetB3
python3 debug/dental_implant_debug.py "$@" 2>&1 | tee debug_output.log
echo "Script execution completed. Output logged to debug_output.log"
