#!/usr/bin/env bash

# Make sure the script fails if any command fails
set -e

# Install system dependencies
apt-get update
apt-get install -y tesseract-ocr libtesseract-dev

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
