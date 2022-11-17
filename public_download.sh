#!/bin/sh
mkdir downloaded_files/
mkdir downloaded_files/outputs/
aws s3 cp s3://open-jobs-indicators/escoe_extension/outputs/ downloaded_files/outputs/ --recursive
