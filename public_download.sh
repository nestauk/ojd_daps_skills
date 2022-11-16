#!/bin/sh
mkdir downloaded_files/
aws s3 cp s3://open-jobs-indicators/escoe_extension/ downloaded_files/ --recursive
