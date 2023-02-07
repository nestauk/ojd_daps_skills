# Demo App

This directory contains the scripts needed to generate the streamlit demo app.

## Run Locally

To run the app locally (in `ojd_daps_skills/app`):

```
pip install -r requirements.txt
streamlit run app.py
```

## Docker

To containerise the app:

```
docker build -t demo_app .
docker run -p 8501:8501 demp_app
```
