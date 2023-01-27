# Streamlit app

We created a visualistion in Streamlit in order for others to explore the dataset.

The first step in creating this visualisation is preparing the data for it. This is done by running:

```
python ojd_daps_skills/analysis/OJO/process_viz_data.py
```

[This requires the data saved by running `python ojd_daps_skills/analysis/ojo/get_skill_occurrences_matrix.py`]

## In progress

A preliminary analysis app can be seen by running:

```
pip install -r requirements_analysis_app.txt
streamlit run ojd_daps_skills/analysis/OJO/streamlit_viz/streamlit_viz.py
```

Make sure that you have the appropriate `AvertaDemo-Regular.otf` font saved to `/.fonts/` directory to ensure the app runs with Nesta brand compliant fonts.
