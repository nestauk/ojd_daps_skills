# Streamlit app

We created a visualistion in Streamlit in order for others to explore the dataset.

The first step in creating this visualisation is preparing the data for it. This is done by running:

```
python ojd_daps_skills/analysis/OJO/process_viz_data.py
```

[This requires the data saved by running `python ojd_daps_skills/analysis/ojo/get_skill_occurrences_matrix.py`]

## In progress

Exploring the data by sector (a.k.a occupation) can be seen by running:

```
streamlit run ojd_daps_skills/analysis/OJO/streamlit_viz/per_sector.py
```
