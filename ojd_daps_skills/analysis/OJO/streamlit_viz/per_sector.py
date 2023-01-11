"""
Run

streamlit run ojd_daps_skills/analysis/OJO/streamlit_viz/per_sector.py
"""
import os

import pandas as pd
import streamlit as st
import altair as alt

from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    load_s3_data,
)
from ojd_daps_skills import bucket_name


@st.cache
def load_data():

    s3 = get_s3_resource()

    s3_folder = "escoe_extension/outputs/data"

    file_name = os.path.join(
        s3_folder, "streamlit_viz", "similar_sectors_per_sector_sample.json"
    )
    similar_sectors_per_sector = load_s3_data(s3, bucket_name, file_name)

    file_name = os.path.join(
        s3_folder, "streamlit_viz", "top_skills_per_sector_sample.json"
    )
    top_skills_per_sector = load_s3_data(s3, bucket_name, file_name)

    file_name = os.path.join(
        s3_folder, "streamlit_viz", "number_job_adverts_per_sector_sample.json"
    )
    number_job_adverts_per_sector = load_s3_data(s3, bucket_name, file_name)

    total_num_job_adverts = sum(number_job_adverts_per_sector.values())
    percentage_job_adverts_per_sector = {
        sector_name: round(num_ads * 100 / total_num_job_adverts, 2)
        for sector_name, num_ads in number_job_adverts_per_sector.items()
    }

    return (
        similar_sectors_per_sector,
        top_skills_per_sector,
        number_job_adverts_per_sector,
        percentage_job_adverts_per_sector,
    )


(
    similar_sectors_per_sector,
    top_skills_per_sector,
    number_job_adverts_per_sector,
    percentage_job_adverts_per_sector,
) = load_data()


st.title("Central Government Use Case")
st.markdown(
    "For a selected occupation you can see the most common skills, most similar jobs (based on the skills asked for), and see a network of skills connected by when they frequently co-occur with one another."
)


top_sectors = [k for k, v in number_job_adverts_per_sector.items() if v > 500]

sector = st.selectbox("Select sector", top_sectors)

top_skills = pd.DataFrame.from_dict(
    top_skills_per_sector[sector], orient="index", columns=["percent"]
)
top_skills.sort_values(by="percent", inplace=True, ascending=False)
top_skills = top_skills[0:10]
top_skills["sector"] = top_skills.index

similar_sectors = pd.DataFrame.from_dict(
    similar_sectors_per_sector[sector], orient="index", columns=["euclid_dist"]
)
similar_sectors.sort_values(
    by="euclid_dist", inplace=True, ascending=True
)  # Smaller Euclid dist is closer
similar_sectors = similar_sectors[0:10]
similar_sectors["sector"] = similar_sectors.index


common_skills_chart = (
    alt.Chart(top_skills)
    .mark_bar(size=10, opacity=0.8, color="#0000FF")
    .encode(
        y=alt.Y("sector", sort=None, axis=alt.Axis(title=None)),
        x=alt.X(
            "percent", axis=alt.Axis(title="Percentage of job adverts with this skill")
        ),
        tooltip=["percent"],
    )
    .properties(
        title="Most common skills",
        # height=100,
        width=75,
    )
)

similar_jobs_chart = (
    alt.Chart(similar_sectors)
    .mark_bar(size=10, opacity=0.8, color="#0000FF")
    .encode(
        y=alt.Y("sector", sort=None, axis=alt.Axis(title=None)),
        x=alt.X("euclid_dist", axis=alt.Axis(title="Skill similarity")),
        tooltip=["euclid_dist"],
    )
    .properties(title="Most similar jobs", width=75)
)

metric1, metric2 = st.columns((1, 1))
metric1.metric(
    label="Number of job adverts", value=number_job_adverts_per_sector[sector]
)
metric2.metric(
    label="Percentage of all job adverts",
    value=f"{percentage_job_adverts_per_sector[sector]}%",
)

st.altair_chart(
    alt.hconcat(common_skills_chart, similar_jobs_chart).configure_axis(labelLimit=300),
    use_container_width=True,
)
