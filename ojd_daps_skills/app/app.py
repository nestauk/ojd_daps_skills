import streamlit as st
from ojd_daps_skills.pipeline.extract_skills.extract_skills import ExtractSkills
import os
from typing import List
import pandas as pd
from IPython.display import HTML

st.set_page_config(
    page_title="Nesta Skills Extractor",
    page_icon="images/nesta_logo.png",
)


def hover(hover_color="#d3d3d3"):
    return dict(selector="tr:hover", props=[("background-color", "%s" % hover_color)])


styles = [
    hover(),
    dict(
        selector="th",
        props=[
            ("font-size", "125%"),
            ("text-align", "center"),
            ("font-weight", "bold"),
            ("font-style", "italic"),
        ],
    ),
    dict(selector="caption", props=[("caption-side", "bottom")]),
]

# CSS to inject contained in a string
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """


def format_skills_list(
    raw_skills_list: List[str], mapped_skills_list: List[str], tax: str
) -> pd.DataFrame:
    """Formats raw and mapped skills into a dataframe

    Args:
        raw_skills_list (List[str]): List of raw skills extracted from job advert
        mapped_skills_list (List[str]): List of mapped skills
        tax (str): Taxonomy used to map skills (esco or lightcast

    Returns:
        pd.DataFrame: Dataframe of raw and mapped skills
    """

    df = pd.DataFrame(
        {"Extracted Skill": raw_skills_list, f"Mapped {tax} Skill": mapped_skills_list}
    ).style.set_table_styles(styles)

    return df


def hash_config_name(es):
    # custom hash function in order to use st.cache
    return es.taxonomy_name


@st.cache(hash_funcs={ExtractSkills: hash_config_name})
def load_model(app_mode):

    if app_mode == esco_tax:
        es = ExtractSkills(config_name="extract_skills_esco", local=True)
    elif app_mode == lightcast_tax:
        es = ExtractSkills(config_name="extract_skills_lightcast", local=True)
    es.load()
    return es


image_dir = "images/nesta_escoe_skills.png"
st.image(image_dir)

# ----------------- streamlit config START ------------------#

with open("style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

# ----------------- streamlit config END ------------------#
st.markdown(
    """
This app shows how Nesta's [Skills Extractor Library](https://github.com/nestauk/ojd_daps_skills) can extract skills from a job advert and then match those terms to skills from a standard list or ‘skills taxonomy’.

At present, you can choose to match extracted skills to one of two skills taxonomies that have been developed by other groups:

1. The [European Commission's ESCO taxonomy v1.1.1](https://esco.ec.europa.eu/en/classification/skill_main) which is a multilingual classification of European Skills, Competences, Qualifications and Occupations and;
2. [Lightcast's Open Skills taxonomy](https://lightcast.io/open-skills) (as of 22/11/22) which is open source library of 32,000+ skills.
"""
)

st.warning(
    "As with any algorithm, our approach has limitations. As a result, we cannot guarantee the accuracy of every extracted or mapped skill. To learn more about the strengths and limitations, consult our [model cards](https://nestauk.github.io/ojd_daps_skills/build/html/model_card.html).",
    icon="🤖",
)

st.markdown(
    """
If you would like to extract skills from many adverts, you can use our [open-source python library](https://github.com/nestauk/ojd_daps_skills) by simply `pip install ojd-daps-skills` and following the [instructions in our documentation](https://nestauk.github.io/ojd_daps_skills/build/html/about.html).

If you would like to explore how the algorithm can provide new insights, check out this interactive blog (link pending) that analyses extracted skills from thousands of job adverts.

The Skills Extractor library was made possible by funding from the Economic Statistics Centre of Excellence.

If you have any feedback or questions about the library or app, do reach out to dataanalytics@nesta.org.uk.
"""
)

esco_tax = "ESCO"
lightcast_tax = "Lightcast"
app_mode = st.selectbox("🗺️ Choose a taxonomy to map onto", [esco_tax, lightcast_tax])
txt = st.text_area(
    "✨ Add your job advert text here ... or try out the phrase 'You must have strong communication skills.'",
    "",
)

es = load_model(app_mode)

button = st.button("Extract Skills")

if button:
    with st.spinner("🤖 Running algorithms..."):

        extracted_skills = es.extract_skills(txt)

    if "SKILL" in extracted_skills[0].keys():
        st.success(f"{len(extracted_skills[0]['SKILL'])} skill(s) extracted!", icon="💃")
        raw_skills = [s[0] for s in extracted_skills[0]["SKILL"]]
        mapped_skills = [s[1][0] for s in extracted_skills[0]["SKILL"]]
        skills_table = format_skills_list(raw_skills, mapped_skills, tax=app_mode)
        st.markdown(hide_table_row_index, unsafe_allow_html=True)
        st.table(skills_table)

    else:
        st.warning("No skills were found in the job advert", icon="⚠️")
