import streamlit as st
from annotated_text import annotated_text
from ojd_daps_skills.pipeline.extract_skills.extract_skills import ExtractSkills
from app_utils import download_file_from_s3

st.set_page_config(
    page_title="Nesta Skills Extractor",
    page_icon="images/nesta_logo.png",
)


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

# ----------------- streamlit config ------------------#

# download s3 file
download_file_from_s3(local_path="fonts/AvertaDemo-Regular.otf")

with open("style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

# ----------------- streamlit config ------------------#
st.markdown(
    """
This app shows how Nesta's [Skills Extractor Library](https://github.com/nestauk/ojd_daps_skills) can extract skills from a job advert and then match those terms to skills from a standard list or ‘skills taxonomy’.

At present, you can choose to match extracted skills to one of two skills taxonomies that have been developed by other groups:

1. The [European Commission's ESCO taxonomy v1.1.1](https://esco.ec.europa.eu/en/classification/skill_main) which is a multilingual classification of European Skills, Competences, Qualifications and Occupations and;
2. [Lightcast's Open Skills taxonomy](https://lightcast.io/open-skills) (as of 22/11/22) which is an open source library of 32,000+ skills.
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

The Skills Extractor library was made possible by funding from the [Economic Statistics Centre of Excellence](https://www.escoe.ac.uk/).

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
    txt = txt.replace("\n", ". ")
    with st.spinner("🤖 Running algorithms..."):

        extracted_skills = es.extract_skills(txt)

    if "SKILL" in extracted_skills[0].keys():
        st.success(f"{len(extracted_skills[0]['SKILL'])} skill(s) extracted!", icon="💃")
        st.markdown(f"**The extracted skills are:** ")
        annotated_text(
            *[
                highlight
                for s in extracted_skills[0]["SKILL"]
                for highlight in [(s[0], "", "#F6A4B7"), " "]
            ]
        )
        st.markdown("")  # Add a new line
        st.markdown(f"**The _{app_mode}_ taxonomy skills are**: ")
        annotated_text(
            *[
                highlight
                for s in extracted_skills[0]["SKILL"]
                for highlight in [(s[1][0], "", "#FDB633"), " "]
            ]
        )

    else:
        st.warning("No skills were found in the job advert", icon="⚠️")
