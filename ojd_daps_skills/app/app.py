import streamlit as st
from ojd_daps_skills.pipeline.extract_skills.extract_skills import ExtractSkills
import os


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


image_dir = "nesta_escoe_skills.png"
st.image(image_dir)

with open("style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

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
        st.success("skills extracted!", icon="💃")
        raw_skills = ", ".join([s[0] for s in extracted_skills[0]["SKILL"]])
        mapped_skills = ", ".join([s[1][0] for s in extracted_skills[0]["SKILL"]])
        if len(extracted_skills[0]["SKILL"]) == 1:
            st.markdown(f"**{len(extracted_skills[0]['SKILL'])}** skill was extracted.")
        else:
            st.markdown(
                f"**{len(extracted_skills[0]['SKILL'])}** skills were extracted."
            )
        st.markdown(f"**The extracted skills are:** {raw_skills}")
        st.markdown(f"**The _{app_mode}_ taxonomy skills are**: {mapped_skills}")
    else:
        st.warning("No skills were found in the job advert", icon="⚠️")
