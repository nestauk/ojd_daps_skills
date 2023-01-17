import streamlit as st
from ojd_daps_skills.pipeline.extract_skills.extract_skills import ExtractSkills
import os

image_dir = "nesta_escoe_skills.png"
st.image(image_dir)

with open("style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
This demo app is using Nesta's [Skills Extractor Library](https://github.com/nestauk/ojd_daps_skills)
to extract skills for a given job advert and map them onto a skills taxonomy of
your choice. Multiple organisations, from private corporations to government bodies, have developed skills taxonomies to organise labour market skills in a structured way.
By mapping extracted skills to a pre-defined taxonomy, you are able to take advantage of the additional benefits of a taxonomy, including its structure and skill definitions.
This library was made possible via funding from the [Economic Statistics Centre of Excellence](https://esco.ec.europa.eu/en/classification/skill_main).
We currently support three taxonomies out-of-the-box:
1. The [European Commission's Skills Taxonomy](https://esco.ec.europa.eu/en/classification/skill_main), a multilingual classification of European Skills, Competences, Qualifications and Occupations;
2. [Lightcast's Open Skills Taxonomy](https://skills.lightcast.io/) and;
3. A [Toy Taxonomy](https://github.com/nestauk/ojd_daps_skills/blob/dev/ojd_daps_skills/config/extract_skills_toy.yaml) that is helpful for testing.
"""
)

st.warning(
    "As with any algorithm, our approach has limitations. As a result, we cannot guarantee the accuracy or completeness of every extracted or mapped skill.",
    icon="🤖",
)

test_tax = "Toy"
esco_tax = "ESCO"
lightcast_tax = "Lightcast"
app_mode = st.selectbox(
    "Choose a taxonomy to map onto 🗺️", [esco_tax, lightcast_tax, test_tax]
)
txt = st.text_area("Add your job advert text here ✨", "")

if app_mode == esco_tax:
    es = ExtractSkills(config_name="extract_skills_esco", local=True)
elif app_mode == test_tax:
    es = ExtractSkills(config_name="extract_skills_toy", local=True)
elif app_mode == lightcast_tax:
    es = ExtractSkills(config_name="extract_skills_lightcast", local=True)

m = st.markdown(
    """
<style>
div.stButton > button:first-child {
    background-color: #ffcccb;
    color:#ffcccb;
}
</style>""",
    unsafe_allow_html=True,
)

button = st.button("extract skills")

if button:
    with st.spinner("🤖 Loading algorithms - this may take some time..."):
        es.load()
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
