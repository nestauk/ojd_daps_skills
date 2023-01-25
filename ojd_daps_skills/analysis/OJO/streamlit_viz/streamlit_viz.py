import os

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from streamlit_agraph import agraph, Node, Edge, Config
from colour import Color

from streamlit_viz_utils import *

s3 = get_s3_resource()
s3_folder = "escoe_extension/outputs/data"


def load_summary_data():

    file_name = os.path.join(
        s3_folder,
        "streamlit_viz",
        "per_skill_group_proportions_sample.json",
    )
    top_skills_by_skill_groups = load_s3_data(s3, bucket_name, file_name)

    return top_skills_by_skill_groups


def load_sector_data():

    file_name = os.path.join(
        s3_folder, "streamlit_viz", "per_sector_sample_updated.json"
    )
    all_sector_data = load_s3_data(s3, bucket_name, file_name)

    file_name = os.path.join(
        s3_folder,
        "streamlit_viz",
        "lightweight_skill_similarity_between_sectors_sample.csv",
    )
    sector_similarity = load_s3_data(s3, bucket_name, file_name)

    file_name = os.path.join(s3_folder, "streamlit_viz", "sector_2_kd_sample.json")
    sector_2_kd = load_s3_data(s3, bucket_name, file_name)

    number_job_adverts_per_sector = {
        sector_name: v["num_ads"] for sector_name, v in all_sector_data.items()
    }

    total_num_job_adverts = sum(number_job_adverts_per_sector.values())
    percentage_job_adverts_per_sector = {
        sector_name: round(num_ads * 100 / total_num_job_adverts, 2)
        for sector_name, num_ads in number_job_adverts_per_sector.items()
    }

    return (
        all_sector_data,
        percentage_job_adverts_per_sector,
        sector_similarity,
        sector_2_kd,
    )


def load_regional_data():

    file_name = os.path.join(
        s3_folder, "streamlit_viz", "top_skills_per_loc_sample.json"
    )
    all_region_data = load_s3_data(s3, bucket_name, file_name)

    file_name = os.path.join(
        s3_folder,
        "streamlit_viz",
        "top_skills_per_loc_quotident_sample.csv",
    )

    loc_quotident_data = load_s3_data(s3, bucket_name, file_name)

    return (
        all_region_data,
        loc_quotident_data,
    )


def create_sector_skill_sim_network(
    high_sector_similarity, sector_2_kd, percentage_job_adverts_per_sector
):
    # Node size is scaled by the percentage of job ads with this skill
    min_node_size = 5
    max_node_size = 10

    # Create colour mapper for sectors to be coloured by their parent knowledge domain (broad occupational group).
    # If you run out of Nesta colours, then reloop through them
    color_i = 0
    knowledge_domain_colors = {}
    for knowledge_domain in set(sector_2_kd.values()):
        if color_i > len(NESTA_COLOURS):
            color_i = 0
        knowledge_domain_colors[knowledge_domain] = NESTA_COLOURS[color_i]
        color_i += 1

    nodes = []
    edges = []
    node_ids = set()
    for _, connection in high_sector_similarity.iterrows():
        target_skill = connection["target"]
        source_skill = connection["source"]

        if target_skill not in node_ids:
            nodes.append(
                Node(
                    id=target_skill,
                    label=target_skill,
                    color=knowledge_domain_colors[sector_2_kd[target_skill]],
                    size=percentage_job_adverts_per_sector[connection["target"]]
                    * (max_node_size - min_node_size)
                    + min_node_size,
                )
            )
            node_ids.add(target_skill)
        if source_skill not in node_ids:
            nodes.append(
                Node(
                    id=source_skill,
                    label=source_skill,
                    color=knowledge_domain_colors[sector_2_kd[source_skill]],
                    size=percentage_job_adverts_per_sector[connection["source"]]
                    * (max_node_size - min_node_size)
                    + min_node_size,
                )
            )
            node_ids.add(source_skill)
        edges.append(
            Edge(
                source=source_skill,
                target=target_skill,
                color="#0F294A",
                weight=connection["weight"],
                directed=False,
                arrows={
                    "to": {"scaleFactor": 0}
                },  # Hack to make the graph undirected - make arrows invisible!
            )
        )

    config = Config(
        width=1000,
        height=500,
        directed=False,
        nodeHighlightBehavior=True,
        collapsible=True,
    )

    # Legend (is actually an altair plot)
    legend_df = pd.DataFrame(
        {
            "x": [
                i
                for i, v in enumerate(
                    np.array_split(list(knowledge_domain_colors.keys()), 3)
                )
                for ii, vv in enumerate(v)
            ],
            "y": [
                ii
                for i, v in enumerate(
                    np.array_split(list(knowledge_domain_colors.keys()), 3)
                )
                for ii, vv in enumerate(v)
            ],
            "value": list(knowledge_domain_colors.keys()),
            "color": list(knowledge_domain_colors.values()),
        }
    )

    legend_chart = (
        alt.Chart(legend_df, title="Broad occupational groups")
        .mark_circle(size=150)
        .encode(
            x=alt.X("x", axis=alt.Axis(labels=False, grid=False), title=""),
            y=alt.Y("y", axis=alt.Axis(labels=False, grid=False), title=""),
            color=alt.Color(
                "value",
                scale=alt.Scale(
                    domain=list(knowledge_domain_colors.keys()),
                    range=list(knowledge_domain_colors.values()),
                ),
                legend=None,
            ),
        )
        .properties(height=200)
    )

    legend_text = (
        alt.Chart(legend_df)
        .mark_text(
            align="left",
            baseline="middle",
            fontSize=12,
            color="black",
            dx=10,
            font="Averta Demo",
        )
        .encode(x="x", y="y", text="value")
    )

    legend_chart = legend_chart + legend_text

    configure_plots(legend_chart)

    return nodes, edges, config, legend_chart.configure_title(fontSize=24)


def create_similar_sectors_text_chart(all_sector_data, sector):

    similar_sectors = pd.DataFrame.from_dict(
        all_sector_data[sector]["similar_sectors"],
        orient="index",
        columns=["euclid_dist"],
    )
    similar_sectors.drop(index="Other", inplace=True)
    similar_sectors.sort_values(
        by="euclid_dist", inplace=True, ascending=True
    )  # Smaller Euclid dist is closer
    similar_sectors = similar_sectors[0:10]
    similar_sectors["sector"] = similar_sectors.index
    similar_sectors["Similarity score"] = 1 / (
        similar_sectors["euclid_dist"] + 0.0001
    )  # Just so a value of 1 means most similar, and 0 is least

    most_similar_color = Color("green")
    least_similar_color = Color("red")
    similarity_colors = {
        sim_value / 10: str(c.hex)
        for sim_value, c in enumerate(
            list(most_similar_color.range_to(least_similar_color, 10))
        )
    }

    similar_sectors_text = pd.DataFrame(
        {
            "x": [0] * 5 + [1] * 5,
            "y": list(range(5, 0, -1)) + list(range(5, 0, -1)),
            "value": [f"{num+1}. {similar_sectors.index[num]}" for num in range(10)],
            "color": [
                np.floor(euclid_dist * 10) / 10
                for euclid_dist in similar_sectors[0:10]["euclid_dist"].tolist()
            ],
            "sim_score": similar_sectors[0:10]["euclid_dist"].tolist(),
        }
    )

    circle_chart = (
        alt.Chart(similar_sectors_text, title="Most similar occupations")
        .mark_circle(size=100)
        .encode(
            x=alt.X("x", axis=alt.Axis(labels=False, grid=False), title=""),
            y=alt.Y("y", axis=alt.Axis(labels=False, grid=False), title=""),
            text="value",
            tooltip=[alt.Tooltip("sim_score", title="Similarity score", format=".2")],
            color=alt.Color(
                "color",
                scale=alt.Scale(
                    domain=list(similarity_colors.keys()),
                    range=list(similarity_colors.values()),
                ),
                legend=None,
            ),
        )
        .properties(height=200, width=300)
    )

    text_chart = (
        alt.Chart(similar_sectors_text, title="Most similar occupations")
        .mark_text(align="left", baseline="middle", fontSize=16, dx=10, color="black")
        .encode(
            x=alt.X("x", axis=alt.Axis(labels=False, grid=False), title=""),
            y=alt.Y("y", axis=alt.Axis(labels=False, grid=False), title=""),
            text="value",
            tooltip=[alt.Tooltip("sim_score", title="Similarity score", format=".2")],
        )
        .properties(height=200, width=300)
    )

    similar_sectors_colors = pd.DataFrame(
        {
            "x": [0, 0, 0, 0],
            "y": [0, 0, 0, 0],
            "color": ["#008000", "#72aa00", "#d58e00", "#f00"],
            "sim_type": [
                "Very similar",
                "Quite similar",
                "Mid similarity",
                "Not similar",
            ],
        }
    )

    legend_chart = (
        alt.Chart(similar_sectors_colors)
        .mark_circle(size=0)
        .encode(
            x=alt.X("x", axis=alt.Axis(labels=False, grid=False), title=""),
            y=alt.Y("y", axis=alt.Axis(labels=False, grid=False), title=""),
            color=alt.Color(
                "sim_type",
                scale=alt.Scale(
                    domain=list(
                        dict(
                            zip(
                                similar_sectors_colors["sim_type"],
                                similar_sectors_colors["color"],
                            )
                        ).keys()
                    ),
                    range=list(
                        dict(
                            zip(
                                similar_sectors_colors["sim_type"],
                                similar_sectors_colors["color"],
                            )
                        ).values()
                    ),
                ),
                legend=alt.Legend(title=""),
            ),
        )
        .properties(height=200, width=10)
    )

    base = alt.hconcat(circle_chart + text_chart, legend_chart)

    configure_plots(base)

    return base.configure_title(fontSize=24)


def create_common_skills_chart_by_skill_groups(top_skills_by_skill_groups, skill_group):
    plot_title = f"Most common skills in {skill_group} skill group"
    if skill_group == "all":
        plot_title += "s"

    top_skills = pd.DataFrame.from_dict(
        top_skills_by_skill_groups[skill_group],
        orient="index",
        columns=["percent"],
    )
    top_skills.sort_values(by="percent", inplace=True, ascending=False)
    top_skills = top_skills[0:10]
    top_skills["skill"] = top_skills.index

    common_skills_chart = (
        alt.Chart(top_skills)
        .mark_bar(size=10, opacity=0.8, color="#0000FF")
        .encode(
            y=alt.Y("skill", sort=None, axis=alt.Axis(title=None)),
            x=alt.X(
                "percent:Q",
                axis=alt.Axis(
                    title="Percentage of job adverts with this skill", format="%"
                ),
            ),
            tooltip=[alt.Tooltip("percent", title="Percentage", format=".1%")],
        )
        .properties(
            title=plot_title,
            # height=100,
            width=75,
        )
    )

    configure_plots(common_skills_chart)

    return common_skills_chart.configure_title(fontSize=24)


def create_common_skills_chart(
    all_sector_data, skill_group_level, sector, remove_trans=False
):

    skill_group_select_text = {
        "all": "skills or skill groups",
        "0": "skill groups",
        "1": "skill groups",
        "2": "skill groups",
        "3": "skill groups",
        "4": "skill",
    }

    if remove_trans:
        key_name = "top_skills_no_transversal"
    else:
        key_name = "top_skills"

    top_skills = pd.DataFrame.from_dict(
        all_sector_data[sector][key_name][skill_group_level],
        orient="index",
        columns=["percent"],
    )
    top_skills.sort_values(by="percent", inplace=True, ascending=False)
    top_skills = top_skills[0:10]
    top_skills["sector"] = top_skills.index

    common_skills_chart = (
        alt.Chart(top_skills)
        .mark_bar(size=10, opacity=0.8, color="#0000FF")
        .encode(
            y=alt.Y("sector", sort=None, axis=alt.Axis(title=None)),
            x=alt.X(
                "percent:Q",
                axis=alt.Axis(
                    title="Percentage of job adverts with this skill", format="%"
                ),
            ),
            tooltip=[alt.Tooltip("percent", title="Percentage", format=".1%")],
        )
        .properties(
            title=f"Most common {skill_group_select_text[skill_group_level]}",
            # height=100,
            width=75,
        )
    )

    configure_plots(common_skills_chart)

    return common_skills_chart.configure_title(fontSize=24)


def create_location_quotident_graph(all_location_data, location):

    geo_df = all_location_data[
        (all_location_data["region"] == location)
        & (all_location_data["skill_percent"] >= 0.05)
    ].sort_values("absolute_location_change", ascending=False)[:15]
    geo_df["skill_percent"] = round(geo_df["skill_percent"] * 100, 2)

    base = (
        alt.Chart(geo_df)
        .mark_point(size=10, opacity=0.8, color="#0000FF", filled=True)
        .encode(
            y=alt.Y("skill", sort="-x", axis=alt.Axis(title=None, labelLimit=1000)),
            x=alt.X(
                "location_quotident",
                axis=alt.Axis(title="Location Quotident"),
            ),
            size=alt.Size(
                "skill_percent:Q",
                title=["Percentage of job adverts", "with this skill group (%)"],
            ),
            color=alt.Color(
                "location_change",
                scale=alt.Scale(domainMid=0, scheme="redblue"),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip(
                    "skill_percent:Q",
                    title="% of job adverts with this skill group",
                    format=",.2f",
                ),
                alt.Tooltip(
                    "location_change",
                    title="Location Quotident Change",
                    format=",.2f",
                ),
            ],
        )
        .properties(
            title=f"Skill Intensity in {location}",
        )
    )

    vline = (
        alt.Chart(pd.DataFrame({"location_quotident": [1], "color": ["red"]}))
        .mark_rule(opacity=0.8)
        .encode(x="location_quotident", color=alt.Color("color:N", scale=None))
    )

    base_line = base + vline
    configure_plots(base_line)

    return base_line.configure_title(fontSize=24)


def create_location_pie_chart(all_region_data, region):
    source = pd.DataFrame(
        {
            "region": list(all_region_data.keys()),
            "value": [
                skill_info["num_ads"] for region, skill_info in all_region_data.items()
            ],
        }
    )
    source["percent"] = source["value"] / source.value.sum() * 100

    pie = (
        alt.Chart(source)
        .mark_arc(innerRadius=50)
        .encode(
            theta=alt.Theta(field="value", type="quantitative"),
            color=alt.Color(field="region", type="nominal", legend=None),
            tooltip=[
                alt.Tooltip("region", title="Region"),
                alt.Tooltip("percent", title="% of Vacancies"),
            ],
            opacity=alt.condition(
                alt.datum.region == region, alt.value(1), alt.value(0.1)
            ),
        )
        .properties(
            title="% of Vacancies by Region",
        )
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
    )

    configure_plots(pie)

    return pie.configure_title(fontSize=24)


# ========================================
# ---------- Streamlit configs ------------

with open("style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
<style>
.big-font {
    font-size:42px !important;
    font-weight: bold;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
.medium-font {
    font-size:36px !important;
    font-weight: bold;
}
</style>
""",
    unsafe_allow_html=True,
)

# ----- Introduction -----

col1, col2 = st.columns([50, 50])
with col1:
    st.image("images/nesta_escoe_transparent.png")
st.markdown("<p class='big-font'>Introduction</p>", unsafe_allow_html=True)

intro_text = """
Nesta’s [Open Jobs Observatory (OJO)](https://www.nesta.org.uk/data-visualisation-and-interactive/open-jobs-observatory/) provides free and up-to-date information on UK skills demands by collecting thousands of online job adverts. A set of algorithms, including our open source [Skills Extractor library](https://nestauk.github.io/ojd_daps_skills/build/html/index.html), were developed in order to make the most of the data scraped from job advertisements websites as part of OJO.

Both our suite of algorithms and ever growing database of online job adverts allow us to get a granular sense of the skills demanded in the UK. These insights are valuable to many stakeholders, including the national government, local councils and HR professionals.

For example, the national government can identify growing and declining UK occupations and skills to inform economic growth or skill policy agendas.

Similarly, local councils can better understand their regional skill demand profile and use these insights to allocate resources to appropriate professional training bodies.

Finally, HR professionals and career advice personnel can use our [skills extractor app](http://18.169.52.145:8501/) to quickly identify the skills required of a job and the most qualified individuals for it.
"""

st.markdown(intro_text)

method_text = """
We’ve developed a number of example analyses to demonstrate the value of both OJO and our suite of algorithms. We’ve conducted this analysis on a sample of 100,000 job adverts posted online between January 2021 and August 2022. The analyses are organised by relevance to the identified stakeholders.
"""

st.markdown(method_text)

# ----- Summary -------

top_skills_by_skill_groups = load_summary_data()

skill_group = st.selectbox(
    "Select skill group", sorted(list(top_skills_by_skill_groups.keys())), index=1
)

common_skills_chart_by_skill_groups = create_common_skills_chart_by_skill_groups(
    top_skills_by_skill_groups, skill_group
)
st.altair_chart(
    common_skills_chart_by_skill_groups.configure_axis(labelLimit=500),
    use_container_width=True,
)

# ----- National Government Use Case -----

(
    all_sector_data,
    percentage_job_adverts_per_sector,
    sector_similarity,
    sector_2_kd,
) = load_sector_data()


st.markdown(
    "<p class='big-font'>National Government Use Case</p>", unsafe_allow_html=True
)

st.markdown("<p class='medium-font'>Skills per occupation</p>", unsafe_allow_html=True)


st.markdown(
    "For a selected occupation you can see the most similar occupations (based on the skills asked for) and the most common skills or skill groups. Only occupations with over 500 job adverts are included."
)

top_sectors = [k for k, v in all_sector_data.items() if v["num_ads"] > 500]

sector = st.selectbox("Select occupation", top_sectors)

metric1, metric2 = st.columns((1, 1))
metric1.metric(
    label="**Number of job adverts**", value=all_sector_data[sector]["num_ads"]
)
metric2.metric(
    label="**Percentage of all job adverts**",
    value=f"{percentage_job_adverts_per_sector[sector]}%",
)

## ----- Similar sectors [selections: sector] -----

similar_sectors_text_chart = create_similar_sectors_text_chart(all_sector_data, sector)

st.altair_chart(similar_sectors_text_chart, use_container_width=True)

## ----- The most common skills [selections: sector] -----

selection_mapper = {
    "Any (closest skill or skill group)": "all",
    'Most broad (e.g. "S")': "0",
    'Broad/mid (e.g. "S1")': "1",
    'Mid/granular (e.g. "S1.2")': "2",
    'Most granular (e.g. "S1.2.3")': "3",
    "Skill": "4",
}

skill_group_level = st.selectbox(
    "Select skill group level", list(selection_mapper.keys())
)
remove_trans = st.checkbox("Remove transversal skills")

skill_group_level = selection_mapper[skill_group_level]

common_skills_chart = create_common_skills_chart(
    all_sector_data, skill_group_level, sector, remove_trans=remove_trans
)

st.altair_chart(
    common_skills_chart.configure_axis(labelLimit=500),
    use_container_width=True,
)

## ----- Skill similarities network [selections: none] -----

st.markdown(
    "<p class='medium-font'>Skill similarities between occupations</p>",
    unsafe_allow_html=True,
)

st.markdown(
    "Connections between occupations are made when the similarity between the two occupations is over a threshold. Node size shows number of job adverts from this occupation."
)
# sim_thresh = st.slider('Similarity threshold', 0.4, 1.0, value=0.5, step=0.1)
sim_thresh = (
    0.4  # lower than this is either a big clump (0.3-0.4) and/or crashes things (<0.3)
)
high_sector_similarity = sector_similarity[sector_similarity["weight"] > sim_thresh]

nodes, edges, config, legend_chart = create_sector_skill_sim_network(
    high_sector_similarity, sector_2_kd, percentage_job_adverts_per_sector
)

agraph(nodes, edges, config)

st.altair_chart(legend_chart, use_container_width=True)

# ========================================
# ----- Local Government Use Case -----
(
    all_region_data,
    loc_quotident_data,
) = load_regional_data()

regions_list = list(all_region_data.keys())

st.markdown(
    "<p class='big-font'>Use Case for Local Governments</p>", unsafe_allow_html=True
)

geo = st.selectbox("Select Region", regions_list)

local_gov_text = """
    In addition to national level insights, we are also able to get a local sense of the skills landscape using our algorithms and database.

    For example, we can get a breakdown of vacancies by statistical regions of the UK. This provides insights into where jobs in the UK are concentrated.
"""

st.markdown(local_gov_text)

st.markdown("<p class='medium-font'>Vacancies per Region</p>", unsafe_allow_html=True)

metric1, metric2 = st.columns((1, 1))
metric1.metric(label="**Number of job adverts**", value=all_region_data[geo]["num_ads"])
metric2.metric(
    label="**Percentage of all job adverts**",
    value=f"{round((all_region_data[geo]['num_ads']/100000)*100,2)}%",
)

locations_pie_chart = create_location_pie_chart(all_region_data, geo)
st.altair_chart(
    locations_pie_chart,
    use_container_width=True,
)

## ----- The most common skills [selections: skill level] -----

st.markdown("<p class='medium-font'>Skills per Region</p>", unsafe_allow_html=True)

local_gov_text = """In addition to vacancies by region, we can also interrogate the types of skills commonly requested by region. By taking advantage of the ESCO taxonomy's structure, we can get a high level sense of
skills demanded at any level of the taxonomy, ranging from very high level skill groups right down to the individual skill.
"""

st.markdown(local_gov_text)

skill_group_level = st.selectbox(
    "Select skill or skill group level", list(selection_mapper.keys())
)

skill_group_level = selection_mapper[skill_group_level]

common_skills_chart = create_common_skills_chart(
    all_region_data, skill_group_level, geo
)

st.altair_chart(
    common_skills_chart.configure_axis(labelLimit=300),
    use_container_width=True,
)

## ----- Skill specialisms [selections: location] -----

st.markdown(
    "<p class='medium-font'>Regional Skill Intensity</p>", unsafe_allow_html=True
)

loc_text = """Finally, we can also calculate regional skill intensities by calculating the location quotident between regional skills requested and overall skills requested. We calculate the location quotident by dividing the % of job vacancies requiring a skill group in a given area by the whole of the UK. Regions specialise in skill groups with Location Quotident Scores above 1 while skill groups with scores below 1 are underrepresented regionally.

Local governments can take advantage of this analysis by identifying regional skill gaps or by attracting businesses in specific industries based on regional skill specialisms.
"""

st.markdown(loc_text)

location_quotident_chart = create_location_quotident_graph(loc_quotident_data, geo)
st.altair_chart(
    location_quotident_chart.configure_axis(labelLimit=300),
    use_container_width=True,
)

# ========================================
# ----- Career Advice Personnel Use Case -----

st.markdown(
    "<p class='big-font'>Use Case for Career Advice Personnel</p>",
    unsafe_allow_html=True,
)

hr_text = """
    In addition to the open-source library, we have also developed an app for HR professionals and career advice personnel to quickly identify the skills required of a job and the most qualified individuals for it.

    The app works by simply **copying and pasting text** for a given job advert and **clicking 'extract skills'** to get a list of raw and mapped skills for a given taxonomy.

    We currently support three taxonomies out-of-the-box:

    1. The [European Commission's Skills Taxonomy](https://esco.ec.europa.eu/en/classification/skill_main), a multilingual classification of European Skills, Competences, Qualifications and Occupations;
    2. [Lightcast's Open Skills Taxonomy](https://skills.lightcast.io/) and;
    3. A [Toy Taxonomy](https://github.com/nestauk/ojd_daps_skills/blob/dev/ojd_daps_skills/config/extract_skills_toy.yaml) that is helpful for testing;
"""
st.markdown(hr_text)

col1, col2, col3 = st.columns([1, 6, 1])

with col1:
    st.write("")

with col2:
    st.image("images/scene_1.gif")

with col3:
    st.write("")


col1, col2 = st.columns([50, 50])
with col1:
    st.header("**_ESCO_ Extracted Skills**")
    st.image("images/esco_extracted_skills.png")
with col2:
    st.header("**_Lightcast_ Extracted Skills**")
    st.image("images/lightcast_extracted_skills.png")

tax_text = """The job text in the example above is for a Commercial Finance Manager. The app allows you to pick a taxonomy to map onto before extracting skills. You are able to easily change the taxonomy to map onto and see how the taxonomy skills differ."""

st.markdown(tax_text)

conc_text = """Ultimately, the tool helps HR professionals by saving time on searching for skills in lengthy job postings and provides an official 'skills' language by mapping skills to known taxonomies."""

st.markdown(conc_text)

# ========================================
# ----- Conclusions -----

st.markdown("<p class='big-font'>Conclusions</p>", unsafe_allow_html=True)

conclusion_text = """
    Both our ever growing database of online job adverts and our suite of algorithms can ultimately help stakeholders ranging from government bodies to HR professionals make better labour market decisions.

    If you are interested in the analysis, our [Skills Extractor library](https://nestauk.github.io/ojd_daps_skills/build/html/index.html) or the [Open Jobs Observatory](https://www.nesta.org.uk/data-visualisation-and-interactive/open-jobs-observatory/), do get in contact with
    [Cath Sleeman](mailto:cath.sleeman@nesta.org.uk), [Elizabeth Gallagher](mailto:elizabeth.gallagher@nesta.org.uk) or [India Kerle](mailto:india.kerle@nesta.org.uk).
"""

st.markdown(conclusion_text)
