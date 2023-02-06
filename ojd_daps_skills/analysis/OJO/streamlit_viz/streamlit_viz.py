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
    all_sector_data = {k: v for k, v in all_sector_data.items() if k != "Other"}

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
                "Somewhat similar",
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
                    title="Percentage of job adverts that mention this skill at least once",
                    format="%",
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
            title={
                "text": [
                    f"Most common {skill_group_select_text[skill_group_level]}",
                    f' for "{sector}"',
                ],
                "color": "Black",
            },
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
            title=f'Skill Intensity in "{location}"',
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


# ========================================
# ---------- Streamlit configs ------------

with open("style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
<style>
.big-font {
    font-size:34px !important;
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
    font-size:26px !important;
    font-weight: bold;
}
</style>
""",
    unsafe_allow_html=True,
)


st.sidebar.markdown(
    """
[Introduction](#introduction) \n
[Most Common Skills](#common_skills) \n
[A use case for career advisers: _Enriching career advice_](#occupations) \n
[A use case for local authorities: _regional skill demand_](#regions) \n
[A use case for HR: _Understanding skills in a job advert_](#hr) \n
[Conclusions](#conclusions)
""",
    unsafe_allow_html=True,
)


# ----- Introduction -----

col1, col2 = st.columns([50, 50])
with col1:
    st.image("images/nesta_escoe_transparent.png")

st.header("", anchor="introduction")
with st.expander("Introduction", expanded=True):
    intro_text = """
    Many stakeholders like policy makers, local authorities and career advisers don’t have access to information on the skill demand landscape because there is no publicly available data on the skills requested in UK job adverts. This data gap means that stakeholders have a less complete evidence base to inform labour market policy efforts, address regional skill shortages or suggest occupation transitions.

    To address this, Nesta has released the [Open Jobs Observatory (OJO)](https://www.nesta.org.uk/data-visualisation-and-interactive/open-jobs-observatory/), a project to:

    1. *collect* millions of online job adverts and;
    2. *develop* a suite of algorithms to extract insights from the text.

    We have recently released an [open-source skills extraction python library](https://nestauk.github.io/ojd_daps_skills/build/html/about.html) - to learn more about it, you can read our previous blog (link pending).

    Both our suite of algorithms and ever growing database of online job adverts allow us to drill down on occupational and regional skill demand. To illustrate the types of analysis made possible by addressing this data gap, we have created a number of example interactive visualisations. All the examples are based on a static sample of 100,000 job adverts that were posted online between January 2021 and August 2022. All the skills extracted were matched to the **[European Commission’s European Skills, Competences, and Occupations (ESCO)](https://esco.ec.europa.eu/en/about-esco/what-esco) skills taxonomy**.

    This blog is intended as a demonstration only, to showcase what is possible with rich data and extracted insights. In the future, we hope to be able to build a real-time tool that relies on the whole database to more completely capture the UK’s skills demand landscape.

    """

    st.markdown(intro_text)

# ----- Summary -------

st.header("", anchor="common_skills")
with st.expander("Most Common Skills"):

    sum_text = """ We are able to get a sense of the most common skills across regions and occupations. At the highest level, the most commonly occurring ESCO skill in the sample of job adverts is "communication" which is mentioned in *20%* of job adverts.  "Show positive attitude" features in *14%* of adverts and "show organisational abilities" appears in *10%*.

    To get a more granular sense of the most common i.e. information or transversal skills, you can use the interactive visualisation below to interrogate the most commonly occuring skills in the skills taxonomy.

    """

    st.markdown(sum_text)

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

st.header("", anchor="occupations")
with st.expander("A use case for career advisers: _Enriching career advice_"):

    st.markdown(
        "<p class='medium-font'>Providing new insights on a single occupation</p>",
        unsafe_allow_html=True,
    )

    occ_text = """ For a selected occupation, the visualisations below show the most commonly requested skills (and skill groups) for that role as well as the ten occupations that require the most similar skills.

    Only occupations with over 500 job adverts have been included. This type of information could be used to enrich careers advice by providing new insights on the skills required for an occupation. It could also be used to broaden the search horizons of job seekers by providing names of occupations that require similar skills which they may have previously not considered.

    """

    st.markdown(occ_text)

    top_sectors = [k for k, v in all_sector_data.items() if v["num_ads"] > 500]

    sector = st.selectbox("Select an occupation", top_sectors)

    metric1, metric2 = st.columns((1, 1))
    metric1.metric(
        label="*Number of job adverts for this occupation*",
        value=all_sector_data[sector]["num_ads"],
    )
    metric2.metric(
        label="*Percentage of all job adverts*",
        value=f"{percentage_job_adverts_per_sector[sector]}%",
    )

    ## ----- Similar sectors [selections: sector] -----

    similar_sectors_text_chart = create_similar_sectors_text_chart(
        all_sector_data, sector
    )

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
    remove_trans = st.checkbox("Remove transversal skills", key="1")

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
        "<p class='medium-font'>Providing a map of all occupations</p>",
        unsafe_allow_html=True,
    )

    occ_map_text = """ We have developed a map of occupations where connections between them are made when they demand similar skills.

    The map shows both expected and unexpected skill relationships between occupations. For example, and unsurprisingly, the skills demanded for different teaching occupations such as ‘Adult Educator’ and ‘Teaching Assistant’ are similar. However, and somewhat unexpectedly, skills required of solicitors/lawyers are similar to both other legal and engineering occupations. This could be due to the analytical nature of both roles.

    This map could be used to help guide transitions from one occupation to another. For example, it could help job seekers leave occupations at risk of automation or encourage them into green occupations. Although skill overlap is only one aspect of recommending occupation transitions, it is still an essential consideration.

    """

    st.markdown(occ_map_text)
    # sim_thresh = st.slider('Similarity threshold', 0.4, 1.0, value=0.5, step=0.1)
    sim_thresh = 0.4  # lower than this is either a big clump (0.3-0.4) and/or crashes things (<0.3)
    high_sector_similarity = sector_similarity[
        (
            (sector_similarity["weight"] > sim_thresh)
            & (sector_similarity["target"] != "Other")
            & (sector_similarity["source"] != "Other")
        )
    ]

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

# st.markdown(
#     "<p class='big-font'>A use case for local authorities: <i>regional skill demand</i></p>",
#     unsafe_allow_html=True,
# )
st.header("", anchor="regions")
with st.expander("A use case for local authorities: _regional skill demand_"):

    local_gov_text = """ In addition to providing insights at a national level, job adverts can also give a sense of the skills landscape at a regional level.

    The visualisation below shows the breakdown of vacancies across regions. For example, the tool shows that a large proportion of vacancies can be found in London, Surrey and Berkshire, Buckinghamshire and Oxfordshire.

    """

    st.markdown(local_gov_text)

    geo = st.selectbox("Select Region", regions_list)

    st.markdown(
        "<p class='medium-font'>Vacancies per Region</p>", unsafe_allow_html=True
    )

    metric1, metric2 = st.columns((1, 1))
    metric1.metric(
        label="*Number of job adverts*", value=all_region_data[geo]["num_ads"]
    )
    metric2.metric(
        label="*Percentage of all job adverts*",
        value=f"{round((all_region_data[geo]['num_ads']/100000)*100,2)}%",
    )

    ## ----- The most common skills [selections: skill level] -----

    st.markdown("<p class='medium-font'>Skills per Region</p>", unsafe_allow_html=True)

    local_gov_text_top_skills = """The visualisation below shows the most common skills (and skill groups) requested in job adverts for a chosen region. By taking advantage of the ESCO multi-level structure, we can provide a sense of both the broad and narrow skill mixes within the chosen region.

    """

    st.markdown(local_gov_text_top_skills)

    skill_group_level = st.selectbox(
        "Select skill or skill group level", list(selection_mapper.keys())
    )

    remove_trans = st.checkbox("Remove transversal skills", key="2")

    skill_group_level = selection_mapper[skill_group_level]

    common_skills_chart = create_common_skills_chart(
        all_region_data, skill_group_level, geo, remove_trans=remove_trans
    )

    st.altair_chart(
        common_skills_chart.configure_axis(labelLimit=300),
        use_container_width=True,
    )

    ## ----- Skill specialisms [selections: location] -----

    st.markdown(
        "<p class='medium-font'>Regional Skill Intensity</p>", unsafe_allow_html=True
    )

    loc_text_intensity = """ Finally, we can find out about a region’s skill ‘intensity’ by identifying those skill groups that are requested relatively more or less frequently in a region than across the rest of the country. This metric is called a location quotient and is calculated by dividing the % of job vacancies requiring a skill group in that region by the same percentage for the whole of the UK. Scores above 1 indicate that the region may have relatively greater demand for that skill group while scores below 1 suggest the opposite.

    This information might be useful for local authorities who are seeking to understand the skills that are in relatively greater demand within their region, to in turn inform decisions around local skills provision. Alternatively, local authorities could use information on regionally underrepresented skill groups to inform decisions on skill training programmes.

    """

    st.markdown(loc_text_intensity)

    location_quotident_chart = create_location_quotident_graph(loc_quotident_data, geo)
    st.altair_chart(
        location_quotident_chart.configure_axis(labelLimit=300),
        use_container_width=True,
    )

# ========================================
# ----- Career Advice Personnel Use Case -----

st.header("", anchor="hr")
with st.expander("A use case for HR: _Understanding skills in a job advert_"):

    hr_text = """
        We have also developed a beta app that uses our algorithm to extract skills from a single job advert supplied by a user. This could be useful for HR professionals to quickly identify the skills that they are requesting within an otherwise densely worded job advert.

        The user simply pastes their advert into the empty box, shown below. The algorithm then extracts the skill-related terms within the text and seeks to find their closest matches within an official list (or taxonomy) or skills.

        The user can choose from one of two skill taxonomies:

        1. The [European Commission's Skills Taxonomy](https://esco.ec.europa.eu/en/classification/skill_main), a multilingual classification of European Skills, Competences, Qualifications and Occupations and;
        2. [Lightcast's Open Skills Taxonomy](https://skills.lightcast.io/) (as of 22/11/22) which is an open source library of 32,000+ skills.

    """
    st.markdown(hr_text)

    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        st.write("")

    with col2:
        st.image("images/scene_1.gif")
        st.caption(
            '<p style="text-align: center;">Demo app in action.</p>',
            unsafe_allow_html=True,
        )

    with col3:
        st.write("")

    tax_text = """ For example, the text above is from a job advert for a Commercial Finance Manager. When the user clicks ‘Extract Skills’, the algorithm outputs two types of skills:
    the first are the raw skill spans as exactly worded from the job advert. The second are taxonomy skills based on how closely the raw skill spans map to them.

    As the algorithm supports multiple taxonomies, a user can compare the nature and types of skills present in the different taxonomies. For example, when the algorithm is applied to a software engineer role, the lightcast taxonomy skills are more specific to programming languages than ESCO. This suggests that there could be more technical skills included in the lightcast taxonomy.

    """
    st.markdown(tax_text)

    col1, col2 = st.columns([50, 50])
    with col1:
        st.header("*_ESCO_ Skills*")
        st.image("images/esco_extracted_skills_engineer.png")
    with col2:
        st.header("*_Lightcast_ Skills*")
        st.image("images/lightcast_extracted_skills_engineer.png")

    st.caption(
        '<p style="text-align: center;">Extracted skills from a software developer role.</p>',
        unsafe_allow_html=True,
    )

    conc_text = """
    Ultimately, the tool could help HR professionals to check the skills that they are requesting within an advert and to standardise the language that they use. However, the skills extracted should not be used for any discriminatory hiring purposes.

    """
    st.markdown(conc_text)

# ========================================
# ----- Conclusions -----

st.header("", anchor="conclusions")
with st.expander("Conclusions"):

    conclusion_text = """
    Online job adverts, and the skills mentioned within, have the potential to help a number of groups, ranging from government bodies to HR professionals to make better labour market decisions. The example analysis is based on a static sample of job adverts but they have the potential to be turned into real-time tools for the most up-to-date view on the UK skills landscape.

    If you are interested in the example analysis, our [Skills Extractor library](https://nestauk.github.io/ojd_daps_skills/build/html/index.html) or the [Open Jobs Observatory](https://www.nesta.org.uk/data-visualisation-and-interactive/open-jobs-observatory/), do get in contact with us at dataanalytics@nesta.org.uk.

    """
    st.markdown(conclusion_text)
