"""
Run

streamlit run ojd_daps_skills/analysis/OJO/streamlit_viz/per_sector.py


pip install streamlit_agraph
pip install colour
"""
import os

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from streamlit_agraph import agraph, Node, Edge, Config
from colour import Color

from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    load_s3_data,
)
from ojd_daps_skills import bucket_name

from ojd_daps_skills.utils.plotting import NESTA_COLOURS


@st.cache
def load_data():

    s3 = get_s3_resource()

    s3_folder = "escoe_extension/outputs/data"

    file_name = os.path.join(s3_folder, "streamlit_viz", "per_sector_sample.json")
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
        .mark_text(align="left", baseline="middle", fontSize=12, color="black", dx=10)
        .encode(x="x", y="y", text="value")
    )

    legend_chart = legend_chart + legend_text

    return nodes, edges, config, legend_chart


def create_similar_sectors_text_chart(all_sector_data, sector):

    similar_sectors = pd.DataFrame.from_dict(
        all_sector_data[sector]["similar_sectors"],
        orient="index",
        columns=["euclid_dist"],
    )
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

    similar_sectors_text_invisible = pd.DataFrame(
        {
            "x": [0] * 10,
            "y": [0] * 10,
            "value": list(similarity_colors.keys()),
            "color": list(similarity_colors.values()),
        }
    )

    legend_chart = (
        alt.Chart(similar_sectors_text_invisible)
        .mark_circle(size=0)
        .encode(
            x=alt.X("x", axis=alt.Axis(labels=False, grid=False), title=""),
            y=alt.Y("y", axis=alt.Axis(labels=False, grid=False), title=""),
            color=alt.Color(
                "value",
                scale=alt.Scale(
                    domain=list(similarity_colors.keys()),
                    range=list(similarity_colors.values()),
                ),
                # scale=alt.Scale(domain=["most", "least"], range=[0,1]),
                legend=alt.Legend(title=""),
            ),
        )
        .properties(height=200, width=50)
    )

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

    text_chart = (
        alt.Chart(similar_sectors_text, title="Most similar occupations")
        .mark_text(align="left", baseline="middle", fontSize=16, dx=10, color="black")
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
            ),
        )
        .properties(height=200, width=300)
    )

    return alt.hconcat(text_chart, legend_chart)


def create_common_skills_chart(all_sector_data, skill_group_level, sector):

    skill_group_select_text = {
        "all": "skills or skill groups",
        "0": "skill groups",
        "1": "skill groups",
        "2": "skill groups",
        "3": "skill groups",
    }

    top_skills = pd.DataFrame.from_dict(
        all_sector_data[sector]["top_skills"][skill_group_level],
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
                "percent",
                axis=alt.Axis(title="Percentage of job adverts with this skill"),
            ),
            tooltip=[alt.Tooltip("percent", title="Percentage", format=".1%")],
        )
        .properties(
            title=f"Most common {skill_group_select_text[skill_group_level]}",
            # height=100,
            width=75,
        )
    )
    return common_skills_chart


# ========================================
# ----- National Government Use Case -----

(
    all_sector_data,
    percentage_job_adverts_per_sector,
    sector_similarity,
    sector_2_kd,
) = load_data()

st.title("National Government Use Case")

st.subheader("Skills per occupation")

st.markdown(
    "For a selected occupation you can see the most similar occupations (based on the skills asked for) and the most common skills or skill groups."
)

top_sectors = [k for k, v in all_sector_data.items() if v["num_ads"] > 500]

sector = st.selectbox("Select occupation", top_sectors)

metric1, metric2 = st.columns((1, 1))
metric1.metric(label="Number of job adverts", value=all_sector_data[sector]["num_ads"])
metric2.metric(
    label="Percentage of all job adverts",
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
}

skill_group_level = st.selectbox(
    "Select skill group level", list(selection_mapper.keys())
)
skill_group_level = selection_mapper[skill_group_level]

common_skills_chart = create_common_skills_chart(
    all_sector_data, skill_group_level, sector
)

st.altair_chart(
    common_skills_chart.configure_axis(labelLimit=300),
    use_container_width=True,
)

## ----- Skill similarities network [selections: none] -----

st.subheader("Skill similarities between occupations")
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
