# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Occupational skill cooccurences analysis
#

# %%
import os
from datetime import date
from collections import Counter
from itertools import chain, combinations

import pandas as pd
import numpy as np
import altair as alt
import networkx as nx
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances

from ojd_daps_skills.utils.plotting import NESTA_COLOURS, nestafont, configure_plots
from ojd_daps_skills.utils.save_plotting import AltairSaver

from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    get_s3_data_paths,
    load_s3_json,
    load_s3_data,
    save_to_s3,
)
from ojd_daps_skills import bucket_name
from ojd_daps_skills.analysis.OJO.get_skill_occurrences_matrix import (
    get_cooccurence_matrix,
)

s3 = get_s3_resource()

# %%
from ipywidgets import interact
import bokeh.plotting as bpl
from bokeh.io import output_file, show, push_notebook, output_notebook
from bokeh.plotting import (
    figure,
    from_networkx,
    ColumnDataSource,
    output_file,
    show,
    save,
)
from bokeh.layouts import row

from bokeh.models import (
    BoxZoomTool,
    WheelZoomTool,
    HoverTool,
    SaveTool,
    Circle,
    MultiLine,
    Plot,
    Range1d,
    ResetTool,
    Label,
    CategoricalColorMapper,
    ColorBar,
    ColumnDataSource,
    LinearColorMapper,
)


from bokeh.palettes import Turbo256, Spectral, Spectral4, viridis, inferno, Spectral6

from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.transform import linear_cmap

bpl.output_notebook()

# %%
s3_folder = "escoe_extension/outputs/data"

# %%
# Get todays date for the output name prefix
today = date.today().strftime("%d%m%Y")
today

# %% [markdown]
# ## Import the data

# %%
# The skill sample (with metadata)
file_name = os.path.join(
    s3_folder, "model_application_data", "dedupe_analysis_skills_sample_temp_fix.json"
)
skill_sample = load_s3_data(s3, bucket_name, file_name)

# %%
# The skill occurences
file_name = os.path.join(
    s3_folder, "analysis", "job_ad_to_mapped_skills_occurrences_sample_30112022.json"
)
job_id_2_skill_count = load_s3_data(s3, bucket_name, file_name)

# %%
# The esco skill to ix mapper
file_name = os.path.join(
    s3_folder, "analysis", "mapped_skills_index_dict_30112022.json"
)
skill_id_2_ix = load_s3_data(s3, bucket_name, file_name)
skill_id_2_ix = {k: str(v) for k, v in skill_id_2_ix.items()}

# %%
esco_hier_mapper = load_s3_data(
    s3,
    bucket_name,
    "escoe_extension/outputs/data/skill_ner_mapping/esco_hier_mapper.json",
)

# %%
esco_skills = load_s3_data(
    s3,
    bucket_name,
    "escoe_extension/outputs/data/skill_ner_mapping/esco_data_formatted.csv",
)


# %%
esco_code2name = {
    c: b
    for job_skills in skill_sample
    if job_skills.get("SKILL")
    for a, [b, c] in job_skills.get("SKILL", [None, [None, None]])
}

# %% [markdown]
# ## Group by occupations and get average skill occurrences

# %%
skill_sample_df = pd.DataFrame(skill_sample)
sector_2_job_ids = skill_sample_df.groupby("sector")["job_id"].unique()

# %%
sector_2_parent = dict(zip(skill_sample_df["sector"], skill_sample_df["parent_sector"]))
sector_2_kd = dict(zip(skill_sample_df["sector"], skill_sample_df["knowledge_domain"]))

# %%
average_sector_skills = {}
for sector, job_ids in tqdm(sector_2_job_ids.items()):
    total_sector_skills = Counter()
    for job_id in job_ids:
        total_sector_skills += Counter(job_id_2_skill_count[job_id])
    average_sector_skills[sector] = {
        k: v / len(job_ids) for k, v in total_sector_skills.items()
    }

# %%
average_sector_skills_df = get_cooccurence_matrix(
    average_sector_skills, skill_id_2_ix, convert_int=False
)
average_sector_skills_df.head(2)


# %% [markdown]
# ## Create weighted network of sectors skill similarity

# %%
def get_euc_dist(source, target, dists, field_name_2_index):
    return dists[field_name_2_index[str(source)], field_name_2_index[str(target)]]


# %%
def get_edge_list(average_sector_skills_df, dists, field_name_2_index):
    pairs = list(
        combinations(sorted(list(set(average_sector_skills_df.index.tolist()))), 2)
    )
    pairs = [x for x in pairs if len(x) > 0]
    edge_list = pd.DataFrame(pairs, columns=["source", "target"])
    edge_list["weight"] = edge_list.apply(
        lambda x: get_euc_dist(x.source, x.target, dists, field_name_2_index), axis=1
    )
    max_weight_value = edge_list["weight"].max()
    edge_list["weight"] = edge_list["weight"].apply(
        lambda x: 1 / (x + 0.000001)
    )  # max_weight_value- x) # Because a lower euclide is a higher weighting
    return edge_list


# %%
def create_network(edge_list, weight_type="weight"):

    occ_edge_list_weighted = (
        edge_list.groupby(["source", "target"])[weight_type]
        .sum()
        .reset_index(drop=False)
    )
    net = nx.from_pandas_edgelist(occ_edge_list_weighted, edge_attr=True)

    # net=nx.minimum_spanning_tree(net, weight='weight')

    print(len(net.nodes))
    if weight_type == "weight":
        min_weight = 2  # 1.1 for parent sector
    elif weight_type == "weight_exp":
        min_weight = 3000

    high_weight = (
        (s, e) for s, e, w in net.edges(data=True) if (w[weight_type] > min_weight)
    )
    net = net.edge_subgraph(high_weight)
    print(len(net.nodes))

    return net


# %%
def set_node_attributes(net, sector_2_kd, sector_2_parent):

    node_attrs_name = {node_num: node_num for node_num in net.nodes()}
    nx.set_node_attributes(net, node_attrs_name, "sector")

    parent_sector_2_number = {
        name: n_i for n_i, name in enumerate(set(sector_2_parent.values()))
    }
    node_attrs_name = {
        node_num: parent_sector_2_number.get(sector_2_parent.get(node_num))
        for node_num in net.nodes()
    }
    nx.set_node_attributes(net, node_attrs_name, "parent_sector_num")

    node_attrs_name = {
        node_num: sector_2_parent.get(node_num) for node_num in net.nodes()
    }
    nx.set_node_attributes(net, node_attrs_name, "parent_sector")

    kd_2_number = {name: n_i for n_i, name in enumerate(set(sector_2_kd.values()))}
    node_attrs_name = {
        node_num: kd_2_number.get(sector_2_kd.get(node_num)) for node_num in net.nodes()
    }
    nx.set_node_attributes(net, node_attrs_name, "knowledge_domain_num")

    node_attrs_name = {node_num: sector_2_kd.get(node_num) for node_num in net.nodes()}
    nx.set_node_attributes(net, node_attrs_name, "knowledge_domain")

    return net, kd_2_number


# %%
def plot_net(net, color_by_mapper, color_by="knowledge_domain", weight_type="weight"):
    net_plotted = net

    plot = Plot(plot_width=600, plot_height=600)
    plot.title.text = (
        "Skill similarity network for each job, coloured by knowledge domain"
    )

    node_hover_tool = HoverTool(
        tooltips=[
            ("job", f"@sector"),
            #         ('parent_sector', f'@parent_sector'),
            ("knowledge_domain", f"@knowledge_domain"),
        ]
    )
    plot.add_tools(
        node_hover_tool, BoxZoomTool(), ResetTool(), WheelZoomTool(), SaveTool()
    )

    graph_renderer = from_networkx(
        net_plotted,
        nx.spring_layout,
        scale=1,
        center=(0, 0),
    )

    graph_renderer.node_renderer.glyph = Circle(
        size=6,
        fill_color=linear_cmap(f"{color_by}_num", "Turbo256", 0, len(color_by_mapper)),
        line_color=None,
    )

    graph_renderer.edge_renderer.glyph = MultiLine(
        line_color="black",
        line_alpha=0.3,
        line_width=0.1,
    )
    if weight_type == "weight":
        graph_renderer.edge_renderer.data_source.data["line_width"] = [
            net_plotted.get_edge_data(a, b)[weight_type] / 10
            for a, b in net_plotted.edges()
        ]
    elif weight_type == "weight_exp":
        graph_renderer.edge_renderer.data_source.data["line_width"] = [
            1 for a, b in net_plotted.edges()
        ]
    graph_renderer.edge_renderer.glyph.line_width = {"field": "line_width"}

    plot.renderers.append(graph_renderer)

    show(plot, notebook_handle=True)

    return plot


# %%
field_name_2_index = {
    field: n for n, field in enumerate(average_sector_skills_df.index)
}
dists = euclidean_distances(average_sector_skills_df, average_sector_skills_df)

# %%
edge_list = get_edge_list(average_sector_skills_df, dists, field_name_2_index)
edge_list.head(2)

# %%
net = create_network(edge_list, weight_type="weight")
net, kd_2_number = set_node_attributes(net, sector_2_kd, sector_2_parent)

# %%
plot = plot_net(net, color_by_mapper=kd_2_number, color_by="knowledge_domain")

# %%
output_file(
    filename=f"between_occupation_skill_similarity_{today}.html", title=plot.title.text
)
save(plot)

# %% [markdown]
# ## Skill occurrences in 2 jobs
# - just for HGV and careworker job adverts
# - each node is a job advert

# %%
## job id to occupation dict
job_id_2_occ = dict(zip(skill_sample_df["job_id"], skill_sample_df["occupation"]))

# %%
hgv_driver_data = skill_sample_df[
    skill_sample_df["occupation"].apply(lambda x: "Hgv" in str(x) if x else None)
]
print(hgv_driver_data["occupation"].unique())
hgv_driver_job_ids = hgv_driver_data["job_id"].tolist()
len(hgv_driver_job_ids)

# %%
careworker_data = skill_sample_df[
    skill_sample_df["occupation"].apply(
        lambda x: "care work" in str(x).lower() if x else None
    )
]
print(careworker_data["occupation"].unique())
careworker_job_ids = careworker_data["job_id"].tolist()
len(careworker_job_ids)

# %%
hgv_carer_counts_dict = {
    j: v
    for j, v in job_id_2_skill_count.items()
    if j in set(hgv_driver_job_ids + careworker_job_ids)
}

# %%
hgv_carer_skills_df = get_cooccurence_matrix(
    hgv_carer_counts_dict, skill_id_2_ix, convert_int=False
)
hgv_carer_skills_df.head(2)

# %% [markdown]
# ## One sector skill information
# - Network of cooccuring skills coloured by skill taxonomy
# - Most and least common skills

# %%
ix_2_skill_id = {v: k for k, v in skill_id_2_ix.items()}

# %%
# The average presence of a skill in each sector (doesnt account for duplication)
average_presence_sector_skills = {}
for sector, job_ids in tqdm(sector_2_job_ids.items()):
    total_sector_skills = Counter()
    for job_id in job_ids:
        total_sector_skills += Counter({j: 1 for j in job_id_2_skill_count[job_id]})
    average_presence_sector_skills[sector] = {
        k: v / len(job_ids) for k, v in total_sector_skills.items()
    }

# %%
sector = "1st Line Support/Helpdesk"

# %%
sorted_average_skill_mentions = sorted(
    average_presence_sector_skills[sector].items(),
    key=lambda item: item[1],
    reverse=True,
)
top_bottom_n = 10
skill_props = (
    sorted_average_skill_mentions[0:top_bottom_n]
    + sorted_average_skill_mentions[-top_bottom_n:]
)
skill_props_df = pd.DataFrame(
    [(esco_code2name[ix_2_skill_id[s]], p) for s, p in skill_props],
    columns=["skill", "proportion"],
)
skill_props_df.head(2)

# %%
chart_title = (
    f"Skills in the highest and lowest proportions of job adverts for {sector}"
)

chart = (
    alt.Chart(skill_props_df)
    .mark_bar()
    .encode(
        alt.Y("skill", title="ESCO skill", sort=None),
        alt.X("proportion", title="Proportion of job adverts"),
        #     alt.Color('proportion'),
        tooltip=["skill", "proportion"],
    )
    .configure_mark(opacity=0.8, color="pink")
    .properties(width=300)
)
chart = configure_plots(chart, chart_title=chart_title)
# AltairSaver().save(chart, f"{today}_num_job_adverts_chunked")
chart

# %%
job_ids = sector_2_job_ids[sector]

# The combinations of unique skills within a job advert
job2skill = {
    job_id: skills
    for job_id, skills in job_id_2_skill_count.items()
    if job_id in job_ids
}

pairs = list(
    chain(*[list(combinations(sorted(list(set(x))), 2)) for x in job2skill.values()])
)
pairs = [x for x in pairs if len(x) > 0]
edge_list = pd.DataFrame(pairs, columns=["source", "target"])
edge_list["weight"] = 1
edge_list_weighted = (
    edge_list.groupby(["source", "target"])["weight"].sum().reset_index(drop=False)
)

# %%
edge_list_weighted.head(2)

# %%
edge_list_weighted_sorted = edge_list_weighted.sort_values(
    "weight", ascending=False
).reset_index(drop=True)
edge_list_weighted_sorted.head(2)

# %%
for top_n in range(10):
    print(
        f"'{esco_code2name[ix_2_skill_id[edge_list_weighted_sorted.iloc[top_n]['source']]]}' and '{esco_code2name[ix_2_skill_id[edge_list_weighted_sorted.iloc[top_n]['target']]]}' are commonly co-occuring"
    )

# %%
net = nx.from_pandas_edgelist(edge_list_weighted, edge_attr=True)

min_degree = 1
max_degree = 500  # This skill can only co-occur with another in a maximum of max_degree job adverts
min_weight = 10  # Has to cooccur in at least min_weight job adverts

keepnodes = [
    node
    for node, degree in dict(net.degree()).items()
    if degree in range(min_degree, max_degree)
]
net_filt = net.subgraph(keepnodes)
high_weight = (
    (s, e) for s, e, w in net_filt.edges(data=True) if (w["weight"] > min_weight)
)
net_filt = net_filt.edge_subgraph(high_weight)

node_attrs_name = {
    node_num: esco_code2name[ix_2_skill_id[node_num]] for node_num in net_filt.nodes()
}
nx.set_node_attributes(net_filt, node_attrs_name, "skill")
len(net_filt.nodes())

# %%
net_plotted = net_filt

plot = Plot(plot_width=600, plot_height=600)
plot.title.text = "Skill graph col"

node_hover_tool = HoverTool(tooltips=[("Skill", "@skill")])
plot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool(), WheelZoomTool(), SaveTool())

graph_renderer = from_networkx(net_plotted, nx.spring_layout, scale=1, center=(0, 0))

graph_renderer.node_renderer.glyph = Circle(
    size=10, fill_color="red", line_color=None
)  # Spectral4[0])
graph_renderer.edge_renderer.glyph = MultiLine(
    line_color="black", line_alpha=0.3, line_width=0.5
)
plot.renderers.append(graph_renderer)

show(plot, notebook_handle=True)

# %%
