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
# # Setup

# %%
import json
import boto3
import pandas as pd
import altair as alt
import seaborn as sns

# %%
bucket_name = "open-jobs-lake"
file_name = (
    "escoe_extension/inputs/data/skill_ner/data_sample/20220622_sampled_job_ads.json"
)
s3 = boto3.resource("s3")
obj = s3.Object(bucket_name, file_name)
file = obj.get()["Body"].read().decode()
data = json.loads(file)

# %%
df = pd.DataFrame(data).T
df

# %%
df["job_count"] = df.groupby("job_title_raw")["job_title_raw"].transform("count")

# %% [markdown]
# # Analysis

# %% [markdown]
# ## Number of job adverts from each occupation

# %%
len(df["job_title_raw"].unique())
# In the sample there are 3709 unique jobs

# %%
# jobs that appear once:
len(df[df["job_count"] == 1])
appear_once = df[df["job_count"] == 1]["job_title_raw"].tolist()
appear_once
appear_more = df[df["job_count"] != 1]["job_title_raw"].tolist()
appear_more

df_more = df[df["job_count"] != 1]
df_more

# %% [markdown]
# ### (1)

# %%
source = df_more

alt.Chart(source).mark_bar().encode(x="job_title_raw", y="count()")

# %% [markdown]
# ## Number over time

# %% [markdown]
# ### (1)

# %%
source = df

alt.Chart(source).mark_bar().encode(
    y="count(job_title_raw)",
    x=alt.X("month(created)", title="Month"),
    column=alt.Column("year(created)", title="Year"),
)

# %% [markdown]
# ## Top skills found

# %%
df_top = df.sort_values("job_count", ascending=False)
df_top.drop_duplicates(subset="job_title_raw", keep="first", inplace=True)

# %%
df

# %% [markdown]
# ### (1)

# %%
top_10 = df_top.head(10)[["job_title_raw", "job_count"]].reset_index(drop=True)
top_10

# %% [markdown]
# expl

# %% [markdown]
# ## Exploration

# %%
df_new = df.sort_values("job_count", ascending=False)
df_50 = df.head(50)

# %%

# %%
source = df_50

alt.Chart(source).mark_bar().encode(
    x="job_location_raw",
    y="count()",
)


# %%
df.columns.to_list()

# %%
df["raw_salary_unit"].unique()

# %%
df_new
