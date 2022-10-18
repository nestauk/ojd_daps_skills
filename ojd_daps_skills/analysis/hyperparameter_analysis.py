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
# ## Hyperparameter analysis
#
# The NER model was trained several times with different hyperparameters (number of iterations, learning rate, drop out rate) to find which parameters gave the best results. The results of the models were saved out and are analysed here.
#
# ### [Experiment 1](#exp_1)
# 30 iterations, 27 different random combos learn rate and drop out rates, no evaluation results
# - Do losses stabilise after 30 iterations?
# - Which hyperparameters give the best final loss?
#
# ### [Experiment 2](#exp_2)
# 10 iterations, 20 different random combos learn rate and drop out rates, evaluation results
# - Does the F1 score for all and just skill entities change much with the hyperparams?
# - Which hyperparameters give the best final loss and model scores?
#
# #### Download the data:
# ```
# aws s3 cp s3://open-jobs-lake/escoe_extension/outputs/quality_analysis/results_12_20220824.json outputs/data/skill_ner/parameter_experiments/results_12_20220824.json
# aws s3 cp s3://open-jobs-lake/escoe_extension/outputs/quality_analysis/results_10_20220824.json outputs/data/skill_ner/parameter_experiments/results_10_20220824.json
# aws s3 cp s3://open-jobs-lake/escoe_extension/outputs/quality_analysis/results_5_20220824.json outputs/data/skill_ner/parameter_experiments/results_5_20220824.json
# aws s3 cp s3://open-jobs-lake/escoe_extension/outputs/quality_analysis/results_20_20220824_eval.json outputs/data/skill_ner/parameter_experiments/results_20_20220824_eval.json
#
# ```

# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import json
import ast
import random

from ojd_daps_skills import PROJECT_DIR

# %% [markdown]
# ## Experiment 1<a id='exp_1'></a>

# %%
data = []
with open(
    f"{PROJECT_DIR}/outputs/data/skill_ner/parameter_experiments/results_12_20220824.json",
    "r",
) as f:
    for line in f:
        data.append(ast.literal_eval(line))
print(len(data))
with open(
    f"{PROJECT_DIR}/outputs/data/skill_ner/parameter_experiments/results_10_20220824.json",
    "r",
) as f:
    for line in f:
        data.append(ast.literal_eval(line))
with open(
    f"{PROJECT_DIR}/outputs/data/skill_ner/parameter_experiments/results_5_20220824.json",
    "r",
) as f:
    for line in f:
        data.append(ast.literal_eval(line))
data = pd.DataFrame(data)
len(data)

# %%
data.head(2)

# %%
data["last_loss"] = data["losses"].apply(lambda x: x[-1])

# %%
num_its = len(data["losses"].iloc[0])

for i, row in data.iterrows():
    plt.plot(range(num_its), row["losses"])

plt.xlabel("Number of iterations")
plt.ylabel("Loss")
plt.title(f"Training losses for {len(data)} models")
plt.savefig("outputs/training_losses_sweep.pdf")

# %%
fig, ax = plt.subplots(figsize=(6, 5))

data.plot.scatter(x="drop_out", y="learn_rate", c="last_loss", cmap=cm.plasma, ax=ax)

num_its = len(data["losses"].iloc[0])

# plt.scatter(x=0.3, y=0.001, marker='x', c='red', s=50);
ax.set_xlabel("Drop out rate")
ax.set_ylabel("Learning rate")
ax.set_title(f"Final training loss (iteration {num_its}) of {len(data)} models")
plt.savefig("outputs/last_loss_sweep.pdf")

# %% [markdown]
# ## Experiment 2<a id='exp_2'></a>

# %%
all_data = []
with open(
    f"{PROJECT_DIR}/outputs/data/skill_ner/parameter_experiments/results_20_20220824_eval.json",
    "r",
) as f:
    for line in f:
        all_data.append(ast.literal_eval(line))
all_data = pd.DataFrame(all_data)
len(all_data)

# %%
all_data["last_loss"] = all_data["losses"].apply(lambda x: x[-1])
all_data["all_f1"] = all_data["eval_results"].apply(
    lambda x: x["results_summary"]["All"]["f1"]
)
all_data["skill_f1"] = all_data["eval_results"].apply(
    lambda x: x["results_summary"]["SKILL"]["f1"]
)
all_data["ex_f1"] = all_data["eval_results"].apply(
    lambda x: x["results_summary"]["EXPERIENCE"]["f1"]
)

# %%
fig, ax = plt.subplots(figsize=(6, 5))

all_data.plot.scatter(
    x="drop_out", y="learn_rate", c="last_loss", cmap=cm.plasma, ax=ax
)

num_its = len(all_data["losses"].iloc[0])

# plt.scatter(x=0.3, y=0.001, marker='x', c='red', s=50);
ax.set_xlabel("Drop out rate")
ax.set_ylabel("Learning rate")
ax.set_title(f"Final training loss (iteration {num_its}) of {len(all_data)} models")
plt.savefig("outputs/last_loss_sweep_10its.pdf")

# %%
fig, axs = plt.subplots(1, 3, figsize=(23, 5))

all_data.plot.scatter(
    x="drop_out", y="learn_rate", c="all_f1", cmap=cm.plasma, ax=axs[0], s=50
)

axs[0].set_xlabel("Drop out rate")
axs[0].set_ylabel("Learning rate")
axs[0].set_title(f"All labels F1 of {len(all_data)} models")

all_data.plot.scatter(
    x="drop_out", y="learn_rate", c="skill_f1", cmap=cm.plasma, ax=axs[1], s=50
)

axs[1].set_xlabel("Drop out rate")
axs[1].set_ylabel("Learning rate")
axs[1].set_title(f"Skill labels F1 of {len(all_data)} models")

all_data.plot.scatter(
    x="drop_out", y="learn_rate", c="ex_f1", cmap=cm.plasma, ax=axs[2], s=50
)

axs[2].set_xlabel("Drop out rate")
axs[2].set_ylabel("Learning rate")
axs[2].set_title(f"Experience labels F1 of {len(all_data)} models")

plt.savefig("outputs/model_metrics_10its.pdf", bbox_inches="tight")

# %%
