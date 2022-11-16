#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Comparing lightcast's algorithm with our lightcast mapped skills at the skill level.

We guarantee to map to skills based on setting the minimum cosine similarity threshold to 0.
"""


# In[3]:


from ojd_daps_skills import config, bucket_name, logger
from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    load_s3_data,
    save_to_s3,
)
import pandas as pd


# In[5]:


def percent_overlap(ojo_skills, lightcast_skills):
    """Calculate the percent overlap between two lists"""
    
    if len(ojo_skills) and len(lightcast_skills) > 0:

        setA = set(ojo_skills)
        setB = set(lightcast_skills)

        overlap = setA & setB
        universe = setA | setB


        result1 = float(len(overlap)) / len(setA) * 100
        result2 = float(len(overlap)) / len(setB) * 100
        result3 = float(len(overlap)) / len(universe) * 100

        return result1, result2, result3
    
    else:
        return 100, 100, 100


# In[4]:


s3 = get_s3_resource()
ojo_lightcast_skills = load_s3_data(s3, bucket_name, 'escoe_extension/outputs/evaluation/ojo_esmi_skills/ojo_lightcast_skills_20221115.json')


# In[6]:


for job_id, skill_info in ojo_lightcast_skills.items():
    comps = percent_overlap(skill_info['ojo_skills'], skill_info['lightcast_skills'])
    for comp_type, comp in zip(('ojo_skills_overlap', 'lightcast_skills_overlap', 'universal_overlap'), comps):
        skill_info[comp_type] = comp    


# In[25]:


ojo_lightcast_skills_df = pd.DataFrame(ojo_lightcast_skills).T
ojo_lightcast_skills_df = ojo_lightcast_skills_df.sort_values('lightcast_skills_overlap', ascending=False)
ojo_lightcast_skills_df = ojo_lightcast_skills_df[~(ojo_lightcast_skills_df['ojo_skills'].str.len() == 0) & (ojo_lightcast_skills_df['lightcast_skills'].str.len() != 0)]


# In[31]:


print('percent overlap analysis')

print(f"the % of job adverts with no skills overlap is: {len(ojo_lightcast_skills_df[ojo_lightcast_skills_df['ojo_skills_overlap'] == 0.0])/len(ojo_lightcast_skills_df)}")
print(f"the average # of lightcast skills we extract is: {ojo_lightcast_skills_df.ojo_skills.apply(lambda x: len(x)).mean()}")
print(f"the median # of lightcast skills we extract is: {ojo_lightcast_skills_df.ojo_skills.apply(lambda x: len(x)).median()}")

print(f"the average # of lightcast skills lightcast extracts is: {ojo_lightcast_skills_df.lightcast_skills.apply(lambda x: len(x)).mean()}")
print(f"the median # of lightcast skills lightcast extracts is: {ojo_lightcast_skills_df.lightcast_skills.apply(lambda x: len(x)).median()}")

print(f"of the job adverts with overlap, on average, {ojo_lightcast_skills_df[ojo_lightcast_skills_df['lightcast_skills_overlap'] != 0.0].lightcast_skills_overlap.mean()} of lightcast skills are present in our current approach.")
print(f"of the job adverts with overlap, the median is {ojo_lightcast_skills_df[ojo_lightcast_skills_df['lightcast_skills_overlap'] != 0.0].lightcast_skills_overlap.median()} of lightcast skills are present in our current approach.")

print(f"of the job adverts with overlap, on average, {ojo_lightcast_skills_df[ojo_lightcast_skills_df['ojo_skills_overlap'] != 0.0].ojo_skills_overlap.mean()} of our skills are present in lighcast skills.")
print(f"of the job adverts with overlap, the median is {ojo_lightcast_skills_df[ojo_lightcast_skills_df['ojo_skills_overlap'] != 0.0].ojo_skills_overlap.median()} of our skills are present in lightcast skills.")

