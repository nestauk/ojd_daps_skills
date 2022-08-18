# Model Cards

This page contains model cards for different parts of the skills extraction and mapping pipeline. We detail a high level summary of the pipeline and the pipeline's overall intended use. We then detail different parts of the pipeline components.

### Pipeline High level summary

<p><img alt="am-pm-1" src="/img/full_pipeline.jpeg" /></p>

- Developed by data scientists in Nesta’s Data Analytics Practice, (last updated on X X 2022).
- Overall Pipeline:

  - NER model to extract skill or experience entities in job adverts;
  - SVM model to predict whether the skill entity is a skill or multiskill; if multiskill, apply rules to split multiskills into individual skill entities;
  - embed all skill, split multiskills, multiskill sentences and taxonomy skills using [huggingface’s "sentence-transformers/all-MiniLM-L6-v2"](https://www.google.com/url?q=https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2&sa=D&source=docs&ust=1660824880413352&usg=AOvVaw2dpSwPbxZibtqIcyi7sIIT)
  - map skills (skill, split multiskill and multiskills) onto taxonomy skills using cosine similarity

- For further information or feedback please contact Liz Gallagher (elizabeth.gallagher@nesta.org.uk), India Kerle (india.kerle@nesta.org.uk) or Cath Sleeman (cath.sleeman@nesta.org.uk)

### Pipeline Intended Use

- Extract skills from online job adverts and match extracted skills to a user’s skill taxonomy of choice, such as the [European Commission’s European Skills, Competences, and Occupations (ESCO)](https://esco.ec.europa.eu/en) or [Lightcast’s Open Skills](https://www.economicmodeling.com/2022/03/08/open-skills-taxonomy/#:~:text=Skills%20Taxonomy%2FOpen%20methodology%3A&text=To%20help%20everyone%20speak%20the,resumes%E2%80%94updated%20every%20two%20weeks.).
- Intended users include researchers in labour statistics or related government bodies.

- Out of scope is extracting and matching skills from job adverts in non-English languages; extracting and matching skills from texts other than job adverts; drawing conclusions on general skill demands from many job adverts without expert steer and input; drawing conclusions on new, unidentified skills

### Component Model Card: Extracting Skills

<p><img alt="am-pm-1" src="/img/predicting_entities.jpeg" /></p>

**_Summary_**

**_Intended Use_**

**_Model Factors_**

**_Metrics and Evaluation_**

**_Ethical Considerations_**

**_Caveats and Reccomendations_**

### Component Model Card: Skills to Taxonomy Mapping

<p><img alt="am-pm-1" src="/img/matching_to_taxonomy.jpeg" /></p>

**_Summary_**

**_Intended Use_**

**_Model Factors_**

**_Metrics and Evaluation_**

**_Ethical Considerations_**

**_Caveats and Reccomendations_**
