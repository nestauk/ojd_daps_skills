## Evaluation methods

This directory contains the approaches taken to evaluate both the first and second iteration of the skills algorithm.

### 1. Aggregated ESCO-OJO occupation evaluation

This approach compares a list of ESCO skills with a list of extracted OJO skills per occupation that is in both ESCO and OJO. To ensure that there are a reasonable amount of job adverts per ESCO occupation, we only examine occupations that have at least 100 job adverts associated to them. We also only compare skills that are mentioned in at least X% of job adverts per occupation, where X is 0.5 x the standard deviation above the skill percentage median (although this threshold can be modified) per occupation.

The output is a .json where we report on the skills that are mentioned in at least X percent of OJO job adverts AND ESCO skills per occuption, skills mentioned in OJO but not ESCO, skills mentioned in ESCO but not OJO and % of ESCO skills in the skills mentioned in at least X% of OJO job adverts.

To run the script:

You will need to have the Nesta SQL database credentials stored on your computer, and for this location to be saved in an environmental variable:

Create a mysqldb_team_ojo_may22.config file of SQL credentials, this looks like:

```
[client]
user=XXX
password=XXX
host=XXX
port=XXX
```

Then, connected to the Nesta VPN and in your activated conda environment, run:

```
export SQL_DB_CREDS="$HOME/path/to/mysqldb_team_ojo_may22.config"
python aggregate_ojo_esco_evaluation.py
```

#### Key Observations

By manually reviewing both the lists of OJO and ESCO skill comparisons per occupation and the % of ESCO skills in the skills mentioned in at least X% of OJO job adverts, we can make a number of observations:

- **There is not a huge amount of overlap between top OJO skills and ESCO skills...** The maximum % of ESCO skills in the skills mentioned in at least X% of OJO job adverts is only 50%. This is for the occupation auditor, with only 145 job adverts associated to it. There are 78 occupations where there is no overlap between the top OJO skills and ESCO skills.

- **...however, some OJO skills not in ESCO extracted appear to be relevant to the occupation.** For example, some of the top OJO skills associated to the sheet metal worker occupation NOT in ESCO include 'perform metal work', 'lasers', 'manipulate stainless steel', 'repair metal sheets', 'welding techniques'. These appear very relevant to the occupation. This is similarly the case for the chef de partie occupation where a number of OJO skills not in ESCO appear relevant, like 'food service operations', 'food engineering', and 'food safety standards'.

- **There are many similar top OJO skills across different occupations.** These can be transversal-ish like 'attention to detail', 'English' and 'perform services in a flexible manner'. However, they can also be unrelated to the occupation but extracted from non-skill sentences, a key flaw in the first iteration of the skills algorithm. The top skills 'health and safety in the workplace', 'health and safety regulations', 'communication', 'economics', 'databases' are far reaching. For 'database', occupations that require that skill in OJO range from web developer to team leader, senior buyer, school administrator, marketing assistant and chef de partie.

- **ESCO skills can be particular.** while the skills extracted from OJO may not always be relevant, there is critisim for ESCO skills themselves. For example, for the occupation 'sustainability manager', ESCO skills include: 'integrate headquarter's guidelines into local operations', 'monitor compliance with licensing agreements' and 'assess groundwater environmental impact'. While we could extract this from some 'sustainability manager' job adverts, it does appear quite specific and perhaps not relevant to _all_ sustainability managers.

### 2. EMSI skills

To extract EMSI skills from a random sample of 50 OJO job adverts, you need to [first create an account with EMSI]("https://skills.emsidata.com/extraction"). They will send you API credentials that you will need to run the script.

With your emailed credentials, to run the script:
`python emsi_evaluation.py --client-id CLIENT_ID --client-secret CLIENT_SECRET`

This will output a saved .json with job ids, job description, extracted v1 OJO skills and ESMI skills. BEWARE: you can only call the API 50 times A MONTH! So running the script will take you out for the whole month.
