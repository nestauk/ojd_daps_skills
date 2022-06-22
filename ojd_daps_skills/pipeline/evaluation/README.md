## Evaluation methods

This directory contains the approaches taken to evaluate both the first and second iteration of the skills algorithm.

### 1. Aggregated ESCO-OJO occupation evaluation

This approach compares a list of ESCO skills with a list of extracted OJO skills per occupation that is in both ESCO and OJO. To ensure that there are a reasonable amount of job adverts per ESCO skill, we only examine occupations that have at least 100 job adverts associated to them.

We then calculate the occupation-level accuracy of the first skills extraction algorithm by comparing the number of skills extracted by the first iteration of the skills algorithm in OJO and ESCO compared to the total number of ESCO skills associated to each occupation. This is saved in a timetamped directory in .json.

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