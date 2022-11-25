"""
Add salaries and raw job titles to deduplicated job data generated from process_analysis_data.py

python ojd_daps_skills/analysis/OJO/add_additional_metadata.py

You will need to be connected to Nesta's VPN or on Nesta's wifi to access the SQL database
"""
from ojd_daps_skills.getters.data_getters import get_s3_resource, load_s3_data, save_to_s3
from ojd_daps_skills.utils.sql_conn import est_conn
from ojd_daps_skills import bucket_name, logger
import os
import pandas as pd
from functools import reduce

s3_folder = "escoe_extension/outputs/data/model_application_data"

if __name__ == "__main__":

    conn = est_conn()
    s3 = get_s3_resource()
    #load deduplicated data 
    file_name = os.path.join(s3_folder, 'dedupe_analysis_skills_sample.json')
    job_ads = load_s3_data(s3, bucket_name, file_name)
    logger.info('loaded deduplicated job data...')
    
    job_ads_df = pd.DataFrame(job_ads)
    job_ads_df['job_id'] = job_ads_df.job_id.astype(int)

    job_ids_formatted = ", ".join([f'"{id_}"' for id_ in list(job_ads_df.job_id)])
    logger.info('formatted job ids for querying...')

    #get raw job titles and salary
    query_job_titles_salary = f"SELECT id, job_title_raw, raw_salary, raw_min_salary, raw_max_salary, raw_salary_band, raw_salary_unit, raw_salary_currency, salary_competitive, salary_negotiable FROM raw_job_adverts WHERE id IN ({job_ids_formatted})"
    job_titles_salary_df = pd.read_sql(query_job_titles_salary, conn).rename(columns={'id': 'job_id'})
    job_titles_salary_df['job_id'] = job_titles_salary_df.job_id.astype(int)
    logger.info('got raw job titles and salaries from sql database...')

    #also get annualised salaries
    query_clean_salary = f"SELECT id, min_annualised_salary, max_annualised_salary FROM salaries WHERE id IN ({job_ids_formatted})"
    clean_salaries_df = pd.read_sql(query_clean_salary, conn).rename(columns={'id': 'job_id'})
    logger.info('got annualised clean salaries from sql database...')

    #merge all 3 dataframes
    all_dfs = [job_ads_df, job_titles_salary_df, clean_salaries_df]
    job_ads_with_metadata = reduce(lambda left,right: pd.merge(left,right,on="job_id",
                                                how='outer'), all_dfs)
    logger.info('merged deduplicated job data with additional metadata...')

    file_name_with_metadata = os.path.join(s3_folder, 'dedupe_analysis_metadata_salaries_titles.csv')
    save_to_s3(s3, bucket_name, job_ads_with_metadata, file_name_with_metadata)