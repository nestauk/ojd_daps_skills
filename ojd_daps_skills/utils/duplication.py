"""
Functions to deal with duplication in job adverts.

get_deduplicated_job_adverts takes the job adverts (only job id and date columns are needed)
and chunks up the dates and then removes duplicates from each chunk, before recombining.

The duplicates output from the OJO pipeline is needed here (which is a list of pairs of job ids
which are semantically the same).
"""

from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from ojd_daps_skills import logger


def get_date_ranges(start, end, num_units=7, unit_type="days"):
    """
    Chunk up a range of dates into user specified units of time.
    Inputs:
        start, end: strings of a date in the form %Y-%m-%d (e.g. '2021-05-31')
        num_units: time period of chunks, e.g. if num_units=3 and unit_type="days" then this is 3 days
        unit_type: ["days", "weeks", "months"]
    Outputs:
        A list of the start and end dates to each chunk
        The last chunk may be partial.
    """
    start = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")

    if unit_type == "days":
        time_addition = timedelta(days=num_units)
    elif unit_type == "weeks":
        time_addition = timedelta(weeks=num_units)
    elif unit_type == "months":
        time_addition = relativedelta(months=num_units)
    else:
        print('unit_type not recognised - use ["days", "weeks", "months"]')

    chunk_start = start
    chunk_end = chunk_start + time_addition
    list_ranges = []
    while chunk_end < end:
        list_ranges.append((chunk_start, chunk_end))
        chunk_start += time_addition
        chunk_end = chunk_start + time_addition
    list_ranges.append((chunk_start, end))
    return list_ranges


def create_date_chunks_mapper(first_date, last_date, num_units=7, unit_type="days"):
    """
    Gets a dict of all dates in the range and which start date chunk they map to.
    e.g. if num_units=2 and unit_type="days"
    {
    '2021-06-01': '2021-06-01',
    '2021-06-02': '2021-06-01',
    '2021-06-03': '2021-06-03',
    '2021-06-04': '2021-06-03'
    }
    """

    date_ranges_list = get_date_ranges(
        first_date, last_date, num_units, unit_type=unit_type
    )

    date_chunk_mapper = {}
    for start, end in date_ranges_list:
        curr_date = start
        while curr_date < end:
            date_chunk_mapper[curr_date.strftime("%Y-%m-%d")] = start.strftime(
                "%Y-%m-%d"
            )
            curr_date += timedelta(days=1)
        date_chunk_mapper[curr_date.strftime("%Y-%m-%d")] = start.strftime("%Y-%m-%d")
    return date_chunk_mapper


def check_span(num_units, unit_type):

    warning_message = "This is designed for use with spans < 8 weeks"

    if unit_type == "days":
        if num_units >= 56:
            logger.warning(warning_message)
    elif unit_type == "weeks":
        if num_units >= 8:
            logger.warning(warning_message)
    elif unit_type == "months":
        if num_units >= 1:
            logger.warning(warning_message)


def get_deduplicated_job_adverts(
    job_adverts,
    duplicates,
    num_units=7,
    unit_type="days",
    id_col="job_id",
    date_col="date",
    job_loc_col="job_location_raw",
):
    """
    Find the job ids of the deduplicated job adverts based on whether they had any
    duplicates found in job stock chunks (date spans).

    Due to the way duplicates are found, this shouldn't be run for chunks over 8 weeks.

    Input:
        job_adverts: pandas DataFrame with job_id, date and job_loc_col columns
            The date column must be in the form %Y-%m-%d e.g. '2021-05-31'
        duplicates: pandas DataFrame with mappings of duplicated job adverts
                from previously found semantic similarity.
        num_units: time period of chunks, e.g. if num_units=3 and unit_type="days" then this is 3 days
        unit_type: ["days", "weeks", "months"]

    Output:
        job_adverts_dedupe: pandas DataFrame with job_id, date and which date
            chunk it belong to columns. Duplicates have been removed
        dedupe_job_ids (set): the deduplicated job ids
    """

    check_span(num_units, unit_type)

    date_chunk_mapper = create_date_chunks_mapper(
        job_adverts[date_col].min(),
        job_adverts[date_col].max(),
        num_units=num_units,
        unit_type=unit_type,
    )

    job_adverts["start_date_chunk"] = job_adverts[date_col].map(date_chunk_mapper)

    # Find the job advert duplicates for each time chunk
    duplicates_per_group = set()
    for _, grouped_job_adverts in tqdm(
        job_adverts.groupby(["start_date_chunk", job_loc_col])
    ):
        group_job_ids = set(grouped_job_adverts[id_col].tolist())
        group_duplicates = duplicates[duplicates["first_id"].isin(group_job_ids)]
        duplicates_per_group.update(
            group_job_ids.intersection(set(group_duplicates["second_id"].tolist()))
        )

    job_adverts_dedupe = job_adverts[~job_adverts[id_col].isin(duplicates_per_group)]

    return job_adverts_dedupe
