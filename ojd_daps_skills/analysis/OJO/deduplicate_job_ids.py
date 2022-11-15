"""
A script to load job adverts from OJO, deduplicate them and save out the ids of the deduplicated job adverts.

"""

import os
from argparse import ArgumentParser

import pandas as pd

from ojd_daps_skills.analysis.OJO.duplication import get_deduplicated_job_adverts
from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    load_s3_data,
    save_to_s3,
)
from ojd_daps_skills import bucket_name
from ojd_daps_skills import logger

s3 = get_s3_resource()


def create_argparser():

    parser = ArgumentParser()

    parser.add_argument(
        "--s3_folder",
        help="S3 folder of data",
        default="escoe_extension/outputs/data/model_application_data",
        type=str,
    )

    parser.add_argument(
        "--raw_job_adverts_file_name", default="raw_job_adverts_no_desc.csv", type=str
    )

    parser.add_argument("--itl_file_name", default="job_ad_to_itl_v2.csv", type=str)

    parser.add_argument(
        "--duplicates_file_name", default="job_ad_duplicates.csv", type=str
    )

    parser.add_argument(
        "--output_file_name", default="deduplicated_job_ids_6_weeks_v2.csv", type=str
    )

    parser.add_argument("--num_units", default=6, type=int)

    parser.add_argument("--unit_type", default="weeks", type=str)

    return parser


if __name__ == "__main__":

    parser = create_argparser()
    args = parser.parse_args()

    # Load job adverts. First row was in the header (for the sample).
    raw_job_adverts_file_name = os.path.join(
        args.s3_folder, args.raw_job_adverts_file_name
    )
    obj = s3.Object(bucket_name, raw_job_adverts_file_name)

    if "sample" in raw_job_adverts_file_name:
        raw_job_adverts = pd.read_csv(
            "s3://" + bucket_name + "/" + raw_job_adverts_file_name, header=None
        )
        raw_job_adverts.rename(
            columns={0: "job_id", 1: "date", 2: "job_title", 3: "job_ad"}, inplace=True
        )
    else:
        raw_job_adverts = pd.read_csv(
            "s3://" + bucket_name + "/" + raw_job_adverts_file_name
        )
        raw_job_adverts.rename(
            columns={"id": "job_id", "created": "date"}, inplace=True
        )

    # ITL links
    itl_file_name = os.path.join(args.s3_folder, args.itl_file_name)
    job_ad_to_itl = load_s3_data(s3, bucket_name, itl_file_name)

    # Duplicates
    duplicates_file_name = os.path.join(args.s3_folder, args.duplicates_file_name)
    duplicates = load_s3_data(s3, bucket_name, duplicates_file_name)

    # Add raw location - needed for the deduplication
    raw_job_adverts = (
        raw_job_adverts.set_index("job_id")
        .join(job_ad_to_itl.set_index("id")["job_location_raw"])
        .reset_index()
    )

    raw_job_adverts_dedupe = get_deduplicated_job_adverts(
        raw_job_adverts,
        duplicates,
        num_units=args.num_units,
        unit_type=args.unit_type,
        id_col="job_id",
        date_col="date",
        job_loc_col="job_location_raw",
    )
    logger.info(
        f"{len(raw_job_adverts)} job adverts are deduplicated down to {len(raw_job_adverts_dedupe)}"
    )

    save_to_s3(
        s3,
        bucket_name,
        raw_job_adverts_dedupe[["job_id", "end_date_chunk"]],
        os.path.join(args.s3_folder, args.output_file_name),
    )
