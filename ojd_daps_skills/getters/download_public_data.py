from ojd_daps_skills import PUBLIC_DATA_FOLDER_NAME, PROJECT_DIR

import os


def download():

    public_data_dir = os.path.join(PROJECT_DIR, PUBLIC_DATA_FOLDER_NAME)

    os.system(f"mkdir {public_data_dir}/")
    os.system(f"mkdir {os.path.join(public_data_dir, 'outputs')}/")
    os.system(
        f"aws --no-sign-request --region=eu-west-1 s3 cp s3://open-jobs-indicators/escoe_extension/{PUBLIC_DATA_FOLDER_NAME}.zip {PUBLIC_DATA_FOLDER_NAME}.zip"
    )
    os.system(f"unzip {PUBLIC_DATA_FOLDER_NAME}.zip")
    os.system(f"rm {PUBLIC_DATA_FOLDER_NAME}.zip")


if __name__ == "__main__":
    download()
