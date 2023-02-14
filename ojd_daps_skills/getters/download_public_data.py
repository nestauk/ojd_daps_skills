from ojd_daps_skills import PUBLIC_DATA_FOLDER_NAME, PROJECT_DIR

import os
import platform
import zipfile

def download():

    public_data_dir = os.path.join(PROJECT_DIR, PUBLIC_DATA_FOLDER_NAME)

    if platform.system() == "Windows":
      os.system(
        f'aws --no-sign-request --region=eu-west-1 s3 cp s3://open-jobs-indicators/escoe_extension/{PUBLIC_DATA_FOLDER_NAME}.zip "{public_data_dir}.zip"'
      )
      with zipfile.ZipFile(f"{public_data_dir}.zip", 'r') as zip_ref:
        zip_ref.extractall(f"{PROJECT_DIR}")
        zip_ref.close()
        os.remove(f"{public_data_dir}.zip")
      
      return

    os.system(
      f"aws --no-sign-request --region=eu-west-1 s3 cp s3://open-jobs-indicators/escoe_extension/{PUBLIC_DATA_FOLDER_NAME}.zip {public_data_dir}.zip"
    )
    os.system(f"unzip {public_data_dir}.zip -d {PROJECT_DIR}")
    os.system(f"rm {public_data_dir}.zip")
    


if __name__ == "__main__":
    download()
