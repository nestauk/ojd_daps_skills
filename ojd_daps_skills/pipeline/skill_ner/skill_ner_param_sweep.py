from ojd_daps_skills.pipeline.skill_ner.ner_spacy import JobNER
from ojd_daps_skills import bucket_name
from ojd_daps_skills.getters.data_getters import save_to_s3, get_s3_resource
from ojd_daps_skills import bucket_name

s3 = get_s3_resource()

import random
from tqdm import tqdm
from datetime import datetime as date
import os

labelled_date_filename = (
    "escoe_extension/outputs/labelled_job_adverts/combined_labels_20220824.json"
)
convert_multiskill = True
train_prop = 0.8

job_ner = JobNER(
    BUCKET_NAME=bucket_name,
    labelled_date_filename=labelled_date_filename,
    convert_multiskill=convert_multiskill,
    train_prop=float(train_prop),
)
data = job_ner.load_data()
train_data, test_data = job_ner.get_test_train(data)


def train_model(drop_out, learn_rate, num_its):
    job_ner.prepare_model()
    nlp = job_ner.train(
        train_data,
        test_data,
        print_losses=False,
        drop_out=float(drop_out),
        num_its=int(num_its),
        learn_rate=learn_rate,
    )
    eval_results = job_ner.evaluate(test_data)
    return job_ner.all_losses, eval_results


num_its = 10
num_experiments = 20

date_stamp = str(date.today().date()).replace("-", "")
file_name = f"outputs/data/skill_ner/parameter_experiments/results_{num_experiments}_{date_stamp}_eval.json"

s3_file_name = f"escoe_extension/{file_name}"

experiments = []
if os.path.exists(file_name):
    os.remove(file_name)
for i in tqdm(range(num_experiments)):
    random.seed(10 * i)
    drop_out = random.uniform(0.08, 0.2)
    learn_rate = random.uniform(0.0009, 0.005)
    result = {"drop_out": drop_out, "learn_rate": learn_rate}
    losses, eval_results = train_model(drop_out, learn_rate, num_its)
    result["losses"] = losses
    result["eval_results"] = eval_results
    experiments.append(result)
    with open(file_name, "a") as file:
        file.write(str(result))
        file.write("\n")


save_to_s3(
    s3,
    bucket_name,
    experiments,
    file_name,
)
