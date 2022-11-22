# Entity Labelling

To extract skills from job adverts we took an approach of training a named entity recognition (NER) model to predict which parts of job adverts were skills ("skill entities") and which were experiences ("experience entities").

To train the NER model we needed labelled data. First we created a random sample of job adverts and got them into a form needed for labelling using [Label Studio](https://labelstud.io/). More about this labelling process can be found in the [`skill_ner` pipeline](https://nestauk.github.io/ojd_daps_skills/pipeline/skill_ner/README.md).

There are 3 entity labels in our training data:

1. `SKILL`
2. `MULTISKILL`
3. `EXPERIENCE`


The user interface for this labelling task looks like:

![](../../outputs/reports/figures/label_studio.png)

We tried our best to label from the start to end of each individual skill, starting at the verb (if given):
![](../../ojd_daps_skills/pipeline/skill_ner/ner_label_examples/label_eg1.jpg)

Sometimes it wasn't easy to label individual skills, for example an earlier part of the sentence might be needed to define the later part. An example of this is "Working in a team and on an individual basis" - we could label "Working in a team" as a single skill, but "on an individual basis" makes no sense without the "Working" word. In these situations we labelled the whole span as multi skills:
![](../../ojd_daps_skills/pipeline/skill_ner/ner_label_examples/label_eg4.jpg)

Sometimes there were no entities to label:
![](../../ojd_daps_skills/pipeline/skill_ner/ner_label_examples/label_eg5.jpg)

`EXPERIENCE` labels will often be followed by the word "experience" e.g. "insurance experience", and we included some qualifications as experience, e.g. "Electrical qualifications".

### Training dataset

For the current NER model, 5641 entities in 375 job adverts from our dataset of job adverts were labelled; 354 are multiskill, 4696 are skill, and 608 were experience entities. 20% of the labelled entities were held out as a test set to evaluate the models.
