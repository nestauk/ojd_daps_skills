# Unmatched skills

This folder contains a `SpanClassifier` class that predicts whether or not an extracted 'skill' is indeed a skill or not. It is a logistic regression model that is trained on BERT embeddings of 494 labelled spans (skill vs. non-skill).

Examples of extracted non-skills include:

```
['17',
 ', video scripts',
 '<span style="font-family Calibri, sans',
 '00 - 17',
 '42C',
 '6 billion euros and 90,000 employees, who dedicate themselves to supporting the safety, well-being and comfort of millions of people, we are one of the market leaders',
 'a video consultation',
 'a well-managed school in Newham, East London, then please contact Matthew Stanley',
 'Ability to']
```

Examples of extracted skills include:

```
 ['assisting with cleaning duties',
 'Attention to detail',
 'build confidence',
 'build rapport with customers',
 'C# programming',
 'checking stock records',
 'cleaning in warehouses offices',
 'coaching Business Leaders']
 ```

 ## Span classifier - `2022.10.17`

 test_size = `0.2`
 shuffle=`True`
 random_state=`42`

The train results on 395 spans were:

              precision    recall  f1-score   support

    nonskill       0.87      0.83      0.85       195
       skill       0.84      0.88      0.86       200
    accuracy                           0.85       395

The test results on 99 spans were:

              precision    recall  f1-score   support

    nonskill       0.88      0.71      0.79        52
       skill       0.74      0.89      0.81        47
    accuracy                           0.80        99