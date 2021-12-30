# Model Card - Salary Prediction with Census data

## Model Details
* Developed by Chao Li for Udacity MLOps Nanodegree
* RandomForest from sklearn with default parameters

## Intended Use
* Intended to be used for completing project assignment. 

## Training Data
* Census Income data [https://archive.ics.uci.edu/ml/datasets/census+income], training data split 

## Evaluation Data
* Census Income data [https://archive.ics.uci.edu/ml/datasets/census+income], test data split

## Metrics
_Please include the metrics used and your model's performance on those metrics._
* Evaluation metrics include Precision, Recall, and F-betar, and are applied across 'education' subgroups
* Performance:
Education   Precision   Recall  F-Beta
Bachelors   0.693       0.701   0.697
HS-grad     0.631       0.397   0.488
Some-college 0.665      0.495   0.567
Doctorate   0.843       0.908   0.874
Masters     0.832       0.823   0.827
10th        0.333       0.125   0.182
9th         1.0         0.0     0.0
Assoc-acdm  0.7         0.547   0.614
Assoc-voc   0.632       0.581   0.605
12th        1.0         0.25    0.4
7th-8th     1.0         0.2     0.333
5th-6th     1.0         0.25    0.4
Preschool   1.0         1.0     1.0
11th        0.9         0.562   0.692
Prof-school 0.866       0.922   0.893
1st-4th     1.0         0.0     0.0

## Ethical Considerations
* Classification based on publicly available data, and no individual or new information is abtained

## Caveats and Recommendations
* Model performance across different education subgroups vary greatly. Should be careful when applying to low performance subgroups
* To furthur improve model's performance (especially for all subgroups), more data should be collected.
