# Task 1: Data Processing
- Exploratory data analysis was done in
`notebooks/0.0-mng-exploratory-data-analysis.ipynb`.

- There were no missing values on the dataset and so no data
imputation was performed. If there were missing data, the appropriate
data imputation technique will depend on the type of feature / data.
In the case of ECG signals, interpolation to fill in missing timepoints
would be appropriate since ECG signals are continuous in time.

- Data augmentation is a technique used to improve the model's performance.
The main idea is to synthetically create new examples from the training
data by applying some transformations. This should lead to increased model
performance since there is more data for the model to learn from.
Domain knowledge would be helpful in creating appropriate augmentations.
The code for data augmentation can be found in `src/data/transforms.py`.
These have parameters such as the magnitude of the noise and can be varied
when called. We do the data augmentation during training while loading the
data, to create an "infinite" number of augmentations. The notebook in
`notebooks/0.1-mng-data-augmentation.ipynb` shows examples of the data
augmentation as well as explanations for each.

# Task 2: Model Training and Fine-tuning
I chose a 1D convolutional neural network (CNN) model with 11 layers for
this ECG heartbeat categorization task. Neural networks automatically learn
features from the data and provides state-of-the-art results in several
tasks, especially when there is a lot of data.

Note: The training/validation split from the training dataset potentially
has some data leakage. According to the paper on this dataset, the MIT-BIH dataset
was derived from ECG recordings from 47 subjects. The ECG signals were split
into 10 second windows and then pre-processed to extract heartbeats. Ideally,
when splitting the training dataset into training/validation, we split at
the patient level and then use the heartbeats from the training/validation
patients. However, I could not find this information.

I performed hyperparameter tuning by sweeping across the following combinations:

Learning rate: {1e-3, 3e-4}
Batch size: {256, 512}
CNN Kernel size: {5, 7}

This was facilitated with the use of Hydra. We could have swept more parameters
if we have time.

Early stopping is a technique used to prevent overfitting. Since
neural networks have way more parameters than training data, it
may overfit the training data and when this happens, the validation
loss increases and validation accuracy decreases. Early stopping stops
the training process when some validation metric is not moving in
the right direction. In this case, I monitor the validation average
F1 score.

I have also added a training callback to reduce the learning rate
when the validation loss plateaus. A high initial learning rate
helps the model train faster at the beginning but in the later stages,
it may cause the model to bounce around the local minimum. Reducing
the learning rate helps in fitting the model.

There are several metrics for multi-class classification problems.
These include accuracy, precision, recall, and F1-score. Since we
have a highly imbalanced data, the accuracy will be skewed higher.
A model which just predicts "normal" for all cases will get high
accuracy (>80%) instantly. Precision, recall and F1-scores for the
abnormal classes are therefore important metrics to look at.

Potential ideas for improvement:
- Oversampling the classes with fewer examples might help the
model predict these better.


# (Optional) Task 3: Testing Holdout Set
Unfortunately, I did not have time to do this. I believe the main
idea here was to see whether the model trained on the MIT-BIH Arrhythmia
subset would generalize to the PTB Database subset.

# Task 4: Deployment Strategies
An important aspect when deploying the model is to ensure that the
pre-processing steps done before model training and validation are also
applied to the new data. To facilitate this, scripts to convert raw
data to processed data should be reproducible and kept in the same
repository as scripts for model training and validation.

Trained models could be stored as artifacts in a MLflow server.
To address model versioning, MLflow provides a model registry which
we can use. Models registered on the registry are provided a version
number and are linked to the MLflow run that produced the model.

For deployment, I believe the best approach is to package the application
and its dependencies into a container (i.e., Docker container).
Containerization allows for portability, scalability, and environment
consistency. Containers can be scaled out quickly using a container
orchestrator if there is an increase in demand.

The script to build the docker container would fetch a specific version
of the model from the model registry and copy the code for pre-processing and
generating predictions to form a complete application.

Monitoring the deployed system is another big task in real world.
Model predictions would presumably be displayed to the clinicians
to help them make decisions. However, to avoid bias, clinicians
should perform the classification independently without seeing the
model predictions. New data, model predictions on the new data,
as well as the clinician labels should be recorded in a database.
On a specific frequency, e.g., once a week, an application would
search through the database and compare model predictions with
clinician labels to check model performance. These results should
then be displayed on a dashboard or e-mailed to the model owners.
