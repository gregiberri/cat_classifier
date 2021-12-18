# Cat classifier
### Introduction
The main goal of this code is to make a model that is able to decide which of the two cats 
(Sajt or Pali), or whether both of them are on an image. To train it I used a small dataset 
containing either or both cats.

### Requirements
The required packages can be found in *config/env_files/cat_classifier_env.yml*. 
Dependencies could be installed by running:
> conda env create -f config/env_files/cat_classifier_env.yml

### Configuration
The experiments are run according to configurations. The config files for those can be found in 
*config/config_files*.
Configurations can be based on each other. This way the code will use the parameters of the specified 
base config and only the newly specified parameters will be overwritten.
 
The base config file is *base.yaml*. A hpo example can be found in *base_hpo.yaml*
which is based on *base.yaml* and does hyperparameter optimization only on the specified parameters.
An example for test based on *base.yaml* can be found in *test.yaml*.

### Arguments
The code should be run with arguments: 

--id_tag specifies the name under the config where the results will be saved \
--config specifies the config name to use (eg. config "base" for *config/config_files/base.yaml*)\
--mode can be 'train', 'val', 'test' or 'hpo' 
--save_preds to save the predictions during eval/test
--visualize to make visualization of the model during eval/test

### Required data
The required data's path should be specified inside the config file like:
> data: \
  &emsp; params: \
  &emsp; dataset_path: 'dataset' \

During train, val and hpo the files should be under their class subdirectory 
(eg. *dataset/sajt*). \
During test the files should all be in the specified directory.  

### Saving and loading experiment
The save folder for the experiment outputs can be set in the config file like:
> id: "base"\
  env: \
  &emsp; result_dir: 'results'

All the experiment will be saved under the given results dir: {result_dir}/{config_id}/{id_tag arg}
1. tensorboard files
2. train and val metric csv
3. the best model
4. confusion matrices and by class metrics
5. predictions if set
6. visualizations if set

If the result dir already exists and contains a model file then the experiment will automatically resume
(either resume the training or use the trained model for inference.)

### Usage
##### Training
To train the model use:
> python run.py --config base --mode train

#### Eval
For eval the  results dir ({result_dir}/{config_id}/{id_tag arg}) should contain a model as 
*model_best.pth.tar*. During eval the validation files will be inferenced and the metrics will be calculated.
> python run.py --config base --mode val

#### Test
For test the  results dir ({result_dir}/{config_id}/{id_tag arg}) should contain a model as 
*model_best.pth.tar*. During test the predictions will be saved along with the filepaths in a csv file.\
A pretrained model can be found in [here](https://drive.google.com/file/d/1zCYo_C2dIai4zTsj2YFdrvlAK_I4gJ5g/view?usp=sharing). 
For simplicity it is recommended to copy it under *results/base/base* 
and just change the dataset path to yours in *config/config_files/base_test.yaml*.
> python run.py --config base_test --mode test --save_pred

#### HPO
For hpo use:
> python run.py --config base_hpo --mode hpo

### Example of the results:
HPO result:

Confusion matrix:
We can see from the confusion matrix, that it is really good on the train, but less good 
on the validation set, mostly on Pali (due to having way more Pali images than images
of other classes). This suggests overfitting. 

Train:

Test:

By class F1 score:
We can see the same from the classwise F1 scores than the confusion matrix: a large 
difference between the train and val scores, and that on the validation set we have
way higher score for Pali.

Train:

Test: