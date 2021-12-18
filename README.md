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
The best F1 result is: **78.6%**

HPO result:
![Screenshot from 2021-12-18 15-28-15](https://user-images.githubusercontent.com/36601982/146644690-9b2c1a76-5102-4ec4-98fd-b4b877c972fc.png)

Confusion matrix:
We can see from the confusion matrix, that it is really good on the train, but less good 
on the validation set, mostly on Pali (due to having way more Pali images than images
of other classes). This suggests overfitting. 

Train:
![confusion_matrix_train_9](https://user-images.githubusercontent.com/36601982/146644700-7429ab3e-6a73-437a-a57c-3a5e7a82aa72.png)

Val:
![confusion_matrix_val_9](https://user-images.githubusercontent.com/36601982/146644708-df31cba3-3234-440b-9de8-ae0c1a5a6c5e.png)

By class F1 score:
We can see the same from the classwise F1 scores than the confusion matrix: a large 
difference between the train and val scores, and that on the validation set we have
way higher score for Pali.

Train:
![F1_train_9](https://user-images.githubusercontent.com/36601982/146644711-398997d7-6ed3-4ef8-8331-6bd6dc21d675.png)

Val:
![F1_val_9](https://user-images.githubusercontent.com/36601982/146644717-3b34163f-4cdc-4a87-9d50-12874f886338.png)

### Example of model visualization:
According to the model visualization the most important parts for the model during predictions are:
- cat ears
- cat eyes
- cat nose
- cat face
- cat paws

During during good predictions (first 2 image) these are the important areas, while bad predictions (last image), there are no clue of using these areas.
Images:

<p float="left">
 <img src="https://user-images.githubusercontent.com/36601982/146644928-7052f412-9756-418b-8e7f-de37179f13eb.jpg"  width="30%" height="30%" />
 <img src="https://user-images.githubusercontent.com/36601982/146645181-8aa838d4-19a9-4f2d-841a-dac0ae06b0d6.jpg"  width="30%" height="30%" />
 <img src="https://user-images.githubusercontent.com/36601982/146645424-3b519d21-b6a6-434f-a8e0-cf8e60a7250a.jpg"  width="30%" height="30%" />
</p>

Guided gradients:

<p float="left">
 <img src="https://user-images.githubusercontent.com/36601982/146645074-66838283-a6ca-4e44-b4d9-dbbc1c97c054.jpg"  width="30%" height="30%" />
 <img src="https://user-images.githubusercontent.com/36601982/146645472-7616445c-f319-4e7c-aeac-698288e21f2f.jpg"  width="30%" height="30%" />
 <img src="https://user-images.githubusercontent.com/36601982/146645473-d443981b-0713-424b-8e39-83ed73552364.jpg"  width="30%" height="30%" />
</p>

Guided gradient saliency map:

<p float="left">
 <img src="https://user-images.githubusercontent.com/36601982/146645085-6310c9bf-3fec-4796-b347-3f8a33140709.jpg"  width="30%" height="30%" />
 <img src="https://user-images.githubusercontent.com/36601982/146645479-a0d13328-a634-4b2f-b817-10ac1b3586fc.jpg"  width="30%" height="30%" />
 <img src="https://user-images.githubusercontent.com/36601982/146645482-a92841aa-9a7c-4fa5-8508-6f3f5ff3a97d.jpg"  width="30%" height="30%" />
</p>

GradCam:

<p float="left">
 <img src="https://user-images.githubusercontent.com/36601982/146645041-efba7d24-62e8-44f7-a1b9-ff8794e1d1d3.png"  width="30%" height="30%" />
 <img src="https://user-images.githubusercontent.com/36601982/146645495-db06222f-cbb2-455d-a9e0-5e01c61d9799.png"  width="30%" height="30%" />
 <img src="https://user-images.githubusercontent.com/36601982/146645501-302bebec-5bdc-40c8-a0db-78a9e02c4964.png"  width="30%" height="30%" />
</p>

Occlusion sensitivity:

<p float="left">
 <img src="https://user-images.githubusercontent.com/36601982/146645097-412fcd54-3759-48e5-838b-4d48efe920c0.png"  width="30%" height="30%" />
 <img src="https://user-images.githubusercontent.com/36601982/146645508-7fd45790-8a09-427d-9fd2-0e0ed2be319c.png"  width="30%" height="30%" />
 <img src="https://user-images.githubusercontent.com/36601982/146645511-338720cb-0d9e-4640-a2f4-e5163a37d8ac.png"  width="30%" height="30%" />
</p>
