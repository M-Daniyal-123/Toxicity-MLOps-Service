## Toxicity Classification Service
This is a deep learning project for the predicting the probabilities of different categories for a given text. These categories include toxicity, severe toxicity, obscene, identity attack, insult and threat.

The problem is a multi-label classification problem, where I have employed hugging face's distilbert for sequence classification problem. Classification head is attached to the base model of distilbert, and afterwards, the model is fine tuned on the dataset of this problem. 

In this project, the practices of MLOps are utilized. The experimentation and logging are performed by using an open source tool, MLflow. The best model is saved on to AWS S3 bucket. For the reproducibility, the entire project is wrapped into a docker image. For the use of the service, a REST API endpoint is created, which can be used by running the given docker file.

### Training of Model
To train the model you would need to place the dataset into a data folder in the main directory, the paths can be set in the config.py file. There should be three csvs for train, test, eval, containing text and a probability or binary value for each of the class.

If the data is placed in the folder, you can run the following command in the src file to train the model.

``` python train.py ```

The experiments are tracked using mlflow, make sure mlflow server is working in the localhost or you can set it yourself in the train file. After the script runs, a best model will be saved into models folder with corresponding timestamp as its name. You can pass the path of this model in ```push_s3.py``` file, and it'll upload the model onto AWS S3 bucket.

``` python push_s3.py ```

### Running the RESTAPI
After performing all the above steps, you can can first build the model's docker image by using the following command.

``` docker build -t myflaskapp . ```

After bulding the image, you can run the following command to run the REST API

``` docker run -e AWS_SECRET_KEY=${AWS_SECRET_KEY} -e AWS_ACCESS_KEY=${AWS_ACCESS_KEY} -p 5001:5001 -d myflaskapp ```

You can now send post request to the server with endpoint as predict. The format of the input should be: ``` {"text":"some text here"} ```