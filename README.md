##  Using Sagemaker in script mode 

1. Create an IAM role with associated policies.
    1.1 It will allow you to choose between a number of policy options, i.e. Data scientist, MLOps etc. Choose the one that has cloud watch enabled. 

2. Start a Notebook instance.
    2.1 Pick an EC2 instance: family (t, p is for GPU), 
                              size (medium, large), 
                              version 
    Your EC2 instance will have multiple cores, when you run your notebook everything will run through all cores, multithreading - you can resize EC2 and EBS on the fly as per memory requirements
    2.2 Add EBS Volume (stores data, by default stores GB) (NOTE: Didn't encounter this in findings)
    2.3 Add or create a git repo (automaically installed on notebook instance when its created)
    2.4 Other settings: encryption + internet access + lifecycle Config (bash script?)

3. Using a Notebook instance.
    Elastic inference is a portion of GPU ou can attach to EC2 instance (NOTE didn't encounter this in findings)
    3.1 You can create new notebooks and terminal which starts your EBS volume
    3.2 Import sagemaker libraries (look at example)
    3.3 get_execution_role grabs IAM policy associated with notebook and gives access to s3 bucket

4. Training in a Notebook instance. (GPU mode vs local)
    AWS sagemaker manages a container that lives in ECR already
    4.1 It may be possible to choose a container (NOTE: Didn't encounter htis in findings) Choose tf container, pytorch container, etc, 
    4.2 Specify the entry point parameter for the sagemaker estimator (file path to your custom script), An estimator specifies everything, container, role associated with that notebook, EC2instance count, EC2 type, EBS volume size. Set script_mode to True. The estimator parameters, if set correctly, will start the correct contrainer without needing to specify.
    4.3 Include any extra libraries with a requirements.txt. EC2 instances alive for the number of seconds your model is training while you develop on another EC2 instance
    4.4 Pick hyperparameter ranges.  Pick Performance evaluation technique via the objective_metric_name, i.e. AUC from validation 
    Choose number of jobs (number of models you want to train), optimizes hyperparameters at each step in time using a baysian optimzer (you can also use random search)
    4.5 Data sent through channels via s3 in the notebook itself. need separate training and validation datasets. 
    4.6 estimator.fit() spins u a cluster
    4.7 Use web server for inference 
    4.8 It is possible to access the container terminal in anotebook instance (NOTE didn't encounter this in findings)


## Problems encountered with the current script

1. matplotlib was not found een when !pip3 installed in notebook instance or put in requirements.txt In the end had to comment it out of training.py.
2. Could not find out how to access the terminal. 
3. Model did not save, most likely due to path issue. 
