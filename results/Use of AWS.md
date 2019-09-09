# Use of AWS

While there is no way to determine whether Jupyter notebooks were created using Sagemaker, we can look into whether they import packages for AWS functionality. Here, I investigate the use of Sagemaker, Boto3, and MXNet in Jupyter notebooks.

Only 0.52% of Python notebooks use any AWS packages. This breaks down to 0.11% importing Sagemaker, 0.2% Boto, 0.29% MXNet, and some notebooks importing more than one. The uses of all three packages have grown significantly. MXNet has grown by far the quickest, followed by Boto and then Sagemaker.

<img src=./images/aws_use.png width=400>

*Figure 1: Use of AWS packages over time*

There is a moderate positive correlation between the use of Sagemaker and Boto3 (r = 0.49), but the use MXNet is independent of both (MXNet and Sagemaker r = 0.03, MXNet and Boto3 r = 0.02).

### Use of boto

Within Boto, the most frequent use is regioninfo by far, with over 2.5 million uses. The next most popular is murk with slightly over a half of a million uses. The most frequent uses of the more current Boto3, on the other hand, are clients, sessions, resource, and S3. S3 is the most commonly used Boto3 resource and client, followed by Sagemaker.

<img src=./images/boto_use.png width=400>

*Figure 2: Use of Boto3 resources and clients*

When using Boto, users can utilize the create training job api, but frequently train a model as they usually would outside of AWS. I found that 5.3% of notebooks that import Boto, Boto3, or Botocore call a function called “create_training_job” while 35.8% call a function called “fit” or “fit_generator” (which are used to train typical Tensorflow, Keras, and Sklearn models). Further, within notebooks that import a version of Boto and use its sagemaker client, 47.4% call "create_training_job" and 42.0% call "fit" or "fit_generator". Only 2.9% call both.

### use of Sagemaker

Sagemaker is most frequently used for get_execution_role, amazon, predictor, and Session. The execution role for the is the IAM role that was created with the notebook instance. The role has to be passed whenever making API calls to create notebook instances, create hyperparameter tuning jobs. create training jobs, and create model. Amazon contains estimator implementations. Predictor is to make “real-time predictions agains sagemaker endpoints with Python objects” ([Sagemaker Documentation](https://sagemaker.readthedocs.io/en/stable/predictors.html)).

<img src=./images/sagemaker_use.png width=400>

*Figure 3: Uses of Sagemaker*

### use of mxnet

The MXNet package is most frequently used for gluon, symbol, and nd. Gluon is an API for deep learning that “makes it easy to prototype, build, and train deep learning models without sacrificing training speed” ([MXNet documentation on Gluon](https://mxnet.incubator.apache.org/versions/master/gluon/)). Symbol is an API that “provides neural network graphs and auto-differentiation” ([MXNet documentation on Symbol](https://mxnet.incubator.apache.org/api/python/symbol/symbol.html)). ND is usually used for the `nd.array` functionality. The NDArray API “provides imperative tensor operations on CPU/GPU” ([MXNet documentation on NDArray](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html)). 

<img src=./images/mxnet_use.png width=400>

*Figure 4: Uses of MXNet*

## Resources

1. MXNet Documentation. “Gluon API”. https://mxnet.incubator.apache.org/versions/master/gluon/.
2. MXNet Documentation. “
    Symbol API". https://mxnet.incubator.apache.org/api/python/symbol/symbol.html.
3. MXNet Documentation. “NDArray API”. https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html.
4. PyPi. “Boto3”. https://pypi.org/project/boto3/#history.
5. PyPi. “Sagemaker”. https://pypi.org/project/sagemaker/1.0.0/#history
6. PyPi. “MXNet”. https://pypi.org/project/mxnet/#history.
7. Sagemaker Documentation. “Predictors”. https://sagemaker.readthedocs.io/en/stable/predictors.html.





