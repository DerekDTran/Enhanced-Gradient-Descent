# Regression using Enhanced Gradient Descent

1.1 Background
  These files utilizes Gradient Descent to build a model that will predict the Group for an observation. 

1.2 Coding in Python
In this section, we cover how to build and run part1.py

1. The code uses Adaptive Gradient as the enhancement on the basic gradient descent
2. For this code, I selected a dataset of heart failure clinical records. The dataset uses "DEATH_EVENT" as the target and the rest of the variables are considered as predictors.
3. To preprocess the dataset, I first check for missing and null values. Using different plots, I determined that the "time" variable is redundant, which I then removed.
4. When obtaining the training and test datasets, I first split the entire dataset based on the target and predictor variables. Then using the model selection library from sklearn, I split the target and predictor variables into the training and test dataset based on a 80/20 ratio to training and test datasets respectively.
5. In this code, I modified the learning rate in accordance to the Adaptive Gradient Descent algorithm, marked down as the AdaptGrad function, and used MSE as a parameter to optimize, maked down as the getMSE function. Note that in the AdaptGrad function, I create a plot showing the relation between Iterations and Cost for a learning rate. After the functions, I begin the process in reading the training dataset by using the AdaptGD function. During the process, I store the iterations, weights, and mse with respect to the learning rate in a list marked down as model. Furthermore, I am creating a string that stores the information previously stated. This log string will then be stored ina txt file once the process is completed.
6. In order to obtain the optimal weights for the model, I must search through the model list and find the lowest mse value. Since the learning rate and iterations are not used in the model, a new list is created containing weights and the respective mse. The list is than searched for the lowest mse where once found the respective weights are used to create the model. The model is then used to predict values for "DEATH_EVENT" from the test dataset where the error rate values is stored and printed in console.
7. After completing the process and predicting values from the model, I would consider I am satisfied with this solution as it lowers runtime while also providing a reliable model.

2 Linear Regression using ML libraries
In this section, we cover how to build and run part2.py

1. The code uses the SGDRegression library
2. For this code, I selected a dataset of heart failure clinical records. The dataset uses "DEATH_EVENT" as the target and the rest of the variables are considered as predictors.
3. To preprocess the dataset, I first check for missing and null values. Using different plots, I determined that the "time" variable is redundant, which I then removed.
4. When obtaining the training and test datasets, I first split the entire dataset based on the target and predictor variables. Then using the model selection library from sklearn, I split the target and predictor variables into the training and test dataset based on a 80/20 ratio to training and test datasets respectively.
5. At this point, I use the SGDRegression library to fit a model based on the training and test data. The code would the print the coefficients of the model and prediction of the test dataset.
6. I am more satisfied with this code for its simple commands to use the library in fitting a model and creating predictions.
