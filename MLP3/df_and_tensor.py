# Clear the console and remove all variables present on the namespace.
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

import pandas as pd
import tensorflow as tf
import numpy as np

SHUFFLE_BUFFER = 500
BATCH_SIZE = 2

# csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv')
# df = pd.read_csv(csv_file)

df = pd.DataFrame(np.random.rand(6000000, 6) * 100, 
                       columns=['Strike', "Time to Maturity", 
                                "Option_Average_Price", "RF Rate", 
                                "Sigma 20 Days Annualized", 
                                "Underlying Price"])

# numeric_feature_names = ['age', 'thalach', 'trestbps',  'chol', 'oldpeak']
numeric_feature_names = ["Strike", "Time to Maturity", "Option_Average_Price", 
                          "RF Rate", "Sigma 20 Days Annualized", 
                          "Underlying Price"]
numeric_features = df[numeric_feature_names]

array = numeric_features.values

tensor = tf.convert_to_tensor(array)

    