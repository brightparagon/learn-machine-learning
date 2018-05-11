from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# data set
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# load data
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)

# make sure every column has a real value
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# create a DNN with three layers, each of which have 10, 20, 10 units
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/iris_model")

# train
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=2000)

# evaluate
accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)["accuracy"]
print('accuracy: {0:f}'.format(accuracy_score))

# classify two new flowers
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = list(classifier.predict(new_samples, as_iterable=True))
print ('predict: {}'.format(str(y)))
