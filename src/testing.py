from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import tensorflow as tf
from conversion_tf import GEMMDecisionTreeImpl

forest = RandomForestClassifier(n_estimators=105)
X, y = make_classification(n_samples=1000, n_features=8,
                           n_informative=4, n_redundant=1,
                           random_state=0, shuffle=True,
                           n_classes=4)

forest.fit(X, y)

model = GEMMDecisionTreeImpl(forest)
X = tf.constant([1,2,3,4,5,6,7,8], shape=[1, 8])

y_pred, y = model(X)

tf.saved_model.save(model, 'test_model')

#run_model = tf.function(lambda x: model(x))
#concrete_func = run_model.get_concrete_function()

print(f'Prediction: {y_pred.numpy()[0]} \t Class Confidence: {y.numpy()[0]}')

#converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter = tf.lite.TFLiteConverter.from_saved_model('./test_model')

tflite_model = converter.convert()