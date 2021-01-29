from RNN_class import RNN
from keras.datasets import mnist
import tensorflow as tf

"""
# !!!若電腦的gpu大小不夠大(eg. 跑CNN可能需要6GB)，限制memory_limit才可以正常運作(經測試為2GB)
# 參考網址: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
"""
# Loading dataset
(train_feature, train_label), (test_feature, test_label) = mnist.load_data()
print(len(train_feature), len(train_label))
print(train_feature.shape, train_label.shape)

# 資料前處理
# training data
training_input = RNN.preProcessor()
training_input_image = training_input.normalize(training_input.reshape(train_feature))
training_input_label = training_input.one_hot_encoding(train_label)    # default labels in one-hot encoding

# testing data
testing_input = RNN.preProcessor()
testing_input_image = testing_input.normalize(testing_input.reshape(test_feature))
testing_input_label = testing_input.one_hot_encoding(test_label)     # default labels in one-hot encoding

# 模型建構
rnn = RNN.model()
rnn.layers()
rnn.training(training_input_image, training_input_label)

# 建立好的模型進行預測
prediction = rnn.testing(testing_input_image, testing_input_label)
RNN.preProcessor.show_images_labels_predictions(test_feature, test_label, prediction, 0, 15)   # show出前15個testing結果與label的比對狀況

# Save model& weights
rnn.save_all("Mnist_RNN_model.h5")
print("Mnist_RNN_model.h5 模型已儲存")
rnn.save_weights("Mnist_RNN_weights.weight")
print("Mnist_RNN_weights.weight 權重已儲存")
del rnn


