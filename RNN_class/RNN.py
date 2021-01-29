from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dropout, Dense
from keras.layers.recurrent import SimpleRNN, LSTM
import numpy as np
import matplotlib.pyplot as plt



class preProcessor:
    def __init__(self):
        pass

    # Show image
    def show_image(self, image):
        fig = plt.gcf()
        fig.set_size_inches(2, 2)
        plt.imshow(image, cmap='binary')
        plt.show()

    # Show images and their labels
    @staticmethod
    def show_images_labels_predictions(images, labels, predictions, start_id, num=10):
        plt.gcf().set_size_inches(12, 14)
        if num > 25: num = 25
        for i in range(num):
            ax = plt.subplot(5, 5, i + 1)
            ax.imshow(images[start_id + i], cmap="binary")
            if len(predictions) > 0:
                title = 'ai = ' + str(predictions[start_id+i])
                title += (' (o)' if predictions[start_id+i] == labels[start_id + i] else ' (x)')
                title += '\nlabel =' + str(labels[start_id + i])
            else:
                title = 'label =' + str(labels[start_id + i])

            ax.set_title(title, fontsize=12)
            ax.set_xticks([]), ax.set_yticks([])
        plt.show()

    def reshape(self, feature_vector):
        feature_vector = feature_vector.reshape(len(feature_vector), 28, 28).astype('float32')    # 28*28 -> 784
        return feature_vector    # return 1d array

    def normalize(self, feature_vector):
        return feature_vector / 255  # return 像素 between 0~1(因為未來的神經元weights就像機率)

    def one_hot_encoding(self, label):
        return np_utils.to_categorical(label)    # return one-hot encoding in a list


# 假設輸入皆為 rows*cols 的2D圖片，架構為一個RNN Memory System(cols個memory unit)
class model:    # 步驟包含 架構模型(layers)->訓練(training)->測試(testing)
    def __init__(self, rows=28, cols=28, number_per_cell = 256):   # 預設輸入為 28*28 的圖片，隱藏層有256個神經元
        self.model = Sequential()
        self.TIME_STEPS = rows
        self.INPUT_SIZE = cols
        self.UNITS = number_per_cell

    def layers(self):     # numbers用list裝著每層神經元的數目(eg. 輸入層, 隱藏層*2, 輸出層)
        # RNN Memory Unit
        self.model.add(SimpleRNN(input_shape=(self.TIME_STEPS, self.INPUT_SIZE), units=self.UNITS, unroll=True))
        # 拋棄層
        self.model.add(Dropout(0.1))   # 為防止過擬合，拋棄比例為10%
        # 接到輸出層
        self.model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))  # 輸出層


    def training(self, x, y):    # x為data, y為labels
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # loss function為categorical_crossentropy, 優化器為adam, 評估方法為accuracy
        train_history = self.model.fit(x, y, validation_split=0.2, epochs=10, batch_size=200, verbose=2)
        # x,y為設定訓練值特徵值和標籤, verbose是否顯示訓練過程(0不顯示, 1詳盡顯示, 2簡易顯示)

        # summarize history for accuracy
        plt.plot(train_history.history['accuracy'])
        plt.plot(train_history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(train_history.history['loss'])
        plt.plot(train_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    def testing(self, x, y):
        scores = self.model.evaluate(x, y)
        print(f"Accuracy = {scores[1]}, Loss = {scores[0]}")
        prediction = np.argmax(self.model.predict(x), axis=-1)
        #prediction = self.model.predict_classes(x)
        return prediction

    def save_all(self, file_name):    # 儲存model與weights
        self.model.save(file_name)

    def save_weights(self, file_name):   # 只儲存weights
        self.model.save_weights(file_name)

    def load_all(self, file_name):      # 載入model與weights
        self.model = load_model(file_name)

    def load_weights(self, file_name):   # 只載入weights
        self.model.load_weights(file_name)

