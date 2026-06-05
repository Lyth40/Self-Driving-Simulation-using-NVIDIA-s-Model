print('setting up...')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utlis import *

from sklearn.model_selection import train_test_split



### schritt 1 
path = 'my_data'
data = importDataInfo(path)


### schritt 2
data = balanceData(data, display=False)

### schritt 3
imagesPath, steering = loadData(path, data)
# print(imagesPath[0], steering[0])


### schritt 4
X_Train, X_Test, y_Train, y_Test = train_test_split(imagesPath, steering, test_size=0.2, random_state=5)
print('Training Samples: ', len(X_Train))
print('Testing Samples: ', len(X_Test))



### schritt 5
model = createModel()
model.summary()


### schritt 6
print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

history = model.fit(batchGen(X_Train, y_Train, 100, 1), steps_per_epoch=500, epochs=20, validation_data=batchGen(X_Test, y_Test, 100, 0), validation_steps=300) 


### schritt 7

model.save('model.h5')
print('Model saved successfully!')

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.ylim(0, 1)
plt.legend(['Training Loss'],['Validation Loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.show()