from captcha.image import ImageCaptcha
from keras.callbacks import TensorBoard
import numpy as np
import random
from keras.models import *
from keras.layers import *
import string

characters = string.digits + string.ascii_uppercase

width, height, n_len, n_class = 170, 80, 4, len(characters)

generator = ImageCaptcha(width=width, height=height)

random_str = ''.join([random.choice(characters) for j in range(4)])

img = generator.generate_image(random_str)

def gen(batch_size=10000):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    for i in range(batch_size):
        random_str = ''.join([random.choice(characters) for j in range(4)])
        X[i] = generator.generate_image(random_str)
        j=0
        for ch in random_str:
            y[j][i, :] = 0
            y[j][i, characters.find(ch)] = 1
            j=j+1
    return X, y

def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])


n_class = 36

input_tensor = Input((height, width, 3))
x = input_tensor
x = BatchNormalization(axis=1)(x)
for i in range(4):
    x = Conv2D(32*(2)**(i), (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization(axis=1)(x)

x = Flatten()(x)
x = Dropout(0.5)(x)
x = BatchNormalization(axis=-1)(x)
x = [Dense(64, activation='relu', name='c%d'%(i+1))(x) for i in range(4)]
x = [Dense(n_class, activation='softmax', name='y'+str(s))(i) for i,s in zip(x,range(4))]
model = Model(inputs=input_tensor, outputs=x)

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
data_x,data_y = gen()
model.fit(data_x,data_y,batch_size=100,epochs=5,
            callbacks=[TensorBoard(log_dir='./tmp/log')],
            validation_split=0.1)
