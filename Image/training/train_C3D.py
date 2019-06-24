from training.c3d_model import create_c3d_sentiment_model
from ImageSentiment import load_gif_data
import numpy as np
import pathlib
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam


def image_generator(files, batch_size):
    """
    Generate batches of images for training instead of loading all images into memory
    :param files:
    :param batch_size:
    :return:
    """
    while True:
        # Select files (paths/indices) for the batch
        batch_paths = np.random.choice(a=files,
                                       size=batch_size)
        batch_input = []
        batch_output = []

        # Read in each input, perform preprocessing and get labels
        for input_path in batch_paths:
            input = load_gif_data(input_path)
            if "pos" in input_path:  # if file name contains pos
                output = np.array([1, 0])  # label
            elif "neg" in input_path:  # if file name contains neg
                output = np.array([0, 1])  # label

            batch_input += [input]
            batch_output += [output]
        # Return a tuple of (input,output) to feed the network
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)

        yield (batch_x, batch_y)


model = create_c3d_sentiment_model()
print(model.summary())
model.load_weights('models/C3D_Sport1M_weights.h5', by_name=True)

for layer in model.layers[:14]:  # freeze top layers as feature extractor
    layer.trainable = False
for layer in model.layers[14:]:  # fine tune final layers
    layer.trainable = True

train_files = [str(filepath.absolute()) for filepath in pathlib.Path('data/train').glob('**/*')]
val_files = [str(filepath.absolute()) for filepath in pathlib.Path('data/validation').glob('**/*')]

batch_size = 16
train_generator = image_generator(train_files, batch_size)
validation_generator = image_generator(val_files, batch_size)

model.compile(optimizer=Adam(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

mc = ModelCheckpoint('epoch-{epoch:02d}-val_loss-{:.2f}-val_acc-{val_acc:.2f}.hdf5',
                     monitor='val_loss', mode='auto', verbose=1, save_best_only=True)

es = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

history = model.fit_generator(train_generator, validation_data=validation_generator,
                              steps_per_epoch=int(np.ceil(len(train_files) / batch_size)),
                              validation_steps=int(np.ceil(len(val_files) / batch_size)), epochs=100, shuffle=True,
                              callbacks=[mc, es])
