import numpy as np
from imgpy import Img
from PIL import ImageFile
from keras.models import load_model
import urllib.request as urllib
import pathlib
from tqdm import tqdm
from natsort import natsorted
from shutil import rmtree
from os import mkdir
from math import floor

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_gif_data(file_path):
    """
    Load and process gif for input into Keras model
    :param file_path:
    :return: Mean normalised image in BGR format as numpy array
             for more info see -> http://cs231n.github.io/neural-networks-2/
    """
    im = Img(fp=file_path)
    try:
        im.load()
    except:
        print("Error loading image: " + file_path)
        return
    im.resize(size=(112, 112))
    im.convert('RGB')
    im.close()

    np_frames = []
    frame_index = 0
    # need 16 evenly divided frames, if gif is 32 frames take every 2nd frame etc..
    multiplier = floor(im.frame_count / 16)
    if multiplier < 1:
        multiplier = 1
    for i in range(16):  # if image is less than 16 frames, repeat the frames until there are 16
        frame = im.frames[frame_index]
        rgb = np.array(frame)
        bgr = rgb[..., ::-1]
        mean = np.mean(bgr, axis=0)
        np_frames.append(bgr - mean)  # C3D model was originally trained on BGR, mean normalised images
        # it is important that unseen images are in the same format
        if (im.frame_count - frame_index) <= multiplier:
            frame_index = 0
        else:
            frame_index = frame_index + multiplier

    return np.array(np_frames)


def load_c3d_sentiment_model():
    """
    Load saved Keras model
    :return: Keras model
    """
    model = load_model('Image/c3d_sentiment.hdf5')

    return model


def download_gifs(image_urls, path):
    """
    Download gifs given a list of urls
    :param image_urls:
    :param path:
    :return:
    """
    rmtree(path)  # delete folder and contents from previous runs
    mkdir(path)  # make new folder

    print("Downloading images...\n")
    for i in tqdm(range(len(image_urls))):  # tqdm is a download progress meter
        urllib.urlretrieve(image_urls[i],
                           path + "/" + str(i) + '.gif')

    gif_paths = [str(filepath.absolute()) for filepath in
                 pathlib.Path(path + "/").glob('**/*')]

    sorted_gif_paths = natsorted(gif_paths)  # sort paths in natural alphabetical order so results are printed in order

    return sorted_gif_paths


def get_gifs_sentiment(gif_paths, model):
    """
    Get sentiment score for gif using Keras model
    :param gif_paths: list of gif filepaths
    :param model: C3D sentiment model
    :return: sentiment score in range -1, 1 | (very negative, very positive)
    """
    images = np.array([load_gif_data(gif_path) for gif_path in gif_paths])
    predictions = model.predict(images)
    sentiment_scores = [(prediction[0] - prediction[1]) for prediction in predictions]
    # prediction[0] - prediction[1] | positive probability - negative probability
    return sentiment_scores
