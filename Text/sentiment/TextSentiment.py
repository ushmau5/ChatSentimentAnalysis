from keras.models import load_model
import json
from deepmoji.attlayer import AttentionWeightedAverage
from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.global_variables import VOCAB_PATH


def modify_range(val):
    """
    Modify value from range 0,1 -> -1,1 and preserve ratio
    :param val:
    :return: value in rage -1,1
    """
    return (val * 2) - 1


def load_finetuned_models():
    """
    Load finetuned Keras models
    :return: [twitter_model, youtube_model]
    """
    twitter_model = load_model('Text/sentiment/finetuned/twitter_ss.hdf5',
                               custom_objects={'AttentionWeightedAverage': AttentionWeightedAverage})
    youtube_model = load_model('Text/sentiment/finetuned/youtube_ss.hdf5',
                               custom_objects={'AttentionWeightedAverage': AttentionWeightedAverage})

    return [twitter_model, youtube_model]


def get_texts_sentiment(texts, model_ensemble):
    """
    Get sentiment scores for list of texts
    :param texts:
    :param model_ensemble:
    :return: average_sentiment_prediction
    """
    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)

    twitter_maxlen = 30
    youtube_maxlen = 30

    twitter_st = SentenceTokenizer(vocabulary, twitter_maxlen)
    youtube_st = SentenceTokenizer(vocabulary, youtube_maxlen)

    twitter_tokenized, _, _ = twitter_st.tokenize_sentences(texts)
    youtube_tokenized, _, _ = youtube_st.tokenize_sentences(texts)

    twitter_predictions = model_ensemble[0].predict(twitter_tokenized)
    youtube_predictions = model_ensemble[1].predict(youtube_tokenized)

    average_predictions = (twitter_predictions + youtube_predictions) / 2
    average_sentiment_prediction = [modify_range(prediction)[0] for prediction in average_predictions]

    return average_sentiment_prediction
