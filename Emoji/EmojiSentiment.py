import emoji as emoji_library
from Emoji.config import emoji2sentiment


def get_emojis_in_sentence(sentence):
    """
    Return list of emojis in sentence
    :param sentence:
    :return: emojis
    """
    emojis = [c for c in sentence if c in emoji_library.UNICODE_EMOJI]
    return emojis


def get_emoji_sentiments(emojis_list):
    """
    Get average sentiment of emojis given a list containing lists of emojis from each sentence
    :param emojis_list:
    :return: average_sentiment_list
    """

    sentiment_lists = [[emoji2sentiment.get(emoji) for emoji in emoji_list] for emoji_list in emojis_list]

    average_sentiment_list = []
    for sentiment_list in sentiment_lists:
        sentiment_list = [val for val in sentiment_list if val is not None]  # remove None values
        if sentiment_list:
            average_sentiment_list.append(sum(sentiment_list) / len(sentiment_list))
        else:
            average_sentiment_list.append(None)

    return average_sentiment_list
