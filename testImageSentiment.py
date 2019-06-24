from ImageSentiment import load_c3d_sentiment_model, get_gifs_sentiment, download_gifs
from sklearn.metrics import confusion_matrix, classification_report
import pathlib
from natsort import natsorted

urls = [
    'https://media.giphy.com/media/11sBLVxNs7v6WA/giphy.gif',
    'https://media.giphy.com/media/5GoVLqeAOo6PK/giphy.gif',
    'https://media.giphy.com/media/YJ5OlVLZ2QNl6/giphy.gif',
    'https://media.giphy.com/media/xT5LMHxhOfscxPfIfm/giphy.gif',
    'https://media.giphy.com/media/8WJw9kAG3wonu/giphy.gif',
    'https://media.giphy.com/media/P8MxmGnjmytws/giphy.gif',
    'https://media.giphy.com/media/inyqrgp9o3NUA/giphy.gif',
    'https://media.giphy.com/media/WUq1cg9K7uzHa/giphy.gif',
    'https://media.giphy.com/media/3o9bJX4O9ShW1L32eY/giphy.gif',
    'https://media.giphy.com/media/TseBjMu53JgWc/giphy.gif',
    'https://media.giphy.com/media/ZebTmyvw85gnm/giphy.gif',
    'https://media.giphy.com/media/3t7RAFhu75Wwg/giphy.gif',
    'https://media.giphy.com/media/p8Uw3hzdAE2dO/giphy.gif',
    'https://media.giphy.com/media/ntjBjvfnakKJ2/giphy.gif',
    'https://media.giphy.com/media/OOezqqxPB8aJ2/giphy.gif',
    'https://media.giphy.com/media/l3V0H7bYv5Ml5TOfu/giphy.gif'
]


def test_model():
    """
    Test model output on a list of gif links
    :return:
    """
    model = load_c3d_sentiment_model()
    gif_paths = download_gifs(urls, "test_downloads")
    scores = get_gifs_sentiment(gif_paths, model)

    for i in range(len(scores)):
        print(gif_paths[i], "|", scores[i])


def evaluate_model():
    """
    Evaluate the performance of the model on evaluation data
    :return:
    """
    model = load_c3d_sentiment_model()
    gif_paths = [str(filepath.absolute()) for filepath in
                 pathlib.Path("Image/training/data/validation/").glob('**/*')]
    gif_paths = natsorted(gif_paths)

    predicted = []
    actual = []
    currentFile = 1
    top_pos = {}
    top_neg = {}

    print("\nEvaluating Images...\n")
    for path in gif_paths:  # process one at a time
        print(currentFile, "/", len(gif_paths))
        sentiment = get_gifs_sentiment([path], model)

        if "pos" in path:
            actual.append(1)
        elif "neg" in path:
            actual.append(0)

        if sentiment[0] > 0:
            predicted.append(1)
        elif sentiment[0] < 0:
            predicted.append(0)

        if sentiment[0] > 0.95:
            top_pos[path] = sentiment[0]
        elif sentiment[0] < -0.95:
            top_neg[path] = sentiment[0]

        currentFile = currentFile + 1

    confusion_matrix(actual, predicted)
    print("\n", "Confusion Matrix: \n", confusion_matrix(actual, predicted))
    print("\n", "Classification Report: \n", classification_report(actual, predicted))

    print("\nTop Positive\n")
    for path, val in top_pos.items():
        print(path, "|", val)

    print("\nTop Negative\n")
    for path, val in top_neg.items():
        print(path, "|", val)


test_model()
