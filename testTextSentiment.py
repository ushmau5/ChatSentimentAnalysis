from Text.sentiment.TextSentiment import load_finetuned_models, get_texts_sentiment

sentences = [
    'dude what are you doing?',
    'i cant believe she did that',
    'that guy is such a pain',
    'he is definitely gonna lose',
    'she is such a joke',
    'well done mate good effort',
    'you are such a hero haha',
    'thats sick man',
    'you made a laugh of them',
    'lol so funny',
    'i hate you',
    'i hate you :)',
    "my flight is delayed... amazing :(",
    "i think you left your talent at home",
    "you love hurting me, huh?",
    "Are you always this annoying or are you exerting extra effort today?",
    "Not the brightest crayon in the box now, are we?",
    "What planet did you come from?",
    "That speaker was so interesting that I barely needed to drink my third cup of coffee.",
    "It’s so thoughtful for the teacher to give us all this homework right before Spring Break.",
]

model_ensemble = load_finetuned_models()
scores = get_texts_sentiment(sentences, model_ensemble)

for i in range(len(scores)):
    print(sentences[i], "|", scores[i])
