from Emoji.EmojiSentiment import get_emoji_sentiments

emojis = [
    ['😀'],
    ['😆', '😂'],
    ['😍', '😘', '😛'],
    ['😥'],
    ['😭', '😓'],
    ['😩'],
    ['👾'],
    ['🐭'],
['😁']
]

scores = get_emoji_sentiments(emojis)

for i in range(len(scores)):
    print(emojis[i], " | ", scores[i])
