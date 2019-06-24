from SentimentAnalysis import load_models, get_sentiments
from sklearn.metrics import confusion_matrix, classification_report

sentences = [
    'we lost 😒 😅 😛 <img>https://media.giphy.com/media/2rtQMJvhzOnRe/giphy.gif</img>',
    'wow you are so funny 👮 <img>https://media.giphy.com/media/13mPGyNp9otvyg/giphy.gif</img>',
    'you suck 😂',
    'hell yeah <img>https://media.giphy.com/media/5GoVLqeAOo6PK/giphy.gif</img>',
    '<img>https://media.giphy.com/media/9Y5BbDSkSTiY8/giphy.gif</img>',
    '<img>https://media.giphy.com/media/l1KVaj5UcbHwrBMqI/giphy.gif</img>',
    '<img>https://media1.giphy.com/media/a9xhxAxaqOfQs/giphy.gif</img>',
]

# obtained from https://www.kaggle.com/rexhaif/emojifydata-en
evaluation = {
    "I get in my moods where I don’t be wanting to talk and that’s when everybody got a question 😤": 0,
    "THANKS GOD FOR WAKING ME UP TODAY. 🙏👆": 1,
    "i love this sweatshirt so much :,) 🤗💄💙🧡": 1,
    "It’s Official This Weekend I’m Going To Knoxville Tennessee !!! 🧗🏽‍♂️": 1,
    "3/4 of the team is 🗑": 0,
    "literally ovr math 🙄": 0,
    "OH MYY GOSHHH😩😍💙💙": 1,
    "Nice to meet you, Jimu! 😆 Yay I really looking forward for your next concert experience 😉😉 Have fun! 🙌😆": 1,
    "I can't stand it's not that funny kind of people, like okay I'm sorry with your amargada no sense of humor having ass 😒": 0,
    "Cavs look terrible Lebron ain’t making it out the first round with these niggas🤦🏿‍♂️": 0,
    "It was funny when we were like 12😭": 0,
    "That’s your headline? 🙄How about crazed man throws car in reverse &amp; hits patrol car. Assault on officers…": 0,
    "I thought it looked familiar 😉": 1,
    "BEAUTIFUL 😍😍 congratulations!!": 1,
    "This episode 🔥": 1,
    "can school be done already?🙄": 0,
    "God has brought me such a loooong way and he’s nowhere near done ❤️ thank you Lord!!": 1,
    "Okaaaay. I don’t mean to be head ass but I mean to be head ass. 🤧💖💖 and thank you bb 😘": 1,
    "EPIC. You actually look really pretty w/the hairstyle &amp; earrings 😩😂   Twinning 🙌🏽🙌🏽": 1,
    "I don’t care who drops what or when! We’re climbing the chart or we’re going down swinging HARD! 😈🔥🤘🏽": 1,
    "You remind me of a cupcake. Cute and sweet. 😊💜 with an adorably fitting gif to go with it 😍💜": 1,
    "I took an amazing nap 🤗": 1,
    "Yes I luv u too 💕": 1,
    "Happiest birthday, ! God bless you always! 🙏💛  CTTO": 1,
    "Wow RIP 😥": 0,
    "I should be packing for my house sitting job.....but I don’t want tooooo. 😩": 0,
    "They’re signed to the same label &amp; both have ghost writers. 🤷🏾‍♂️": 0,
    "THIS IS GOLD 😂😂😂": 1,
    "New blessings are coming your way 😇": 1,
    "I want a nigga geeked about me.😍😍😍😍🤪": 1,
    "Finding both love and friendship in the same person &gt;&gt;&gt; 😍😭": 1,
    "Right!! Cuz everyone knows he got those medals at the Dollar Store! 😑": 0,
    "Damn coach pop’s wife died 😭🤧 I couldn’t imagine what’s he’s feeling rn": 0,
    "HELLL YESSS QUEEN😍😍": 1,
    "I've felt this kinda pain before...😭😔💔": 0,
    "😢 rt if you cried.": 0,
    "I just love her 😊❤️": 1,
    "Chaela just told me i can’t wear a sweat suit to her baby shower 🙄🙄 ok": 0,
    "Cool♥": 1,
    "i just 😪": 0,
    "Could watch still game aww day, unreal😂": 1,
    "Thank you for putting me on your Insta Story!!  💙💙💙": 1,
    "American🇺🇸 guys hate my face": 0,
    "Stop recycling old skins pls 🖕🏽": 0,
    "🤦‍♂️....🤫 Says the guy who delayed our first fight by 5 months because of injury. 😂🤷‍♂️": 0,
    "Happy 28th birthday to you, 🎈": 1,
    "YALL WRONG FOR THIS 😂😂": 1,
    "Dude 😂😂😂": 1,
    "My dainty friends don't like beef 🤦🏿‍♂️🤦🏿‍♂️": 0,
    "Is it men that are trash or is it your perception of men that is trash? 🤔": 0,
    "Love Tennessee ♥️": 1,
    "Thanks same to you 🌷":
        1,
    "Y'all be so consumed in everyone else's business, but y'all forget to tend to your own. That shit baffles me. 🤦🏽‍♀️😭": 0,
    "My favorite person ever❤️": 1,
    "LMFAOOO YOOO IT ALL MAKES SENSE NOW😭😭😭": 0,
    "I made it to my interview 7 minutes early. I'm proud of myself 🤗": 1,
    "I still respect your opinion 😉": 1,
    "When women support other women, incredible things happen🙌💗": 1,
    "I’m always here for you too!!!!! Ilysmm! 💕☺️💕": 1,
    "they're at a .30 💀": 0,
    "Oh my word this gauche wee turd 😔": 0,
    "Life is amazing when you're surrounded by good people and positive vibes ☺🌴": 1,
    "Stop it 😁": 1,
    "He should leave USA 🇺🇸 stupid unprofessional bonehead!": 0,
    "Yea this how I know I’m grown now ... too much sweets like this makes my stomach hurt just watching it 🤢 …": 0,
    "The more immoral Trump becomes, the more white Evangelicals approve. Hmmm 🤔": 0,
    "Need everyone to be on this level when they see me 😂❤️": 1,
    "Amazing picture 💓": 1,
    "She good. She don’t need no help ✌🏾": 1,
    "Still can't get enough of his accent at 9:30 😍": 1,
    "The fact that this was a true story was sad af.😢": 0,
    "She’s So Adorable ㅠㅠㅠㅠ✨": 1,
    "This shit too clean 😩": 1,
    "ION WANNA SEE NONE OF YALL TWEET HIS LYRICS WHEN YALL WAS CALLIN HIM TRASH FOR YEARS 😠😠😠😠": 0,
    "Hey, guess who officially a Flight Attendant 💅🏾": 1,
    "🎤...and I know without a shadow of a fuxking doubt,you always meant so well...🎤 Presbyteria 😭😭😭": 1,
    "OMG! We LOVE your Memes. 😇🌟🌟🌟🌟🌟 $OOT $KMD": 1,
    "Got a KFC and I’ve actually not been this full in such a long time 😍😍🤤🤤": 1,
    "Feel like dude treats all his hoes like princesses. Then there’s me, the queen, treated like a side piece 🍟": 0,
    "I got the job 😁": 1,
    "Very sad to hear about  RIP 😢": 0,
    "I definitely learnt my lesson on that one 😩🤬": 0,
    "That looks ugly and sooooo not very tasty 😱😱😱": 0,
    "😑 foo Ian made not one joke today you jus be trippin": 0,
    "Social media is so bad for the mind n soul. Seriously thinking about deleting it😴👋🏽": 0,
    "Bitch slapped the blunt out of my mouth, She done fucked up I’m cooking her ass tonight 😤😂": 0,
    "This just adds fuel to the “ niggas ain’t shit “ saying 🤦🏽‍♂️": 0,
    "I go back to work tonight and I dread it so much. 🙄": 0,
    "It’s because we are counting down until graduation😭 it’s torture": 0,
    "gotta love an infected piercing 🤕.": 0,
    "Mentally exhausted 🤕": 0,
    "I am SO scared of birds🤧": 0,
    "i got you bby❤️😉": 1,
    "Denzel is the Goat for this😂😂": 1,
    "I’m going to hell and Rosas following 💀💀": 0,
    "Studying Midwifery😣😓": 0,
    "i love it when it rain outside 😤": 0,
    "wow you are so amazing lol 🤬": 0,
    "wow that move was 💩": 0,
    "that is disgusting 🤮": 0
}


def test_model():
    """
    Print predictions for test media containing emojis, text and images
    :return: none
    """
    image_model, text_model_ensemble = load_models()
    sentiments = get_sentiments(sentences, image_model, text_model_ensemble)
    for i in range(len(sentences)):
        print(sentences[i], " | ", sentiments[i])


def evaluate_model():
    """
    Evaluate model performance on emoji and text combinations from Twitter
    :return:
    """
    image_model, text_model_ensemble = load_models()
    sentiments = get_sentiments(evaluation, image_model, text_model_ensemble)

    predicted = []
    actual = []

    for k, v in evaluation.items():
        actual.append(v)

    for sentiment in sentiments:
        if sentiment > 0:
            predicted.append(1)
        elif sentiment < 0:
            predicted.append(0)

    confusion_matrix(actual, predicted)
    print("\n", "Confusion Matrix: \n", confusion_matrix(actual, predicted))
    print("\n", "Classification Report: \n", classification_report(actual, predicted))


test_model()
