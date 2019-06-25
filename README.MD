# ChatSentimentAnalysis
![main](https://i.imgur.com/zyKgSBZ.png)

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org) 
![Python](https://img.shields.io/badge/python-3.6-orange.svg)

> Sentiment analysis of chat data including text, smileys, emojis and images (gifs) with the included difficuly of sarcasm.

> Emoji sentiment using [Emoji Sentiment Ranking](http://kt.ijs.si/data/Emoji_sentiment_ranking/).

> Image sentiment using [C3D](https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2), a 3D-CNN.

> Text sentiment using [DeepMoji](https://github.com/bfelbo/DeepMoji) finetuned on [SS-YouTube and SS-Twitter](https://pdfs.semanticscholar.org/ec1d/92b3b01421e4f49da52fbcf96b9a9337002b.pdf).

---
## Table of Contents
- [Installation](#installation)
- [SentimentAnalysis.py](#SentimentAnalysis.py)
- [Emoji](#Emoji)
- [Image](#Image)
- [Text](#team)
- [Performance](#performance)
- [FAQ](#faq)
- [Support](#support)
- [Additional Notes](#additionalnotes)


---


## Installation
- Download [DeepMoji SS-Twitter Keras Model](https://drive.google.com/open?id=1ZLD2iSgl4PggYPAG9iG_eoFx2DEegqSL) and [DeepMoji SS-YouTube Keras Model](https://drive.google.com/open?id=1iWc0sUmk7FwXBFPQ8FOIWKhJ8jS2iplh) and place them in ```SentimentAnalysis/Text/sentiment/finetuned```.

- Download [DeepMoji Weights](https://www.dropbox.com/s/xqarafsl6a8f9ny/deepmoji_weights.hdf5?dl=0) and place them in ```SentimentAnalysis/Text/model```.

- Download [C3D Sentiment Model](https://drive.google.com/open?id=1UeEsQYrItUF0NOpD1frvS8qxdqGCUTg9) and place in ```SentimentAnalysis/Image```.

## SentimentAnalysis.py
- See testSentimentAnalysis.py for an example.
- Main implementation of sentiment analysis.

## Emoji
![emoji](https://i.imgur.com/P2tMRtJ.png)
- See testEmojiSentiment.py for an example.
- EmojiSentiment.py: Extract sentiment from emojis.
- config.py: Contains emoji to sentiment mappings.
- build: Build files used to generate emoji sentiment mappings.

## Image
- See testImageSentiment.py for an example.
- ImageSentiment.py: Extract sentiment from images(gifs).
- training: Files related to training the classification model.

## Text
![text](https://imgur.com/oXE5H2W.png)
- See testTextSentiment.py for an example.
- Contains modified version of [DeepMoji Python 3](https://github.com/zzsza/DeepMoji-Python3) repo.
- sentiment/TextSentiment.py: Extract sentiment from text and smileys.
- sentiment/build: Example of manualy entered training data for finetuning.
- sentiment/finetuned: Finetuned Keras models used for classification. 

## Performance
 - Performance of the model was tested on 100 tweets containing emojis from [this](https://www.kaggle.com/rexhaif/emojifydata-en) dataset.
 - [This paper](https://www.sciencedirect.com/science/article/pii/S0950705117304240) showed that emoticon blocking (using emoji sentiment as overall sentiment indicator for a sentence) proved to be an effective method of sentiment detection. 
 - This was tested and the results can be observed below.
### Emoticon Blocking
![emoticon-blocking](https://i.imgur.com/Sm3NUnQ.png)
 
 ### Text Only
 ![text-first](https://i.imgur.com/Uj7Bly0.png)

- Emoticon blocking appears to perform better on this small dataset which would suggest it would also perform better on a larger dataset. 
- Emoticon blocking also handles sarcasm where, for example, ```I hate it when you do that 😉``` is considered positive overall, where as it would be classified as negative if only the text was considered.

### C3D Image Sentiment Model
![accuracy](https://i.imgur.com/7Nx6BNi.png)
- 5000 images used in training with 2500 of each class.
- Model trained and evaluated on balanced data set using a training/validation split of 70/30.
- Below is an example of the top negative and positive images from the validation data.

#### Top Negative
![neg1](https://i.imgur.com/WR0C6Qq.gif) ![neg2](https://i.imgur.com/ios7EqN.gif) ![neg3](https://i.imgur.com/vGWhWhS.gif) ![neg4](https://i.imgur.com/yauDFtM.gif) ![neg5](https://i.imgur.com/s275CbK.gif) ![neg6](https://i.imgur.com/DqI48vB.gif) ![neg7](https://i.imgur.com/OLffiZ5.gif) ![neg8](https://i.imgur.com/jkOkrUo.gif)



#### Top Positive
![1pos](https://i.imgur.com/6D351qa.gif) ![pos2](https://i.imgur.com/qywFKc8.gif) ![pos3](https://i.imgur.com/g8Ln6bj.gif) ![pos4](https://i.imgur.com/4KCDcMe.gif) ![pos5](https://i.imgur.com/26KSllL.gif) ![pos6](https://i.imgur.com/tbuNFFS.gif) ![pos7](https://i.imgur.com/keGjqkC.gif) ![pos8](https://i.imgur.com/d7srKWA.gif)

## FAQ

- **How is sentiment calculated?**
	- Text, emojis and images are extracted from a chat sentence like the example given below.
    - ```we lost 😒 😅 😛 <img>https://media.giphy.com/media/2rtQMJvhzOnRe/giphy.gif</img>```
	- Sentiment for each is calculated and a score is returned based on the rules in SentimentAnalysis.calculate_scores().

- **How was the image model trained?**
	- The [C3D Model](https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2) was finetuned using imges from GIPHY.
	- Labelled GIPHY images were obtained from [GIFGIF](http://gifgif.media.mit.edu/labs/api).
	- An R script to extract the  links from JSON has been included in this project. 
	- Text files are generated containing links to the images which can be downloaded from terminal using ```cat file_name.txt | parallel --gnu "wget {}"```.
	- Or the exact files used for training and validation can be downloaded [here](https://drive.google.com/open?id=117xkFzY2OiYMjf6PTPsKYSXENXnPIZVR).

- **Why is there no sentiment score for some emojis?**
	- In my opinion, not all emojis are good indicators of sentiment. 
	- Only emojis with obvious indicators of sentiment such as facial expressions, popular symbols and hand signs were used. 

- **But what about sarcasm detection?**
	- [*DeepMoji has learned to understand emotions and sarcasm based on millions of emojis*](https://deepmoji.mit.edu/).
	- Whether the text contains sarcasm or not is irrelevant, the features extracted using DeepMoji still accurately represent the emotions in the text. These features are used when finetuning new models. 
	- Using emoticon blocking also helps to calculate the actual sentiment in cases of sarcasm e.g. ```I hate it when you do that 😉``` is actually positive and contains a positive emoji but negative text.  Emoticon blocking considers the emoji as the overall sentiment which would be correct in this case.

- **Why was the model only evaluated on combinations of emojis and text?**
	- The sentiment of images is only used if there is no sentiment available for emojis or text in a sentence. The accuracy of this is the same as when the image model was evaluated individually.

---

## Support
Email: `oisin097@hotmail.com`

---


## Additional Notes
- Sentiment analysis from sarcasm detection could be directly tackled using DeepMoji. DeepMoji  outlined in their paper to have been finetuned on [SCv2-GEN](https://arxiv.org/pdf/1709.05404.pdf) to perform sarcasm detection at 75% accuracy. Text could be first passed through this model to detect sarcasm and then the sarcastic sentences passed to another model finetuned on sentiment labelled sentences containing sarcasm. 

- In my opinion, this approach would not be very accurate as the compounding decrease in classification accuracy (is this sarcastic? -> is this positive/negative?) (75% x ??%) would most likely result in a poor result. There is also an increase in computational overhead which enforces the fact this trade off would most likely not be worthwhile. 


