from __future__ import print_function
import json
from deepmoji.model_def import deepmoji_transfer
from deepmoji.global_variables import PRETRAINED_PATH
from deepmoji.finetuning import (
     load_non_benchmark,
     finetune)
from deepmoji.sentence_tokenizer import SentenceTokenizer
from build import nps_chat_2_class

nb_classes = 2

with open('../model/vocabulary.json', 'r') as f:
    vocab = json.load(f)
st = SentenceTokenizer(vocab, 30)

alg_data = {
    'info': nps_chat_2_class.INFO,
    'texts': nps_chat_2_class.TEXTS
}

data = load_non_benchmark(alg_data, vocab)

# Set up model and finetune
model = deepmoji_transfer(nb_classes, data['maxlen'], PRETRAINED_PATH)
model.summary()
model, acc = finetune(model, data['texts'], data['labels'], nb_classes,
                      data['batch_size'], method='chain-thaw')
print('Acc: {}'.format(acc))
