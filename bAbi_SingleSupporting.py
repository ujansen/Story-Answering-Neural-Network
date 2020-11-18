from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn import metrics
from functools import reduce
import pickle
import tarfile
import numpy as np
import re
import time
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences
    that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data

def get_stories(f, only_supporting = False, max_length = None):
    '''Given a file name, read the file,
    retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting = only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data):
    inputs, queries, answers = [], [], []
    for story, query, answer in data:
        inputs.append([word_idx[w] for w in story])
        queries.append([word_idx[w] for w in query])
        answers.append(word_idx[answer])
    return (pad_sequences(inputs, maxlen = story_maxlen),
            pad_sequences(queries, maxlen = query_maxlen),
            np.array(answers))



try:
    path = get_file('babi-tasks-v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
except:
    print('Error downloading dataset, please download it manually:\n'
          '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
          '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
    raise
tar = tarfile.open(path)

challenges = {
    # QA1 with 10,000 samples
    'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
    # QA2 with 10,000 samples
    'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
}
challenge_type = 'single_supporting_fact_10k'
challenge = challenges[challenge_type]

train_stories = get_stories(tar.extractfile(challenge.format('train')))
test_stories = get_stories(tar.extractfile(challenge.format('test')))


# See what the data looks like

"""for i in range(15):
    print("Story: {}".format(' '.join(train_stories[i][0])))
    print("Query: {}".format(' '.join(train_stories[i][1])))
    print("Answer: {}".format(train_stories[i][2]))
    print("---")"""
    
vocab = set()
for story, q, answer in train_stories + test_stories:
    vocab |= set(story + q + [answer])
vocab = sorted(vocab)

# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
inputs_train, queries_train, answers_train = vectorize_stories(train_stories)
inputs_test, queries_test, answers_test = vectorize_stories(test_stories)

try:
    model = load_model("query_model.h5")    
    vocab = pickle.load(open("vocab.pkl", "rb"))
except:
# placeholders
    input_sequence = Input((story_maxlen,))
    question = Input((query_maxlen,))
    
    # encoders
    # embed the input sequence into a sequence of vectors
    input_encoder_m = Sequential()
    input_encoder_m.add(Embedding(input_dim = vocab_size,
                                  output_dim = 64))
    input_encoder_m.add(Dropout(0.3))
    # output: (samples, story_maxlen, embedding_dim)
    
    # embed the input into a sequence of vectors of size query_maxlen
    input_encoder_c = Sequential()
    input_encoder_c.add(Embedding(input_dim = vocab_size,
                                  output_dim = query_maxlen))
    input_encoder_c.add(Dropout(0.3))
    # output: (samples, story_maxlen, query_maxlen)
    
    # embed the question into a sequence of vectors
    question_encoder = Sequential()
    question_encoder.add(Embedding(input_dim = vocab_size,
                                   output_dim = 64,
                                   input_length = query_maxlen))
    question_encoder.add(Dropout(0.3))
    # output: (samples, query_maxlen, embedding_dim)
    
    # encode input sequence and questions (which are indices)
    # to sequences of dense vectors
    input_encoded_m = input_encoder_m(input_sequence)
    input_encoded_c = input_encoder_c(input_sequence)
    question_encoded = question_encoder(question)
    
    # compute a 'match' between the first input vector sequence
    # and the question vector sequence
    # shape: `(samples, story_maxlen, query_maxlen)`
    match = dot([input_encoded_m, question_encoded], axes = (2, 2))
    match = Activation('softmax')(match)
    
    # add the match matrix with the second input vector sequence
    response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
    response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)
    
    # concatenate the match matrix with the question vector sequence
    answer = concatenate([response, question_encoded])
    
    # the original paper uses a matrix multiplication for this reduction step.
    # we choose to use a RNN instead.  
    answer = LSTM(32)(answer) # (samples, 32)
    # one regularization layer -- more would probably be needed.
    answer = Dropout(0.3)(answer)
    answer = Dense(vocab_size)(answer)  # (samples, vocab_size)
    # we output a probability distribution over the vocabulary
    answer = Activation('softmax')(answer)
    
    # build the final model
    model = Model([input_sequence, question], answer)
    model.compile(optimizer = 'rmsprop', loss = 'sparse_categorical_crossentropy',
                  metrics = ['accuracy'])
    print("Done.\n")
    model.summary()
    
    start_time = time.time()
    # train
    model.fit([inputs_train, queries_train], answers_train,
              batch_size = 32,
              epochs = 150,
              validation_data = ([inputs_test, queries_test], answers_test))
    
    # save
    # save entire network to HDF5 (save everything, suggested)
    model.save("query_model.h5")
    # save the vocab too, indexes must be the same
    pickle.dump(vocab, open("vocab.pkl", "wb"))
    
    elapsed_time = time.time() - start_time
    print("Elapsed time: {}".format(hms_string(elapsed_time)))
        
    pred = model.predict([inputs_test, queries_test])
    # See what the predictions look like, they are just probabilities of each class.
    print(pred)
    
    pred = np.argmax(pred, axis = 1)
    print(pred)
    
    score = metrics.accuracy_score(answers_test, pred)
    print("Final accuracy: {}".format(score))
  
def single_predict(story, query):
    #print("Remember, I only know these words: {}".format(vocab))
    print()
    check = True
    check_name = True
    name = ''
    absent = ''
    for word in tokenize(story):
        if word not in vocab:
            check = False
            absent = word
            break
    if check:
        for word in tokenize(query):
            if word != '?' and word != '.' and word != 'Where' and word != 'is' and word not in tokenize(story):
                name = word
                check_name = False
        if check_name:
            adhoc_stories = (tokenize(story), tokenize(query), '?')
                
            adhoc_train, adhoc_query, adhoc_answer = vectorize_stories([adhoc_stories])
                
            pred = model.predict([adhoc_train, adhoc_query])
            #print(pred[0])
            pred = np.argmax(pred, axis = 1)
            #print("Answer: {}({})".format(vocab[pred[0]-1], pred))
            return vocab[pred[0]-1]
        else:
            return (name+ ' is not in the story. Please try again.')
    else:
        return (absent+' is not in the vocabulary. Please try again')
    
#story = "Mary went to the hallway. Mary went to the bathroom. Daniel went to the bedroom."
#query = "Where is Mary?"
#single_predict(story, query)
