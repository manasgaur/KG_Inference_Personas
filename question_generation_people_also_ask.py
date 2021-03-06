# pip install people_also_ask
# pip install spacy
# python -m spacy download en_core_web_lg

from tqdm import tqdm
import people_also_ask as paa
import json
from scipy import spatial
import pandas as pd
import time
import spacy

nlp = spacy.load('en_core_web_lg')
emotion_vector = nlp('emotion')

def read_sentences_VDIL():
    with open('sentences.json', encoding='utf-8') as lst:
        dataset = json.load(lst)
    new_dataset = dict()
    for key, value in tqdm(dataset.items()):
        words = key.split(':')
        words = ', '.join(w for w in words)
        new_dataset[words]= value
    return new_dataset

def similarity(vec1, vec2):
    return round(1-spatial.distance.cosine(vec1,vec2),2)



new_dataset = read_sentences_VDIL()
test_dataset = dict() 
for k, v in new_dataset.items():
    sentences=""
    for sentence in v:
        sentences+=" " + sentence
    test_dataset[k] = sentences
test_df = pd.DataFrame(test_dataset.items(), columns=['description', 'context'])
test_df.description = test_df.description.str.lower()
test_df.context = test_df.context.str.lower()
emotion_store = []
for idx, row in test_df.iterrows():
    words = row['description'].split(',')
    words = [w.lstrip() for w in words ]
    selected_idx = list(set([idx for w in words if (nlp(w).similarity(emotion_vector)>0.45)]))
    if len(selected_idx) == 0:
        continue
    emotion_store.append({'tags': list(test_df.iloc[selected_idx]['description'].values)})

emotion_context_store=[]
tag_strings = []
for tags in emotion_store:
    tag_string = tags['tags'][0]
    if 'movie' in tag_string:
        tag_string = 'movie, tv, reality'
    if 'surprise' in tag_string:
        tag_string = 'positive, joy, surprise, astonished'
    if 'grief-demo' in tag_string:
        tag_string = 'negative, sad, grief'
    if 'anger-demo' in tag_string:
        tag_string = 'negative, anger'
    if 'excitement-demo' in tag_string:
        tag_string = 'positive, joy, excitement'
    if 'humor' in tag_string:
        tag_string = 'positive, joy, comedy'

    tag_strings.append(tag_string)

    '''
    
    # this is where the error will be !!! 
    try:
        get_questions = paa.get_related_questions(tag_string, 5)
        more_questions=get_questions
        for question in get_questions:
            answer_tab = paa.get_answer(question)['related_questions']
            more_questions += answer_tab
        emotion_context_store.append({'tags': tag_string, 
                                    'questions':more_questions})
    except: # you can add the Google Error that is shown on your screen
        time.sleep(10)
        continue
    
    get_questions = paa.get_related_questions(tag_string, 5)
    more_questions=get_questions
    for question in get_questions:
        time.sleep(10)
        answer_tab = paa.get_answer(question)['related_questions']
        more_questions += answer_tab
    emotion_context_store.append({'tags': tag_string,
                                  'questions':more_questions})
    print (emotion_context_store)
    '''
print (tag_strings)
