import people_also_ask as paa
words = ['negative, sad, regret', 'movie, tv, reality', 'movie, tv, reality', 'positive, joy, excitement', 'negative, fear, nervousness', 'travel, place, nature, forest', 'movie, tv, reality', 'positive, love, romantic', 'positive, joy, excitement, , , sports, ball, ballonly, soccer', 'negative, jealousy', 'travel, place, nature, mountain', 'ambiguous, sarcasm', 'negative, anger, frustration', 'negative, fear, horror', 'movie, tv, reality', 'positive, joy, pride', 'movie, tv, reality', 'positive, joy, surprise, astonished', 'movie, tv, reality', 'ambiguous, confusion', 'positive, joy, curiosity', 'positive, love, platonic', 'negative, sad, grief', 'travel, place, nature, waterbody', 'positive, joy, excitement', 'movie, tv, reality', 'negative, sad, grief', 'negative, disgust']
for word in words:
    questions = paa.get_related_questions(word,5)
    extra_questions = []
    for q in questions:
        extra_questions += paa.get_answer(q)['related_questions']
    questions = questions + extra_questions
    f = open(word+'.txt','a')
    for q in questions:
        f.write(q+'\n')
    f.close()

