import people_also_ask as paa
word = "place"
questions = paa.get_related_questions(word,5)
extra_questions = []
for q in questions:
    extra_questions += paa.get_answer(q)['related_questions']
questions = questions + extra_questions
f = open(word+'.txt','a')
for q in questions:
    f.write(q+'\n')
f.close()

