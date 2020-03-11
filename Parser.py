#!/usr/bin/env python
# coding: utf-8

# In[19]:
 
import csv
from bs4 import BeautifulSoup
import os
import re
for file in os.listdir('C:\\Users\\nimis\\Desktop\\applied topics in ai\\corpus\\fulltext')[1711:]:

    with open('C:\\Users\\nimis\\Desktop\\applied topics in ai\\corpus\\fulltext\\'+file, "r") as f:
        
        contents = f.read()

        soup = BeautifulSoup(contents, 'lxml')


    # In[20]:


    title = soup.find_all('name')[0].text


    # In[21]:


    title


    # In[22]:


    sentences = [] 
    for sentence in soup.find_all('sentence'):
        sentences.append(sentence.text)


    # In[23]:


    len(sentences)


    # ### Removing the empty sentences

    # In[24]:


    import re


    # In[25]:


    sentences=  [s.replace('\n', '').replace('\t','').replace('\r','').strip() for s in sentences]


    # In[26]:

    idx_to_remove = []
    for idx, sentence in enumerate(sentences):
        if len(sentence)< 5 or sentence.split('.')[0].isdigit() or re.match(r'^[0-9\.]*$',sentence):
            idx_to_remove.append(idx)
    #         print("###########")
    # print(idx_to_remove)
    tmp_sentences = []
    for i in range(len(sentences)):
        if i not in idx_to_remove:
            tmp_sentences.append(sentences[i])
    sentences = tmp_sentences
    # In[27]:


    catchphrases = [] 
    for catchphrase in soup.find_all('catchphrase'):
        catchphrases.append(catchphrase.text)


    # In[28]:


    len(catchphrases)


    # ### Importing the GeneticOptimizer

    # In[54]:


    # from multiobjectivetextsummarization import Text
    from GeneticOptimizer import *


    # In[55]:


    from Text import JS_DIVERGENCE, compute_tf, find_avg_freq, KL_DIVERGENCE


    # In[56]:


    doc_freq = compute_tf(sentences)


    # In[57]:


    doc_freq


    # In[69]:


    catch_phrase_frequency = compute_tf(catchphrase)

    def my_js_divergence(sys_summary, catch_phrase_frequency):
        summary_freq = compute_tf(sys_summary)
        average_freq = find_avg_freq(summary_freq, doc_freq)

        jsd = KL_DIVERGENCE(summary_freq, average_freq) + KL_DIVERGENCE(doc_freq, average_freq)
        return jsd / 2.


    # In[70]:


    from GeneticOptimizer import GeneticOptimizer


    # In[114]:


    obj = GeneticOptimizer(my_js_divergence, [(title,sentences)],doc_freq,100,30,0.4,0.3,0.4,False)


    # In[115]:


    final = obj.implementation(100)


    # In[116]:




    # In[117]:




    # In[118]:


    final_str = ' '.join([str(sentence) for sentence in final[0]])
    # print(final_str)


    # In[119]:


    


    # In[120]:


    catchphrases_str = '. '.join([str(catchphrase) for catchphrase in catchphrases])
    # print(catchphrases_str)


    # In[121]:





    # USING JS DIVERGENCE
    # 

    # In[125]:


    catchphrases_freq = compute_tf(catchphrases)


    # In[126]:


    js_similarity = JS_DIVERGENCE(final[0],catchphrases_freq)


    # # ROGUE METHOD

    # In[127]:


    from rouge import Rouge


    # In[128]:


    rouge = Rouge()


    # In[129]:


    match_sentences = set()
    match_catchphrases = set()


    # In[ ]:





    # In[130]:


    for sentence in final[0]:
        for catchphrase in catchphrases:
            print('CatchPhrase : {0}, Sentence: {1}'.format(catchphrase.encode('utf-8'),sentence.encode('utf-8')))
            result=rouge.get_scores(sentence,catchphrase)
            if(result[0]["rouge-1"]["r"]>=0.3):
                match_sentences.add(final[0].index(sentence))
                match_catchphrases.add(catchphrases.index(catchphrase))
                

    recall = len(match_catchphrases)/len(catchphrases)
    precision = len(match_sentences)/len(final[0])
    with open('results'+".csv", 'a',newline='\n') as out:
        writer = csv.writer(out,delimiter=',')
        writer.writerow([file,recall,precision,js_similarity])
        out.close()

    # print(final_str)
    with open('summaries\\'+file+'_summary.txt','wb') as f:
        f.write(final_str.encode('utf-8'))
        f.close()
    





    # In[89]:


    # In[95]:




    # In[ ]:





    # In[ ]:




