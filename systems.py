from utils.dataset import Dataset
from utils.system import System
from utils.scorer import report_score
import numpy as np
import nltk

### instructions
# to repeat the results for the best improved system on the English test data
# set upper = True, all_chars = True, bigrams =True and set everything else to 
# false

# to repeat the results for the best improved system on the Spanish test data 
# set upper = True, suffix = True, pos = True and set everything else to False

# keep the file structure the same as what is downloaded 
# change testset to devset if wanting to repeat the results on the development
# data

def execute_sys(language):
    data = Dataset(language)
    
    print("{}: {} training - {} dev".format(language, len(data.trainset), len(data.testset)))
    
    ### feature selection
    training =[]
    suffix = {}
    vowels_combo={}
    pos_tags={}
    
    chars={}
    bigrams={}
    trigrams={}
    
    vowels =['a', 'e', 'i', 'o', 'u', 'á',  'é', 'í',  'ó'  ,'ú']
    for sent in data.trainset:
        training.append((sent['target_word'], sent['gold_label']))
        tokenised = sent['target_word'].split(' ')
        
        for wd in tokenised:
            
            ### vowels features
            target = wd.lower()
            temp_combo = ''
            for char in target:
                if char in vowels:
                    temp_combo += char
                elif len(temp_combo) > 0:
                    vowels_combo[temp_combo] = 0
                    temp_combo=''
             
            ### suffix features           
            suffix[target[-3:]] = 0
            
            try:
                tag = nltk.pos_tag(nltk.word_tokenize(wd))[0][1]
                pos_tags[tag]=0
            except IndexError:
                pass
            
            for i in range(len(target)):
                chars[target[i]] = 0
            ### char bigram
            for i in range(len(target)-1):
                bigrams[target[i]+target[i+1]]=0
            ### char trigram
            for i in range(len(target)-2):
                trigrams[target[i]+target[i+1]+target[i+2]]=0
            
        
    vowels_combo_list=list(vowels_combo.keys())
    suffix_to_list = list(suffix.keys())
    suffix_list_len3 = [s for s in suffix_to_list if len(s) == 3]
    
    sys_run = System(language, 
                        Baseline_run = False, 
                        vowels = False, v_list = vowels_combo_list,
                        syllables = False, 
                        upper = True, 
                        suffix = False, s_list = suffix_list_len3,
                        vc_ratio = False, 
                        pos = False, pos_dict = pos_tags, 
                        all_chars = True, all_chars_dict = chars,
                        bigrams = True, bigrams_dict = bigrams, 
                        trigrams = False, trigrams_dict = trigrams)


    gold_labels = [sent['gold_label'] for sent in data.testset]
    

    sys_run.train(data.trainset)

    predictions = sys_run.test(data.testset)
    
    score = report_score(gold_labels, predictions, detailed=False)
    
    print(score)
            
    
    
    
    # this output the (target word, gold label) and the predicted label
    words=[]
    for sent in data.testset:
        words.append((sent['target_word'], sent['gold_label']))
    predict=[]
    for x in np.nditer(predictions):
        predict.append(np.asscalar(x))
    word_pred=[]
    for i in range(len(predict)):
        word_pred.append((words[i], predict[i]))
    
    
    return word_pred 

if __name__ == '__main__':
    words_gold_predictions = execute_sys('english')
    words_gold_predictions = execute_sys('spanish')

