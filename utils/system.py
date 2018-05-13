import pyphen
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.svm import SVC

import nltk



class System(object):

    def __init__(self, language, Baseline_run, vowels, v_list, 
                 syllables, upper, suffix, s_list, 
                 vc_ratio, pos, pos_dict, 
                 all_chars, all_chars_dict, 
                 bigrams, bigrams_dict, 
                 trigrams, trigrams_dict):
        self.language = language
        self.Baseline_run = Baseline_run
        self.vowels = vowels
        self.v_list = v_list
        self.syllables = syllables
        self.upper = upper
        self.suffix = suffix
        self.s_list = s_list
        self.vc_ratio = vc_ratio
        self.pos = pos
        self.pos_dict = pos_dict
        self.all_chars = all_chars
        self.all_chars_dict = all_chars_dict        
        self.bigrams = bigrams
        self.bigrams_dict = bigrams_dict
        self.trigrams = trigrams
        self.trigrams_dict = trigrams_dict
        

        if language == 'english':
            self.dic = pyphen.Pyphen(lang='en')
        else:  # spanish
            self.dic = pyphen.Pyphen(lang='es')
        
            
        self.vowel_list = ['a', 'e', 'i', 'o', 'u', 'á',  'é', 'í',  'ó'  ,'ú']
        
        
        if self.Baseline_run == True:
            self.model = DecisionTreeClassifier(random_state=0)
        else:
            self.model = RandomForestClassifier(random_state=0)

        
        
    def extract_features(self, word):
        features=[]
        len_chars = len(word)
        features.append(len_chars)
        target=word.lower()

        
        if self.vowels == True:
                   
            vowel_combo_counts = []
            for v in self.v_list:
                vowel_combo_counts.append(target.count(v))
            
            features.extend(vowel_combo_counts)
            
        
        if self.syllables == True:
            words = word.split(' ')
            syllable_count = 0
            for wd in words:
                hyphenated_word = self.dic.inserted(wd)
                syllable_count += len(hyphenated_word.split('-'))
            
            features.append(syllable_count)
        
        
        if self.upper == True:
            Upper_true = 0
            for char in word:
                if char.isupper() == True:
                    Upper_true = 1
                    break
                else:
                    Upper_true = 0
           
            features.append(Upper_true)
       
        
        if self.suffix == True:
            
            suffices = []
            target_split = target.split(' ')
            for wd in target_split:
                suffices.append(wd[-3:])
            suffices_joined = " ".join(suffices)
            suffix_count=[]
            for s in self.s_list:
                suffix_count.append(suffices_joined.count(s))
            features.extend(suffix_count)
       
        
        if self.vc_ratio == True:
     
            vowel_counts = []
            for char in self.vowel_list:
                vowel_counts.append(target.count(char))
            
            tot_vowels=sum(vowel_counts)
            if (len_chars-tot_vowels) == 0:
                VC_ratio=tot_vowels
            else:
                VC_ratio=tot_vowels/((len_chars-tot_vowels))
            features.append(VC_ratio)
        
        if self.pos == True:
            tag_dict = dict.fromkeys(self.pos_dict, 0)
            target_split = target.split(' ')
            for wd in target_split:
                try:
                    tag = nltk.pos_tag(nltk.word_tokenize(wd))[0][1]
                    if tag in tag_dict:
                        tag_dict[tag] += 1
                    else:
                        pass
                except IndexError:
                    pass
            features.extend(list(tag_dict.values()))
        
        if self.all_chars == True:
            char_dict = dict.fromkeys(self.all_chars_dict, 0)
            target_split = target.split(' ')
            for wd in target_split:
                for i in range(len(wd)):
                    if wd[i] in char_dict:
                        char_dict[wd[i]] += 1
            
            features.extend(list(char_dict.values()))
        
        if self.bigrams == True:
            bi_dict = dict.fromkeys(self.bigrams_dict, 0)
            target_split = target.split(' ')
            for wd in target_split:
                for i in range(len(wd)-1):
                    if wd[i]+wd[i+1] in bi_dict:
                        bi_dict[wd[i]+wd[i+1]] += 1
            
            features.extend(list(bi_dict.values()))
        
        if self.trigrams == True:
            tri_dict = dict.fromkeys(self.trigrams_dict, 0)
            target_split = target.split(' ')
            for wd in target_split:
                for i in range(len(wd)-2):
                    if wd[i]+wd[i+1]+wd[i+2] in bi_dict:
                        tri_dict[wd[i]+wd[i+1]+wd[i+2]] += 1
            
            features.extend(list(tri_dict.values()))

        return features 

    def train(self, trainset):
        X = []
        y = []
        
        for sent in trainset:
            X.append(self.extract_features(sent['target_word']))
            y.append(sent['gold_label'])

        self.model.fit(X, y)

    def test(self, testset):
        X = []
        for sent in testset:
            X.append(self.extract_features(sent['target_word']))
        
        return self.model.predict(X)
