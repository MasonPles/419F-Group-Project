#In this module we will build methods with which to
'''
Im going to assign 6 new columns ( Neu, Neg, Pos, bigramMatch%, unigramMatch%, perc_HashT_Match) Neu, neg and pos will give us the sentiment in the description, while bigramMatch% is the percentage of bigrams that match those produced by the fake news articles,  divided by the total number of bi-grams possible in the description. Unigram match will do the same thing but with all unigrams or single words that are found in the description.  Hashtag matches will follow similar logic, thus all three of these ranges [0,1]. We can use these attributes in our decision tree to split, hopefully, we see some strong matches, and then can confirm upon viewing the post that they are miss-info


'''
import nltk

import os
import string
import pandas as pd

 
import re 
import time 
import nltk.corpus  
import unidecode 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer 
from autocorrect import Speller 
from bs4 import BeautifulSoup 
from nltk.corpus import stopwords 
from nltk import word_tokenize 
import string 

def genNgr(text, Ngram=1):
    '''
    This function will take in the cleaned text from each post and then produce the number of n-grams
    specified in the second argument 
    
    Arguments: Text file (which you would like to find n-grams from)
               Ngram: is the size of N 
    
    
    '''
    word = [word for word in text.split(" ") if word not in set(stopwords.words('english'))]
    temp = zip(*[words[i:] for i in range(0,Ngram)])
    ans =[' '.join(Ngram) for Ngram in temp]
    return ans

def gen_Sentiment( listt, dataframe ):
    '''
    This function takes in list of clean text and generates a score of neg, neu, or pos in the form of a dictionary.
    We then generate a new dataframe that will be return which include the description of the post cleaned, and the sentiment scores given by the nltk.sentiment SentimentIntensityAnalyzer
    
    '''
    
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer() 
    
    df2 = pd.DataFrame(columns = ['Desc','neg' , 'neu', 'pos'])
    
    for e in range(len(listt)):
        ee = str(dataframe['desc'][e])
        e1 = str(listt[e])
        score = sia.polarity_scores(ee)
        df2 = df2.append({'Desc': e1, 'neg':score['neg'], 'neu':score['neu'], 'pos':score['pos']}, ignore_index=True)
        
    df3 = pd.merge(dataframe, df2, left_index=True, right_index=True)
    return df3 

# def comp_bigram(text, listBigr):
    