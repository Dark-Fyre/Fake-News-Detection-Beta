from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

'''
    execute 
    >>>import nltk
    >>>nltk.download("punkt")
    >>>nltk.download("stopwords")
'''

import pandas as pd
import numpy as np 

import string 
import os

STOP_WORDS_SET = set(stopwords.words('english'))

FILE_NAME = "fake_or_real_news.csv"
FOLDER_NAME = "dataset"
FAKE_LABEL = "FAKE"
REAL_LABEL = "REAL"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_FILE = os.path.join(
    BASE_DIR,
    os.path.join(FOLDER_NAME,FILE_NAME)
)

SANITIZE_DATASET_FOLDER = os.path.join(BASE_DIR,'dataset')

save_to = lambda filename: os.path.join(SANITIZE_DATASET_FOLDER,filename)

'''
    TODO :
        1.Get everything to small letters
        2.Similar encoding
        3.Remove stop words
        4.Remove Punctuations
        5.Tokenize
        6.Change FAKE ->0 and REAL -> 1
'''

def getSantitizeData(filename,commit=True,tokenize=True):
    df = pd.read_csv(DATASET_FILE)

    print("[+]Loaded %s successfully" % (DATASET_FILE))

    df.title = df.title.apply(lambda x: x.lower())
    df.text = df.text.apply(lambda x: x.lower())

    df.text = df['text'].str.replace('[^\w\s]','')
    df.label = df.label.apply(lambda x: not x )

    if tokenize:
        df.text = df.text.apply(word_tokenize)
        print("[+] Corpus Tokenized")

    df.text= df.text.apply(lambda x: [item for item in x if \
        item not in STOP_WORDS_SET])

    print("[+] Stop Words Removed")

    if commit:
        df.to_csv(save_to(filename))
        print("[+] %s committed." %(filename))

    return df

def getTestTrainData(commit=True,**kwargs):
    try:
        df = kwargs.pop('df')
    except: 
        filename = kwargs.get('filename')
        df = pd.read_csv(filename)
    
    #Get 80% in training and 20% for testing

    x = df.loc[:,'text'].values
    y = df.loc[:,'label'].values

    train_size = int(0.8 * len(y))
    test_size = len(x) - train_size

    x_train,y_train = x[:train_size],y[:train_size]
    x_test,y_test = x[train_size:],y[train_size:]

    np.save(save_to("npy\\x_train.npy"),x_train)
    np.save(save_to("npy\\y_train.npy"),y_train)
    print("[+]Training data commited (npy)")

    np.save(save_to("npy\\x_test.npy"),x_test)
    np.save(save_to("npy\\y_test"),y_test)
    print("[+]Testing data commited (npy)")

    return df
    '''    
    msk = np.random.rand(len(df)) < 0.8

    df[msk].to_csv(save_to("train_data.csv"))
    print("[+] Training data saved !")

    df[~msk].to_csv(save_to("test_data.csv"))
    print("[+] Testing data saved !")
    '''


def combineDataFrame(initial_file,destination_file,filename,commit=True):
    df_initial = getSantitizeData(initial_file,tokenize=False,commit=False)
    df_destination = getSantitizeData(destination_file,tokenize=False,commit=False)
    
    df = df_initial.append(df_destination)

    print("[+] %s and %s appended succuessfully" %(
        initial_file,
        destination_file
    ))

    df.to_csv(filename)

    print("[+] %s committed" %(filename))
    return df

if __name__ == "__main__":
    pass
