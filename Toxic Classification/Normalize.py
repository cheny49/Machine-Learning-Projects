import pandas as pd
import string
import re

def remove_non_ascii(text):
    return "".join(i for i in text if ord(i) < 128)

def normalize(df, comment_text):
    contractions = {"y" : "why",
    "r": "are",
    "u": "you",
    "haha": "ha",
    "hahaha": "ha",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "can't": "can not",
    "cannot": "can not",
    "i'm": "i am",
    "m": "am",
    "i'll" : "i will",
    "its" : "it is",
    "it's" : "it is",
    "'s" : " is",
    "that's" : "that is",
    "weren't" : "were not",
    "they'll" : "they will",
    "you'll"  : "you will",
    "'ll"  :  " will"
    }

    keys = [i for i in contractions.keys()]
    new_train_data = []
    ltr = df[comment_text].values
    for i in ltr:
        array = str(i).split()
        sentence = ""
        for word in array:
            word = str(word).lower()
            if word in keys:
                word = contractions[word]
            sentence += word + " "
        new_train_data.append(sentence)
    df[comment_text] = new_train_data

    comment = comment_text
    df[comment] = df[comment].apply(lambda x: x.lower())
    df[comment] = df[comment].replace(["&", "@"], ["and", "at"], regex=True)
    df[comment] = df[comment].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    df[comment] = df[comment].apply(lambda x: x.translate(str.maketrans('', '', string.digits)))
    df[comment] = df[comment].apply(lambda x: " ".join(re.findall('[\w]+', x)))
    df[comment] = df[comment].replace("\n", " ", regex=True)
    df[comment] = df[comment].str.replace('http\S+|www.\S+', "", case=False)
    df[comment] = df[comment].apply(remove_non_ascii)
    df[comment] = df[comment].replace('\s+', ' ', regex=True)

    return df


train_file = r'C:\Users\cheny\Downloads\data\train.csv'
test_file = r'C:\Users\cheny\Downloads\data\test.csv'

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

df_train = train
df_test = test

df_train = normalize(train, 'comment_text')
df_test = normalize(test, 'comment_text')
df_train.to_csv('train_clean.csv')
df_test.to_csv('test_clean.csv')