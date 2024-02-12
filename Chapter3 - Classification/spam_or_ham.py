#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 21:03:24 2024

@author: nitaishah
"""

import os

DATASETS_DIR = '/Users/nitaishah/Desktop/Hands-on-ML/Chapter3 - Classification/spam_or_ham_dataset'
TAR_DIR = os.path.join(DATASETS_DIR, 'tar')

SPAM_URL = 'https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2'
EASY_HAM_URL = 'https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2'
HARD_HAM_URL = 'https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2'

from urllib.request import urlretrieve
import tarfile
import shutil

def download_dataset(url):
    
    if not os.path.isdir(TAR_DIR):
        os.makedirs(TAR_DIR)
    
    filename = url.rsplit('/', 1)[-1]
    tarpath = os.path.join(TAR_DIR, filename)
    
    try:
        tarfile.open(tarpath)
    except:
        urlretrieve(url, tarpath)
        
    with tarfile.open(tarpath) as tar:
        dirname = os.path.join(DATASETS_DIR, tar.getnames()[0])
        if os.path.isdir(dirname):
            shutil.rmtree(dirname)
        tar.extractall(path=DATASETS_DIR)
        
        cmds_path = os.path.join(dirname, 'cmds')
        if os.path.isfile(cmds_path):
                os.remove(cmds_path)
                
    return dirname
    
#spam_dir = download_dataset(SPAM_URL)
#easy_ham_dir = download_dataset(EASY_HAM_URL)
#hard_ham_dir = download_dataset(HARD_HAM_URL)

spam_dir = '/Users/nitaishah/Desktop/Hands-on-ML/Chapter3 - Classification/spam_or_ham_dataset/spam_2'
easy_ham_dir = '/Users/nitaishah/Desktop/Hands-on-ML/Chapter3 - Classification/spam_or_ham_dataset/easy_ham_2'
hard_ham_dir = '/Users/nitaishah/Desktop/Hands-on-ML/Chapter3 - Classification/spam_or_ham_dataset/hard_ham'

import numpy as np
import glob

def load_dataset(dirpath):
    
    files = []
    filepaths = glob.glob(dirpath + '/*')
    for path in filepaths:
        with open(path, 'rb') as f:
            byte_content = f.read()
            str_content = byte_content.decode('utf-8', errors='ignore')
            files.append(str_content)
    return files


spam = load_dataset(spam_dir)
easy_ham = load_dataset(easy_ham_dir)
hard_ham = load_dataset(hard_ham_dir)
ham = easy_ham + hard_ham

import email
import email.policy
from email import policy


def load_emails(dirpath):
    email_list = []

    filepaths = glob.glob(dirpath + '/*')

    for path in filepaths:
        with open(path, 'rb') as f:
            email_message = email.parser.BytesParser(policy=email.policy.default).parse(f)
            email_list.append(email_message)

    return email_list

spam_new = load_emails("/Users/nitaishah/Desktop/Hands-on-ML/Chapter3 - Classification/spam_or_ham_dataset/spam_2")
spam_new[8].get_content().strip()

easy_ham_new = load_emails("/Users/nitaishah/Desktop/Hands-on-ML/Chapter3 - Classification/spam_or_ham_dataset/easy_ham_2")
easy_ham_new[8].get_content().strip()

hard_ham_new = load_emails("/Users/nitaishah/Desktop/Hands-on-ML/Chapter3 - Classification/spam_or_ham_dataset/hard_ham")
hard_ham_new[8].get_content().strip()


def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ]))
    else:
        return email.get_content_type()
    
    
from collections import Counter

def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures

structures_counter(easy_ham_new).most_common()
structures_counter(hard_ham_new).most_common()
structures_counter(spam_new).most_common()

for header, value in spam_new[0].items():
    print(header,":",value)
    
spam_new[0]['Subject']

spam_new[1].items()

import numpy as np
from sklearn.model_selection import train_test_split

X = np.array(spam_new + easy_ham_new + hard_ham_new, dtype=object)
y = np.array([0] * (len(easy_ham_new) + len(hard_ham_new)) + [1] * len(spam_new))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import re 
from html import unescape

def html_to_plain_text(html):
    text = re.sub('<head.*?.*?</head>','', html, flags=re.M | re.S | re.I)
    text = re.sub('<a\s.*?>', 'HYPERLINK', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)


html_spam_emails = [email for email in X_train[y_train==1]
                    if get_email_structure(email) == "text/html"]

sample_html_spam = html_spam_emails[7]
print(sample_html_spam.get_content().strip()[:1000], "...")

print(html_to_plain_text(sample_html_spam.get_content())[:1000], "...")


def email_to_text(email):
    html=None
    for part in email.walk():
        ctype=part.get_content_type()
        if not ctype in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content()
        except:
            content = str(part.get_payload())
        if ctype=="text/plain":
            return content
        else:
            html=content
    if html:
        return html_to_plain_text(html)
    
print(email_to_text(sample_html_spam)[:100], "...")

import nltk

try:
    import nltk

    stemmer = nltk.PorterStemmer()
    for word in ("Computations", "Computation", "Computing", "Computed", "Compute", "Compulsive"):
        print(word, "=>", stemmer.stem(word))
except ImportError:
    print("Error: stemming requires the NLTK module.")
    stemmer = None
    
try:
    import urlextract # may require an Internet connection to download root domain names
    
    url_extractor = urlextract.URLExtract()
    print(url_extractor.find_urls("Will it detect github.com and https://youtu.be/7Pq-S557XQU?t=3m32s"))
except ImportError:
    print("Error: replacing URLs requires the urlextract module.")
    url_extractor = None


from sklearn.base import BaseEstimator, TransformerMixin

class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strip_headers=True, lower_case=True, remove_punctuation=True,
                 replace_urls=True, replace_numbers=True, stemming=True):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_transformed = []
        for email in X:
            text = email_to_text(email) or ""
            if self.lower_case:
                text = text.lower()
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, " URL ")
            if self.replace_numbers:
                text = re.sub(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', 'NUMBER', text)
            if self.remove_punctuation:
                text = re.sub(r'\W+', ' ', text, flags=re.M)
            word_counts = Counter(text.split())
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
            X_transformed.append(word_counts)
        return np.array(X_transformed)

                    
            
   
            
X_few = X_train[:3]
X_few_wordcounts = EmailToWordCounterTransformer().fit_transform(X_few)
X_few_wordcounts            
   
from scipy.sparse import csr_matrix

class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size
    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[:self.vocabulary_size]
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
        return self
    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))
            
   
vocab_transformer = WordCounterToVectorTransformer(vocabulary_size=10)
X_few_vectors = vocab_transformer.fit_transform(X_few_wordcounts)
X_few_vectors            
   
X_few_vectors.toarray()
   
vocab_transformer.vocabulary_            
   

from sklearn.pipeline import Pipeline

preprocess_pipeline = Pipeline([
    ("email_to_wordcount", EmailToWordCounterTransformer()),
    ("wordcount_to_vector", WordCounterToVectorTransformer()),
])

X_train_transformed = preprocess_pipeline.fit_transform(X_train)
            
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

log_clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
score = cross_val_score(log_clf, X_train_transformed, y_train, cv=3, verbose=3)
score.mean()
   
from sklearn.metrics import precision_score, recall_score

X_test_transformed = preprocess_pipeline.transform(X_test)

log_clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
log_clf.fit(X_train_transformed, y_train)

y_pred = log_clf.predict(X_test_transformed)

print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_pred)))
print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_pred)))
            

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
            
tree_clf = DecisionTreeClassifier()
forest_clf = RandomForestClassifier()
svm_clf = SVC()
            
tree_clf.fit(X_train_transformed, y_train)
y_pred_tree = tree_clf.predict(X_test_transformed)

print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_pred_tree)))
print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_pred_tree)))

forest_clf.fit(X_train_transformed, y_train)
y_pred_forest = forest_clf.predict(X_test_transformed)

print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_pred_forest)))
print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_pred_forest)))

svm_clf.fit(X_train_transformed, y_train)
y_pred_svm = svm_clf.predict(X_test_transformed)

print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_pred_svm)))
print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_pred_svm)))

