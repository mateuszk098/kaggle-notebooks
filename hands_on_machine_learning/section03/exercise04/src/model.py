"""
Exercise: Build a spam classifier (a more challenging exercise):

* Download examples of spam and ham from https://homl.info/spamassassin.
* Unzip the datasets and familiarize yourself with the data format.
* Split the datasets into a training set and a test set.
* Write a data preparation pipeline to convert each email into a feature vector. 

Your preparation pipeline should transform an email into a (sparse) vector that
indicates the presence or absence of each possible word. For example, if all emails
only ever contain four words, "Hello," "how," "are," "you," then the email "Hello you 
Hello Hello you" would be converted into a vector [1, 0, 0, 1] (meaning [“Hello" is present,
"how" is absent, "are" is absent, "you" is present]), or [3, 0, 0, 2] if you prefer to
count the number of occurrences of each word.

You may want to add hyperparameters to your preparation pipeline to control whether
or not to strip off email headers, convert each email to lowercase, remove punctuation,
replace all URLs with "URL," replace all numbers with "NUMBER," or even perform stemming
(i.e., trim off word endings; there are Python libraries available to do this).

Finally, try out several classifiers and see if you can build a great spam classifier,
with both high recall and high precision.
"""

import email
import email.policy
import re
import tarfile
import urllib.request
from collections import Counter
from html import unescape
from pathlib import Path

import nltk
import numpy as np
import urlextract
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def fetch_spam_data():
    """Download spam and ham data from https://spamassassin.apache.org/old/publiccorpus/
    Returns paths to ham and spam directories."""

    url_root = "https://spamassassin.apache.org/old/publiccorpus/"
    ham_url = url_root + "20030228_hard_ham.tar.bz2"
    spam_url = url_root + "20050311_spam_2.tar.bz2"

    data_path = Path("../data")
    data_path.mkdir(exist_ok=True)

    for dir_name, tar_name, url in (("hard_ham", "hard_ham", ham_url), ("spam_2", "spam_2", spam_url)):
        if not (data_path / dir_name).is_dir():
            path = (data_path / tar_name).with_suffix(".tar.bz2")
            print("Downloading", path)
            urllib.request.urlretrieve(url, path)
            with tarfile.open(path) as tf:
                tf.extractall(path=data_path)

    return [data_path / dir_name for dir_name in ("hard_ham", "spam_2")]


def load_email(filepath):
    """Returns a parsed `EmailMessage` object."""
    with open(filepath, "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)  # type: ignore


def get_email_structure(email):
    """Returns the email structure type or types if there is a multipart email."""
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):  # Multipart email (different content types).
        multipart = ", ".join([get_email_structure(sub_email) for sub_email in payload])
        return f"multipart({multipart})"
    else:
        return email.get_content_type()


def structures_couter(emails):
    """Returns a counter of the number of emails structure."""
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures


def html_to_plain_text(html):
    """Some emails have html structure. This function first drops the `<head>`
    section, then converts all `<a>` tags to the word HYPERLINK, then it gets
    rid of all HTML tags, leaving only the plain text. For readability, it also
    replaces multiple newlines with single newlines, and finally it unescapes
    html entities (such as `&gt;` or `&nbsp;`)."""

    """
    re.M (multi-line) is a flag that tells the regular expression engine to treat the input
    string as multiple lines, with the ^ and $ characters matching the beginning and end
    of each line, rather than the beginning and end of the entire string.

    re.S (dotall) is a flag that tells the regular expression engine to treat the . character
    as matching any character, including a newline character. Without this flag, the . character
    will only match any character except a newline.

    re.I (ignorecase) is a flag that tells the regular expression engine to perform
    case-insensitive matching. Without this flag, the regular expression engine will
    perform case-sensitive matching.
    """
    text = re.sub("<head.*?>.*?</head>", "", html, flags=re.M | re.S | re.I)
    text = re.sub("<a\s.*?>", " HYPERLINK ", text, flags=re.M | re.S | re.I)
    text = re.sub("<.*?>", "", text, flags=re.M | re.S)
    text = re.sub(r"(\s*\n)+", "\n", text, flags=re.M | re.S)
    return unescape(text)


def email_to_text(email):
    "Returns email as plain string."
    html = None
    for part in email.walk():  # Iterate over all parts of the email.
        ctype = part.get_content_type()
        if ctype not in ("text/plain", "text/html"):
            continue  # Only plain and html are interesting.
        try:
            content = part.get_content()
        except:  # Encoding issues.
            content = str(part.get_payload())
        if ctype == "text/plain":
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(html)


class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    """This class provides a transformer, which transforms an email to a word counter."""

    def __init__(self, strip_headers=True, lower_case=True, remove_punctuation=True,
                 replace_urls=True, replace_numbers=True, stemming=True):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming
        self.stemmer = nltk.PorterStemmer()
        self.url_extractor = urlextract.URLExtract()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = []

        for email in X:
            text = email_to_text(email) or ""  # email_to_text() can return None.
            if self.lower_case:
                text = text.lower()
            if self.remove_punctuation:
                text = re.sub(r"\W+", " ", text, flags=re.M)
            if self.replace_urls:
                urls = list(set(self.url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, " URL ")
            if self.replace_numbers:
                text = re.sub(r"\d+(?:\.\d*)?(?:[eE][+-]?\d+)?", " NUMBER ", text)

            word_counts = Counter(text.split())
            if self.stemming:
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = self.stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts

            X_transformed.append(word_counts)

        return np.array(X_transformed)


class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    """This class provides transformer, which convert counter to a vocabulary vector."""

    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):
        total_count = Counter()
        for word_counter in X:
            for word, count in word_counter.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[:self.vocabulary_size]
        self.vocabulary_ = {word: index+1 for index, (word, count) in enumerate(most_common)}
        return self

    def transform(self, X, y=None):
        rows, cols, data = [], [], []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size+1))


def main():

    ham_dir, spam_dir = fetch_spam_data()
    ham_filenames = [f for f in sorted(ham_dir.iterdir()) if len(f.name) > 20]
    spam_filenames = [f for f in sorted(spam_dir.iterdir()) if len(f.name) > 20]
    ham_emails = [load_email(filepath) for filepath in ham_filenames]
    spam_emails = [load_email(filepath) for filepath in spam_filenames]

    X = np.array(ham_emails + spam_emails, dtype=object)
    y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocess_pipeline = Pipeline([
        ("email_to_wordcount", EmailToWordCounterTransformer()),
        ("wordcount_to_vector", WordCounterToVectorTransformer()),
    ])

    X_train_transformed = preprocess_pipeline.fit_transform(X_train)
    X_test_transformed = preprocess_pipeline.transform(X_test)

    log_clf = LogisticRegression(max_iter=1000, random_state=42)
    log_clf.fit(X_train_transformed, y_train)
    y_pred = log_clf.predict(X_test_transformed)

    print(f"Precision: {precision_score(y_test, y_pred):.2%}")
    print(f"Recall: {recall_score(y_test, y_pred):.2%}")


if __name__ == "__main__":
    main()
