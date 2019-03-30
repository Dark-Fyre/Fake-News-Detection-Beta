import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn import metrics
from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.datasets import make_classification
from sklearn.naive_bayes import MultinomialNB
from joblib import dump, load


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def most_informative_feature(vectorizer, classifier, n=30):
    """
    most in formative feature for binary classification:

    Identify most important features if given a vectorizer and binary classifier. Set n to the number
    of weighted features you would like to show.
    """

    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names()
    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
    topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]

    for coef, feat in topn_class1:
        print(class_labels[0], coef, feat)

    print()

    for coef, feat in reversed(topn_class2):
        print(class_labels[1], coef, feat)



if __name__ == '__main__':
    data=pd.read_csv("fake_or_real_news.csv")
    data=data.set_index("Unnamed: 0")
    df=data.label
    data.drop("label",axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data['text'], df, test_size=0.33, random_state=53)
    # count_vector=CountVectorizer(stop_words="english")
    # count_train=count_vector.fit_transform(X_train)
    # count_test=count_vector.transform(X_test)


    tfidf_vector=TfidfVectorizer(stop_words="english",max_df=0.7)
    tfidf_train=tfidf_vector.fit_transform(X_train)
    tfidf_test=tfidf_vector.transform(X_test)


    count_df = pd.DataFrame(count_train.A, columns=count_vector.get_feature_names())
    tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vector.get_feature_names())


    #Confusion Matrix using Passive Aggressive Classifier


    linear_clf = PassiveAggressiveClassifier(max_iter=50, tol=None)
    linear_clf.fit(tfidf_train, y_train)
    pred = linear_clf.predict(tfidf_test)
    score = metrics.accuracy_score(y_test, pred)
    scr=score*100
    print("accuracy:   %0.3f" % scr," %")
    cm = metrics.confusion_matrix(y_test, pred, labels=	['FAKE', 'REAL'])
    plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
    #The below function shows the most important features of the trained model
    #most_informative_feature(tfidf_vector,linear_clf)
    dump(linear_clf, 'PassiveAggressive_model.joblib')


    file=open("test1.txt")
    text=file.read()
    file.close()
    dict_1={120092:text}
    df=pd.DataFrame(dict_1,index={"text"})
    df=df.T
    df=df['text']
    test=tfidf_vector.transform(df)
    pred = linear_clf.predict(test)
    pred



    #TFIDF vector model using Naive Bayes 
    #In case a different model needs to be showed accuracy: 85.7%

    #clf = MultinomialNB() 
    #clf.fit(tfidf_train, y_train)
    #pred = clf.predict(tfidf_test)
    # score = metrics.accuracy_score(y_test, pred)
    #scr=score*100
    #print("accuracy:   %0.3f" % scr," %")
    #cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
    #plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
    #dump(clf,'Linear_normalclf_tfidf.joblib')
