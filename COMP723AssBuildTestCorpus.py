import re
from sklearn.datasets import load_files
import pickle
from nltk.corpus import stopwords

def BuildTestCorpus():
    #Load messages
    email_test_data = load_files(r".\test")
    X_test, y_test = email_test_data.data, email_test_data.target

    documents = []

    for sen in range(0, len(X_test)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X_test[sen]))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        document = document.split()

        from nltk.stem import WordNetLemmatizer

        stemmer = WordNetLemmatizer()
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)

        documents.append(document)

    #Lemmatise files
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(max_features=1000, min_df=10, max_df=0.6, stop_words=stopwords.words('english'))
    X_test = vectorizer.fit_transform(documents).toarray()

    #Save files to PC
    with open('x_testfile', 'wb') as picklefile:
        pickle.dump(X_test,picklefile)
    
    with open('y_testfile', 'wb') as picklefile:
        pickle.dump(y_test,picklefile)

    print('finished')

BuildTestCorpus()