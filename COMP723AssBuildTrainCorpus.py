
import re
from sklearn.datasets import load_files
import pickle
from nltk.corpus import stopwords

def BuildTrainCorpus():
    #Load messages
    email_train_data = load_files(r".\train")
    X_train, y_train = email_train_data.data, email_train_data.target


    documents = []

    for sen in range(0, len(X_train)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X_train[sen]))

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
    X_train = vectorizer.fit_transform(documents).toarray()

    #Save files to PC
    with open('x_trainfile', 'wb') as picklefile:
        pickle.dump(X_train,picklefile)

    with open('y_trainfile', 'wb') as picklefile:
        pickle.dump(y_train,picklefile)

    print('finished')

BuildTrainCorpus()