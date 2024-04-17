import numpy as np
import re
import nltk
import gensim.downloader as api
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import euclidean_distances

word_vectors = api.load("word2vec-google-news-300")
nltk.download('punkt')
nltk.download('stopwords')

max_features = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
maxlen = 100
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)
stop_words = set(stopwords.words('english'))

# Знайти речення в рецензії
def extract_sentences(review):
    review_text = b' '.join(review)
    sentences = sent_tokenize(review_text.decode('ISO-8859-1'))
    return sentences


# Обробити речення
def preprocess_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    stop_words = set(stopwords.words('english'))
    words = sentence.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)


# Знайти ембедінг речення як середнє ембедінг слів
def compute_sentence_embedding(sentence, word_vectors):
    embeddings = []
    for word in sentence.split():
        if word in word_vectors.key_to_index:
            embeddings.append(word_vectors.get_vector(word))
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(word_vectors.vector_size)

# Побудувати ембедінги "позитивності" P і "негативності" N
def compute_polarity_embeddings(word_vectors, positive_words, negative_words):
    positive_embeddings = [word_vectors.get_vector(word) for word in
                           positive_words if word in word_vectors.key_to_index]
    negative_embeddings = [word_vectors.get_vector(word) for word in
                           negative_words if word in word_vectors.key_to_index]
    if positive_embeddings:
        p_embedding = np.mean(positive_embeddings, axis=0)
    else:
        p_embedding = np.zeros(word_vectors.vector_size)
    if negative_embeddings:
        n_embedding = np.mean(negative_embeddings, axis=0)
    else:
        n_embedding = np.zeros(word_vectors.vector_size)
    return p_embedding, n_embedding

# Списки позитивних та негативних слів
positive_words = ["good", "nice", "great", "super", "cool"]
negative_words = ["bad", "awful", "terrible", "horrible", "not", "no"]

# Обробка рецензій
X_train_processed = []
for review in X_train:
    review_sentences = extract_sentences(review)
    processed_sentences = [preprocess_sentence(sentence) for sentence in review_sentences]
    X_train_processed.extend(processed_sentences)

X_test_processed = []
for review in X_test:
    review_sentences = extract_sentences(review)
    processed_sentences = [preprocess_sentence(sentence) for sentence in review_sentences]
    X_test_processed.extend(processed_sentences)

# Ембедінги позитивності та негативності
p_embedding, n_embedding = compute_polarity_embeddings(word_vectors, positive_words, negative_words)

# Обчислення відстаней до P та N
distances_to_p = euclidean_distances([compute_sentence_embedding(sentence, word_vectors)
                                      for sentence in X_test_processed], [p_embedding])
distances_to_n = euclidean_distances([compute_sentence_embedding(sentence, word_vectors)
                                      for sentence in X_test_processed], [n_embedding])

# Класифікація рецензій
y_pred = []
for dist_p, dist_n in zip(distances_to_p[0], distances_to_n[0]):
    if dist_p < dist_n:
        y_pred.append(1)
    else:
        y_pred.append(0)

accuracy = np.mean(np.array(y_pred) == y_test)
print("Accuracy:", accuracy)
