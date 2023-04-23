import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

# Edit Distance
def edit_distance(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1])
    return dp[m][n]

str1 = "dog"
str2 = "dof"
print(edit_distance(str1, str2))

# Tokenization
text = "Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome. The sky is pinkish-blue. You shouldn't eat cardboard"
tokenized_text = sent_tokenize(text)
print(tokenized_text)

tokenized_word = word_tokenize(text)
print(tokenized_word)

# Stemming
porter_stemmer = PorterStemmer()
stemmed_words = [porter_stemmer.stem(w) for w in tokenized_word]
print("Filtered Sentence:", tokenized_word)
print("Stemmed Sentence:", stemmed_words)

# Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()
lemmatized_words = [wordnet_lemmatizer.lemmatize(w) for w in tokenized_word]
print("Filtered Sentence:", tokenized_word)
print("Lemmatized Sentence:", lemmatized_words)

# Snowball Stemmer
spanish_stemmer = SnowballStemmer('spanish')
print(spanish_stemmer.stem('comiendo'))

french_stemmer = SnowballStemmer('french')
print(french_stemmer.stem('manger'))

# Part-of-speech tagging and lemmatization using SpaCy
import spacy
nlp = spacy.load('en_core_web_sm')

def show_lemmas(text):
    for token in text:
        print(f'{token.text:{12}} {token.pos_:{6}} {token.lemma:<{22}} {token.lemma_}')
        
doc = nlp("I saw eighteen mice today!")
show_lemmas(doc)
