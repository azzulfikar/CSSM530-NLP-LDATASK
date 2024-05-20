#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pandas as pd
import re

entries = []
entry_keys = ["_id", "text", "date", "topic", "author", "egitim", "egitim_sistemi", "egitim_channel"]

with open("/Users/zeynepzulfikar/Desktop/eksi_egitim_240418/egitim_entries.jsonl", "r") as f:
    for line in f:
        entry = json.loads(line)
        if entry["egitim_sistemi"] != True:
            continue
    
        entries.append({key: value for key, value in entry.items() if key in entry_keys})

entries_df = pd.DataFrame(entries)
entries_df


# In[2]:


import re

def remove_newline_from_text(entry):
    """Removes newline characters from the 'text' field of a dictionary."""
    entry['text'] = entry['text'].replace('\n', ' ')
    return entry

def clean_text_for_nlp(text):
    """Cleans text data for NLP tasks by removing punctuation, newlines, and extra whitespace."""
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)  
    # Replace newlines with spaces
    text = text.replace('\n', ' ')
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# Apply the function to clean text for NLP
entries_df['text'] = entries_df['text'].apply(clean_text_for_nlp)
entries_df


# In[3]:


import pandas as pd

topic_counts = entries_df['topic'].value_counts()
print("Number of Texts per Topic:")
topic_counts


# In[4]:


import pandas as pd

topic_counts = entries_df['topic'].value_counts()
print("Number of Texts per Topic:")

pd.set_option('display.max_rows', None)  
print(topic_counts)

with open("topic_counts.txt", "w") as f:
    print(topic_counts, file=f)


# In[5]:


topic_counts = entries_df['topic'].value_counts()

less_than_10_topics = topic_counts[topic_counts < 10].index

entries_df_deleted = entries_df[entries_df['topic'].isin(less_than_10_topics)]

entries_df_deleted

print("Number of rows in the new dataset: ", len(entries_df_deleted))


# In[6]:


topic_counts = entries_df['topic'].value_counts()

valid_topics = topic_counts[topic_counts >= 10].index

filtered_entries_df = entries_df[entries_df['topic'].isin(valid_topics)]

filtered_entries_df

print("Number of rows in the new dataset: ", len(filtered_entries_df))


# In[7]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Get the list of Turkish stopwords from NLTK
stop_words = stopwords.words('turkish')

# Define your additional stopwords
additional_stopwords = """
a
acaba
altı
altmış
ama
ancak
arada
artık
asla
aslında
ayrıca
az
bana
bazen
bazı
bazıları
belki
ben
benden
beni
benim
beri
beş
bile
bilhassa
bin
bir
biraz
birçoğu
birçok
biri
birisi
birkaç
birşey
biz
bizden
bize
bizi
bizim
böyle
böylece
bu
bkz
buna
bunda
bundan
bunlar
bunları
bunların
bunu
bunun
burada
bütün
çoğu
çoğunu
çok
çünkü
da
daha
dahi
dan
de
defa
değil
diğer
diğeri
diğerleri
diye
doksan
dokuz
dolayı
dolayısıyla
dört
e
edecek
eden
ederek
edilecek
ediliyor
edilmesi
ediyor
eğer
elbette
elli
en
etmesi
etti
ettiği
ettiğini
fakat
falan
filan
gene
gereği
gerek
gibi
göre
hala
halde
halen
hangi
hangisi
hani
hatta
hem
henüz
hep
hepsi
her
herhangi
herkes
herkese
herkesi
herkesin
hiç
hiçbir
hiçbiri
i
ı
için
içinde
iki
ile
ilgili
ise
işte
itibaren
itibariyle
kaç
kadar
karşın
kendi
kendilerine
kendine
kendini
kendisi
kendisine
kendisini
kez
ki
kim
kime
kimi
kimin
kimisi
kimse
kırk
madem
mesela
mi
mı
milyar
milyon
mu
mü
nasıl
ne
neden
nedenle
nerde
nerede
nereye
neyse
niçin
nin
nın
niye
nun
nün
o
öbür
olan
olarak
oldu
olduğu
olduğunu
olduklarını
olmadı
olmadığı
olmak
olması
olmayan
olmaz
olsa
olsun
olup
olur
olur
olursa
oluyor
on
ön
ona
önce
ondan
onlar
onlara
onlardan
onları
onların
onu
onun
orada
öte
ötürü
otuz
öyle
oysa
pek
rağmen
sana
sanki
şayet
şekilde
sekiz
seksen
sen
senden
seni
senin
şey
şeyden
şeye
şeyi
şeyler
şimdi
siz
sizden
size
sizi
sizin
sonra
şöyle
şu
şuna
şunları
şunu
ta
tabii
tam
tamam
tamamen
tarafından
trilyon
tüm
tümü
u
ü
üç
un
ün
üzere
var
vardı
ve
veya
ya
yani
yapacak
yapılan
yapılması
yapıyor
yapmak
yaptı
yaptığı
yaptığını
yaptıkları
ye
yedi
yerine
yetmiş
yi
yı
yine
yirmi
yoksa
yu
yüz
zaten
zira
""".split()

# Extend the NLTK stopword list
stop_words.extend(additional_stopwords)

# Function to remove stopwords from a text
def remove_stopwords(text, stop_words):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Apply the function to the 'text' column of filtered_entries_df
new_entries_df = filtered_entries_df.copy()
new_entries_df['text'] = new_entries_df['text'].apply(lambda x: remove_stopwords(x, stop_words))


# In[8]:


from nltk.tokenize import word_tokenize
import string

def remove_stopwords_and_punctuation(text, stop_words):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords and punctuation
    filtered_tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    
    return ' '.join(filtered_tokens)

# Apply the function to the 'text' column
new_entries_df['text'] = new_entries_df['text'].apply(lambda x: remove_stopwords_and_punctuation(x, stop_words))


# In[9]:


import re

def clean_text(text):
    # Remove punctuations
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove single alphabetical letters
    text = re.sub(r'\b\w\b', '', text)
    
    return text

# Apply the function to the 'text' column
new_entries_df['text'] = new_entries_df['text'].apply(clean_text)


# In[10]:


# Convert all words in 'text' column to lowercase
new_entries_df['text'] = new_entries_df['text'].str.lower()


# In[11]:


from collections import Counter
import pandas as pd

# Assuming 'tokens' column contains tokenized text
all_words = [word for text in new_entries_df['text'] for word in word_tokenize(text)]

# Get the count of each word
word_counts = Counter(all_words)

# Convert to DataFrame for easier processing
word_counts_df = pd.DataFrame.from_dict(word_counts, orient='index').reset_index()
word_counts_df.columns = ['Word', 'Frequency']

# Sort by frequency
word_counts_df = word_counts_df.sort_values(by='Frequency', ascending=False)

# Print the most frequent words
print(word_counts_df.head(50))


# In[12]:


pip install zeyrek


# In[ ]:


import zeyrek
analyzer = zeyrek.MorphAnalyzer()

def lemmatize_text(text):
    lemmas = analyzer.lemmatize(text)
    lemmas = [' '.join(map(str, lemma)) for lemma in lemmas]
    return ">>> print('" + " ".join(lemmas) + "')"

new_entries_df['text'] = new_entries_df['text'].apply(lemmatize_text)


# In[20]:


from collections import Counter
import pandas as pd

# Assuming 'tokens' column contains tokenized text
all_words = [word for text in new_entries_df['text'] for word in word_tokenize(text)]

# Get the count of each word
word_counts = Counter(all_words)

# Convert to DataFrame for easier processing
word_counts_df = pd.DataFrame.from_dict(word_counts, orient='index').reset_index()
word_counts_df.columns = ['Word', 'Frequency']

# Sort by frequency
word_counts_df = word_counts_df.sort_values(by='Frequency', ascending=False)

# Print the most frequent words
print(word_counts_df.head(200))


# In[21]:


from collections import Counter
import pandas as pd
from nltk.tokenize import word_tokenize

# Assuming 'tokens' column contains tokenized text
all_words = [word for text in new_entries_df['text'] for word in word_tokenize(text)]

# Get the count of each word
word_counts = Counter(all_words)

# Convert to DataFrame for easier processing
word_counts_df = pd.DataFrame.from_dict(word_counts, orient='index').reset_index()
word_counts_df.columns = ['Word', 'Frequency']

# Sort by frequency
word_counts_df = word_counts_df.sort_values(by='Frequency', ascending=False)

# Exclude words with frequency more than 2000
word_counts_df = word_counts_df[word_counts_df['Frequency'] <= 2000]

# Print the most frequent words
print(word_counts_df.head(300))


# In[22]:


import pandas as pd

# Assuming excel file is 'labels.xlsx' and it has a sheet named 'Sheet1'
labels_df = pd.read_excel('/Users/zeynepzulfikar/Desktop/labeling for LDA.xlsx', sheet_name='Sheet1')

# Assume that labels_df has columns 'topic', 'system', 'problem', 'solution'
# Merge it with your filtered_entries_df
merged_df = pd.merge(new_entries_df, labels_df, on='topic', how='left')

# Now, for each of the categories, create a separate dataframe
system_df = merged_df[merged_df['system'] == 1]
problem_df = merged_df[merged_df['problems'] == 1]
solution_df = merged_df[merged_df['solutions'] == 1]

# Now, you have three dataframes system_df, problem_df, solution_df
# Each of these dataframes contains only the rows of the original dataframe where the respective label was 1
# You can proceed with your analysis on these dataframes


# In[23]:


print(new_entries_df.columns)
print(labels_df.columns)
print(merged_df.columns)


# In[24]:


from gensim.parsing.preprocessing import strip_punctuation, strip_multiple_whitespaces
from gensim import corpora
from gensim.models import LdaModel
from collections import defaultdict

# Tokenize the documents and remove punctuation
documents1 = system_df['text'].apply(lambda x: strip_multiple_whitespaces(strip_punctuation(x.lower())).split())

# Build a dictionary where for each document each word has its own id
dictionary1 = corpora.Dictionary(documents1)

# Get the words to be excluded
excluded_words = ["sistem", "print", "eğitim", "çocuk", "öğrenci", "üniversit", 
                "öğretmen","okul", "lise", "türkiy", "eğitim", "print", 
                "öğretmen", "sistem", "egitim", "çocuk", "turkiyenin", 
                "bilgi", "öğretmek", "türkiy", "türki", "yaaaa", "sınav", 
                "üniversite", "soru", "sadec", "ülke", "demek", "yıl", "sene", "sınıf", "kitap", "türkiye", "insan"]

# Get frequency of each word
frequency = defaultdict(int)
for text in documents1:
    for token in text:
        frequency[token] += 1

# Filter out words that are too frequent or in the excluded list
dictionary1.filter_tokens(bad_ids=[tokenid for tokenid, word in dictionary1.items() if word in excluded_words or frequency[word] > 1000])

# Transform the collection of texts to a numerical form
corpus1 = [dictionary1.doc2bow(text) for text in documents1]

# Build the LDA model
model1 = LdaModel(corpus=corpus1, num_topics=8, id2word=dictionary1) 

# Print the topics
topics = model1.print_topics(num_words=15)
for topic in topics:
    print(topic)


# In[28]:


from gensim.parsing.preprocessing import strip_punctuation, strip_multiple_whitespaces
from gensim.models import CoherenceModel, LdaModel
from gensim import corpora
from collections import defaultdict

# Tokenize the documents and remove punctuation
documents2 = system_df['text'].apply(lambda x: strip_multiple_whitespaces(strip_punctuation(x.lower())).split())

# Build a dictionary where for each document each word has its own id
dictionary = corpora.Dictionary(documents2)

# Transform the collection of texts to a numerical form
corpus = [dictionary.doc2bow(text) for text in documents2]

# Get the words to be excluded
excluded_words = ["sistem", "print", "eğitim", "çocuk", "öğrenci", "üniversit", 
                "öğretmen","okul", "lise", "türkiy", "eğitim", "print", 
                "öğretmen", "sistem", "egitim", "çocuk", "turkiyenin", 
                "bilgi", "öğretmek", "türkiy", "türki", "yaaaa", "sınav", 
                "üniversite", "soru", "sadec", "ülke", "demek", "yıl", "sene", "sınıf", "kitap", "türkiye", "insan"]

# Get frequency of each word
frequency = defaultdict(int)
for text in corpus:
    for token_id, freq in text:
        frequency[dictionary[token_id]] += freq

# Filter out words that are too frequent or in the excluded list
dictionary.filter_tokens(bad_ids=[tokenid for tokenid, word in dictionary.items() if word in excluded_words or frequency[word] > 1000])

# Update the corpus to reflect the new dictionary
corpus = [dictionary.doc2bow(text) for text in documents2]

# Define the list of number of topics to try
num_topics_list = [5, 10, 15, 20, 25]

# Define the list of passes
passes_list = [1, 5, 10, 15, 20]

# Initialize a variable to store the maximum coherence value
max_coherence = 0
model1 = None

# Perform grid search
for num_topics in num_topics_list:
    for passes in passes_list:
        temp_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=passes)
        coherence_model = CoherenceModel(model=temp_model, texts=documents2, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        perplexity_score = temp_model.log_perplexity(corpus)
        
        print(f'Num Topics: {num_topics}, Passes: {passes}, Coherence Score: {coherence_score}, Perplexity Score: {perplexity_score}')
        
        if coherence_score > max_coherence:
            max_coherence = coherence_score
            model1 = temp_model

# Print the topics of the best model
topics = model1.print_topics(num_words=10)
for topic in topics:
    print(topic)


# In[26]:


from gensim.parsing.preprocessing import strip_punctuation, strip_multiple_whitespaces
from gensim.models import CoherenceModel, LdaModel
from gensim import corpora
from collections import defaultdict

# Tokenize the documents and remove punctuation
documents2 = problem_df['text'].apply(lambda x: strip_multiple_whitespaces(strip_punctuation(x.lower())).split())

# Build a dictionary where for each document each word has its own id
dictionary = corpora.Dictionary(documents2)

# Transform the collection of texts to a numerical form
corpus = [dictionary.doc2bow(text) for text in documents2]

# Get the words to be excluded
excluded_words = ["sistem", "print", "eğitim", "çocuk", "öğrenci", "üniversit", 
                "öğretmen","okul", "lise", "türkiy", "eğitim", "print", 
                "öğretmen", "sistem", "egitim", "çocuk", "turkiyenin", 
                "bilgi", "öğretmek", "türkiy", "türki", "yaaaa", "sınav", 
                "üniversite", "soru", "sadec", "ülke", "demek", "yıl", "sene", "sınıf", "kitap", "türkiye", "insan"]

# Get frequency of each word
frequency = defaultdict(int)
for text in corpus:
    for token_id, freq in text:
        frequency[dictionary[token_id]] += freq

# Filter out words that are too frequent or in the excluded list
dictionary.filter_tokens(bad_ids=[tokenid for tokenid, word in dictionary.items() if word in excluded_words or frequency[word] > 1000])

# Update the corpus to reflect the new dictionary
corpus = [dictionary.doc2bow(text) for text in documents2]

# Define the list of number of topics to try
num_topics_list = [5, 10, 15, 20, 25]

# Define the list of passes
passes_list = [1, 5, 10, 15, 20]

# Initialize a variable to store the maximum coherence value
max_coherence = 0
best_model = None

# Perform grid search
for num_topics in num_topics_list:
    for passes in passes_list:
        temp_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=passes)
        coherence_model = CoherenceModel(model=temp_model, texts=documents2, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        perplexity_score = temp_model.log_perplexity(corpus)
        
        print(f'Num Topics: {num_topics}, Passes: {passes}, Coherence Score: {coherence_score}, Perplexity Score: {perplexity_score}')
        
        if coherence_score > max_coherence:
            max_coherence = coherence_score
            best_model = temp_model

# Print the topics of the best model
topics = best_model.print_topics(num_words=10)
for topic in topics:
    print(topic)


# In[29]:


from gensim.parsing.preprocessing import strip_punctuation, strip_multiple_whitespaces
from gensim.models import CoherenceModel, LdaModel
from gensim import corpora
from collections import defaultdict

# Tokenize the documents and remove punctuation
documents2 = solution_df['text'].apply(lambda x: strip_multiple_whitespaces(strip_punctuation(x.lower())).split())

# Build a dictionary where for each document each word has its own id
dictionary = corpora.Dictionary(documents2)

# Transform the collection of texts to a numerical form
corpus = [dictionary.doc2bow(text) for text in documents2]

# Get the words to be excluded
excluded_words = ["sistem", "print", "eğitim", "çocuk", "öğrenci", "üniversit", 
                "öğretmen","okul", "lise", "türkiy", "eğitim", "print", 
                "öğretmen", "sistem", "egitim", "çocuk", "turkiyenin", 
                "bilgi", "öğretmek", "türkiy", "türki", "yaaaa", "sınav", 
                "üniversite", "soru", "sadec", "ülke", "demek", "yıl", "sene", "sınıf", "kitap", "türkiye", "insan"]

# Get frequency of each word
frequency = defaultdict(int)
for text in corpus:
    for token_id, freq in text:
        frequency[dictionary[token_id]] += freq

# Filter out words that are too frequent or in the excluded list
dictionary.filter_tokens(bad_ids=[tokenid for tokenid, word in dictionary.items() if word in excluded_words or frequency[word] > 1000])

# Update the corpus to reflect the new dictionary
corpus = [dictionary.doc2bow(text) for text in documents2]

# Define the list of number of topics to try
num_topics_list = [5, 10, 15, 20, 25]

# Define the list of passes
passes_list = [1, 5, 10, 15, 20]

# Initialize a variable to store the maximum coherence value
max_coherence = 0
model3 = None

# Perform grid search
for num_topics in num_topics_list:
    for passes in passes_list:
        temp_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=passes)
        coherence_model = CoherenceModel(model=temp_model, texts=documents2, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        perplexity_score = temp_model.log_perplexity(corpus)
        
        print(f'Num Topics: {num_topics}, Passes: {passes}, Coherence Score: {coherence_score}, Perplexity Score: {perplexity_score}')
        
        if coherence_score > max_coherence:
            max_coherence = coherence_score
            model3 = temp_model

# Print the topics of the best model
topics = model3.print_topics(num_words=10)
for topic in topics:
    print(topic)


# In[30]:


from gensim.models import Phrases, TfidfModel
from gensim.parsing.preprocessing import STOPWORDS, preprocess_string
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from collections import Counter

# Add your initial list of stop words if any
stop_words = set(STOPWORDS)

# Filtered dataframe for 'problems' column
filtered_df_problems = merged_df[merged_df['problems'] == 1]

while True:
    # Preprocess texts, excluding stop words
    texts = filtered_df_problems['text'].apply(lambda x: [word for word in preprocess_string(x) if word not in stop_words]).tolist()

    # Create a dictionary representation of the documents, and filter out extremes
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=10, no_above=0.2)

    # Convert document into the bag-of-words (BoW) format
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Use TF-IDF
    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    # Train the separate model
    separate_model = LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=10, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)

    # Get the words for each topic
    topic_words = {i: [word for word, _ in separate_model.show_topic(i, topn=10)] for i in range(separate_model.num_topics)}

    # Flatten the list of words for all topics
    all_words = [word for words in topic_words.values() for word in words]

    # Find the most common words
    common_words = [word for word, count in Counter(all_words).items() if count > 3]

    if not common_words:
        # No word is repeated in more than 3 topics, condition is met
        break

    # Add the common words to the stop words list
    stop_words.update(common_words)

# Print the Keyword in the 10 topics
print(separate_model.print_topics())


# In[31]:


from gensim.models import Phrases, TfidfModel
from gensim.parsing.preprocessing import STOPWORDS, preprocess_string
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from collections import Counter

# Add your initial list of stop words if any
stop_words = set(STOPWORDS)

# Filtered dataframe for 'problems' column
filtered_df_problems = merged_df[merged_df['problems'] == 1]

while True:
    # Preprocess texts, excluding stop words
    texts = filtered_df_problems['text'].apply(lambda x: [word for word in preprocess_string(x) if word not in stop_words]).tolist()

    # Create a dictionary representation of the documents, and filter out extremes
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=10, no_above=0.2)

    # Convert document into the bag-of-words (BoW) format
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Use TF-IDF
    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    # Train the separate model
    separate_model = LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=10, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)

    # Get the words for each topic
    topic_words = {i: [word for word, _ in separate_model.show_topic(i, topn=10)] for i in range(separate_model.num_topics)}

    # Flatten the list of words for all topics
    all_words = [word for words in topic_words.values() for word in words]

    # Find the most common words
    common_words = [word for word, count in Counter(all_words).items() if count > 3]

    if not common_words:
        # No word is repeated in more than 3 topics, condition is met
        break

    # Add the common words to the stop words list
    stop_words.update(common_words)

# Print the Keyword in the 10 topics
print(separate_model.print_topics())


# In[32]:


from gensim.models import Phrases, TfidfModel
from gensim.parsing.preprocessing import STOPWORDS, preprocess_string
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from collections import Counter

# Add your initial list of stop words if any
stop_words = set(STOPWORDS)

# Filtered dataframe for 'solutions' column
filtered_df_solutions = merged_df[merged_df['solutions'] == 1]

while True:
    # Preprocess texts, excluding stop words
    texts = filtered_df_solutions['text'].apply(lambda x: [word for word in preprocess_string(x) if word not in stop_words]).tolist()

    # Create a dictionary representation of the documents, and filter out extremes
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=10, no_above=0.2)

    # Convert document into the bag-of-words (BoW) format
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Use TF-IDF
    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    # Train the model
    lda_model3 = LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=10, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)

    # Get the words for each topic
    topic_words = {i: [word for word, _ in lda_model3.show_topic(i, topn=10)] for i in range(lda_model3.num_topics)}

    # Flatten the list of words for all topics
    all_words = [word for words in topic_words.values() for word in words]

    # Find the most common words
    common_words = [word for word, count in Counter(all_words).items() if count > 3]

    if not common_words:
        # No word is repeated in more than 5 topics, condition is met
        break

    # Add the common words to the stop words list
    stop_words.update(common_words)

# Print the Keyword in the 10 topics
print(lda_model3.print_topics())


# In[33]:


# Filtered dataframe for 'system' column
filtered_df_system = merged_df[merged_df['system'] == 1]

# Unique stop words for 'system'
stop_words_system = set(STOPWORDS)

while True:
    # Preprocess texts, excluding stop words
    texts_system = filtered_df_system['text'].apply(lambda x: [word for word in preprocess_string(x) if word not in stop_words_system]).tolist()

    # Create a unique dictionary representation of the documents for 'system'
    dictionary_system = Dictionary(texts_system)
    dictionary_system.filter_extremes(no_below=10, no_above=0.2)

    # Convert document into the bag-of-words (BoW) format for 'system'
    corpus_system = [dictionary_system.doc2bow(text) for text in texts_system]

    # Use TF-IDF for 'system'
    tfidf_system = TfidfModel(corpus_system)
    corpus_tfidf_system = tfidf_system[corpus_system]

    # Train the unique model for 'system'
    lda_model_system = LdaModel(corpus=corpus_tfidf_system, id2word=dictionary_system, num_topics=10, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)

    # Get the words for each topic for 'system'
    topic_words_system = {i: [word for word, _ in lda_model_system.show_topic(i, topn=10)] for i in range(lda_model_system.num_topics)}

    # Flatten the list of words for all topics for 'system'
    all_words_system = [word for words in topic_words_system.values() for word in words]

    # Find the most common words for 'system'
    common_words_system = [word for word, count in Counter(all_words_system).items() if count > 3]

    if not common_words_system:
        # No word is repeated in more than 3 topics for 'system', condition is met
        break

    # Add the common words to the stop words list for 'system'
    stop_words_system.update(common_words_system)

# Print the Keyword in the 10 topics for 'system'
print(lda_model_system.print_topics())


# In[36]:


# Filtered dataframe for 'problems' column
filtered_df_problems = merged_df[merged_df['problems'] == 1]

# Unique stop words for 'problems'
stop_words_problems = set(STOPWORDS)

while True:
    # Preprocess texts, excluding stop words
    texts_problems = filtered_df_problems['text'].apply(lambda x: [word for word in preprocess_string(x) if word not in stop_words_problems]).tolist()

    # Create a unique dictionary representation of the documents for 'problems'
    dictionary_problems = Dictionary(texts_problems)
    dictionary_problems.filter_extremes(no_below=10, no_above=0.2)

    # Convert document into the bag-of-words (BoW) format for 'problems'
    corpus_problems = [dictionary_problems.doc2bow(text) for text in texts_problems]

    # Use TF-IDF for 'problems'
    tfidf_problems = TfidfModel(corpus_problems)
    corpus_tfidf_problems = tfidf_problems[corpus_problems]

    # Train the unique model for 'problems'
    lda_model_problems = LdaModel(corpus=corpus_tfidf_problems, id2word=dictionary_problems, num_topics=10, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)

    # Get the words for each topic for 'problems'
    topic_words_problems = {i: [word for word, _ in lda_model_problems.show_topic(i, topn=10)] for i in range(lda_model_problems.num_topics)}

    # Flatten the list of words for all topics for 'problems'
    all_words_problems = [word for words in topic_words_problems.values() for word in words]

    # Find the most common words for 'problems'
    common_words_problems = [word for word, count in Counter(all_words_problems).items() if count > 3]

    if not common_words_problems:
        # No word is repeated in more than 3 topics for 'problems', condition is met
        break

    # Add the common words to the stop words list for 'problems'
    stop_words_problems.update(common_words_problems)

# Print the Keyword in the 10 topics for 'problems'
print(lda_model_problems.print_topics())


# In[37]:


# Filtered dataframe for 'solutions' column
filtered_df_solutions = merged_df[merged_df['solutions'] == 1]

# Unique stop words for 'solutions'
stop_words_solutions = set(STOPWORDS)

while True:
    # Preprocess texts, excluding stop words
    texts_solutions = filtered_df_solutions['text'].apply(lambda x: [word for word in preprocess_string(x) if word not in stop_words_solutions]).tolist()

    # Create a unique dictionary representation of the documents for 'solutions'
    dictionary_solutions = Dictionary(texts_solutions)
    dictionary_solutions.filter_extremes(no_below=10, no_above=0.2)

    # Convert document into the bag-of-words (BoW) format for 'solutions'
    corpus_solutions = [dictionary_solutions.doc2bow(text) for text in texts_solutions]

    # Use TF-IDF for 'solutions'
    tfidf_solutions = TfidfModel(corpus_solutions)
    corpus_tfidf_solutions = tfidf_solutions[corpus_solutions]

    # Train the unique model for 'solutions'
    lda_model_solutions = LdaModel(corpus=corpus_tfidf_solutions, id2word=dictionary_solutions, num_topics=10, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)

    # Get the words for each topic for 'solutions'
    topic_words_solutions = {i: [word for word, _ in lda_model_solutions.show_topic(i, topn=10)] for i in range(lda_model_solutions.num_topics)}

    # Flatten the list of words for all topics for 'solutions'
    all_words_solutions = [word for words in topic_words_solutions.values() for word in words]

    # Find the most common words for 'solutions'
    common_words_solutions = [word for word, count in Counter(all_words_solutions).items() if count > 3]

    if not common_words_solutions:
        # No word is repeated in more than 3 topics for 'solutions', condition is met
        break

    # Add the common words to the stop words list for 'solutions'
    stop_words_solutions.update(common_words_solutions)

# Print the Keyword in the 10 topics for 'solutions'
print(lda_model_solutions.print_topics())

