#!/usr/bin/env python
# coding: utf-8

# In[1]:


#First, we open up to see how the data contains and how should start the preprocessing process.  

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


#As we saw newlines, punctuations, numbers and extra spaces, with using regex, we start the preprocessing 
#with removing those. 

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

#We apply it to our dataset and see if it worked.
entries_df['text'] = entries_df['text'].apply(clean_text_for_nlp)
entries_df


# In[3]:


#We continue with delving into the dataset; we have to see the scale of it to manage how to deal with it. 
#As we see the number of entries per topic (157 topics exist with maximum 2067 entries with minimum 1). To deal with
#biases, we have to get rid of the outliars; however, in our case, the topics with largest entries must not to be taken
#as the outliers but rather they contain most of our data for second model we will prep for this analysis, "problems in the system"
#So, we only get rid of topics where entries less than 10; when you look at these entry names, you will see that they are just to detailed 
#and sophisticed for the LDA model to capture and put into a topic that it will find for our dataset.

#So, in few lines, we will get rid of these topics. 

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

print("Number of rows deleted from dataset: ", len(entries_df_deleted))


# In[6]:


topic_counts = entries_df['topic'].value_counts()

valid_topics = topic_counts[topic_counts >= 10].index

filtered_entries_df = entries_df[entries_df['topic'].isin(valid_topics)]

filtered_entries_df

print("Number of rows in the new dataset: ", len(filtered_entries_df))


# In[7]:


#After getting rid of the unnecessary rows, we can continue with next step our preprocessing, removing the stop words. 
#For removing stopwords in Turkish, there are few packages we can use. I used NLTK and added additional words. 
#These additional words actually come from trstop package in github, however, since I was not able to deal with
#uploading it to Python, i added them as a list here.

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = stopwords.words('turkish')

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

stop_words.extend(additional_stopwords)

def remove_stopwords(text, stop_words):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# then we apply the function to the 'text' column of our dataset where our entries exist in
new_entries_df = filtered_entries_df.copy()
new_entries_df['text'] = new_entries_df['text'].apply(lambda x: remove_stopwords(x, stop_words))


# In[8]:


#However, I realized that NLTK did not work here, I realized that I had to start with the tokenization first to
#make these functions and packages to work, so to get rid of these stopwords and punctuations, I rewrote the code. 

from nltk.tokenize import word_tokenize
import string

def remove_stopwords_and_punctuation(text, stop_words):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    
    return ' '.join(filtered_tokens)

new_entries_df['text'] = new_entries_df['text'].apply(lambda x: remove_stopwords_and_punctuation(x, stop_words))


# In[9]:


#I rewrote the code here for to rework it after tokenization of the text. I realized that I have to get rid of the
#numbers and single alphabetical letters as well, I added them here.

import re 

def clean_text(text):
    #Remove punctuations
    text = re.sub(r'[^\w\s]', '', text)
    
    #Remove numbers
    text = re.sub(r'\d+', '', text)
    
    #Remove single alphabetical letters
    text = re.sub(r'\b\w\b', '', text)
    
    return text

new_entries_df['text'] = new_entries_df['text'].apply(clean_text)


# In[10]:


#Then I convert all words in 'text' column to lowercase
new_entries_df['text'] = new_entries_df['text'].str.lower()


# In[11]:


#Then it came to more detailed pre-processing steps; what high frequencied words we have at hand that mess up our analysis?
#But before that, I realized that we have to the last pre-processing step to actually see the data we will use
#in the analysis. Before doing lemmatization, me deleting any frequency would not be wise for to continue. So, after this, I
#do the lemmization process. 

from collections import Counter
import pandas as pd

all_words = [word for text in new_entries_df['text'] for word in word_tokenize(text)]

word_counts = Counter(all_words)

word_counts_df = pd.DataFrame.from_dict(word_counts, orient='index').reset_index()
word_counts_df.columns = ['Word', 'Frequency']

word_counts_df = word_counts_df.sort_values(by='Frequency', ascending=False)

print(word_counts_df.head(50))


# In[12]:


pip install zeyrek


# In[ ]:


#I use Zeyrek for lemmatization, a Turkish library for this 

>>> import zeyrek
>>> analyzer = zeyrek.MorphAnalyzer()

def lemmatize_text(text):
    lemmas = analyzer.lemmatize(text)
    lemmas = [' '.join(map(str, lemma)) for lemma in lemmas]
    return ">>> print('" + " ".join(lemmas) + "')"

new_entries_df['text'] = new_entries_df['text'].apply(lemmatize_text)


# In[20]:


#After lemmatization, I open up the current dataset in df to see easily what should my cutoff point for
#getting rid of the outliers. I look at the frequencies of every single word here, sort them by number and see the first 50. 

from collections import Counter
import pandas as pd

all_words = [word for text in new_entries_df['text'] for word in word_tokenize(text)]

word_counts = Counter(all_words)

word_counts_df = pd.DataFrame.from_dict(word_counts, orient='index').reset_index()
word_counts_df.columns = ['Word', 'Frequency']

word_counts_df = word_counts_df.sort_values(by='Frequency', ascending=False)

print(word_counts_df.head(200))


# In[21]:


#After deciding on excluding the words, that has more than 2000 frequencies in the dataset, I rewrite the code 
#with that line and I see the distribution of words if everything is valid and good to go. 

from collections import Counter
import pandas as pd
from nltk.tokenize import word_tokenize

all_words = [word for text in new_entries_df['text'] for word in word_tokenize(text)]

word_counts = Counter(all_words)

word_counts_df = pd.DataFrame.from_dict(word_counts, orient='index').reset_index()
word_counts_df.columns = ['Word', 'Frequency']

word_counts_df = word_counts_df.sort_values(by='Frequency', ascending=False)

word_counts_df = word_counts_df[word_counts_df['Frequency'] <= 2000]

print(word_counts_df.head(300))


# In[22]:


#After seeing that it is, now it comes to building the model. Since I am going to divide the topics to three, 
#I did a labeling in an Excel class and I upload it here with establishing three different sub-datasets after merging
#my existing dataset and my labelling excel dataset. Then divide them into three; 
#each of these dataframes now contains only the rows of the original dataframe where the respective label was 1. 

import pandas as pd

labels_df = pd.read_excel('/Users/zeynepzulfikar/Desktop/labeling for LDA.xlsx', sheet_name='Sheet1')

merged_df = pd.merge(new_entries_df, labels_df, on='topic', how='left')

system_df = merged_df[merged_df['system'] == 1]
problem_df = merged_df[merged_df['problems'] == 1]
solution_df = merged_df[merged_df['solutions'] == 1]


# In[23]:


print(new_entries_df.columns)
print(labels_df.columns)
print(merged_df.columns)


# In[24]:


#After these I repeatedly tried different models with different tunings; the best scores I had were two types of models that I was able to
#get the best scores. Even though the pipeline is more important, I wanted to rely on scores as well. 

#Interestingly, my code didn't start before I re-tokenize 

#After these 'more' unsuccessful models, I realized that spesific education related words were also needed to go.
#I excluded them from the dataset and I also added extra line if prior filters are not valid here. After I checked the data, I saw
#that lemmatization and other things were valid, so I did not extraly added them.
#Since I realized that I needed to tokenize everytime I wrote a new code because I didn't restart the coding process
#after the preprocessing ended with the new data file, I again preprocessed with tokenization and removing, and I continued with creation
#of the dictionary and formulation of the model itself. 

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

dictionary1.filter_tokens(bad_ids=[tokenid for tokenid, word in dictionary1.items() if word in excluded_words or frequency[word] > 1000])

corpus1 = [dictionary1.doc2bow(text) for text in documents1]

model1 = LdaModel(corpus=corpus1, num_topics=8, id2word=dictionary1) 

topics = model1.print_topics(num_words=15)
for topic in topics:
    print(topic)


# In[28]:


#However, the tries of my number of topics here did not produced a valid analysis - left a example for us of the code 
#above to see how I first formulated the model. However, I realized that one of the main issues is the decision
#of the number of topics for LDA and since it generally comes from domain expertism 
#and we instead decided to go an explorative route with this model, so the data itself should reveal the right topic count to us. 
#In the process of deciding on doing a grid search for this, 20 topics with 20 passes won. 


from gensim.parsing.preprocessing import strip_punctuation, strip_multiple_whitespaces
from gensim.models import CoherenceModel, LdaModel
from gensim import corpora
from collections import defaultdict

#Tokenize the documents and remove punctuation
documents2 = system_df['text'].apply(lambda x: strip_multiple_whitespaces(strip_punctuation(x.lower())).split())

#Build a dictionary where for each document each word has its own id
dictionary = corpora.Dictionary(documents2)

#Transform the collection of texts to a numerical form
corpus = [dictionary.doc2bow(text) for text in documents2]

#Get the words to be excluded
excluded_words = ["sistem", "print", "eğitim", "çocuk", "öğrenci", "üniversit", 
                "öğretmen","okul", "lise", "türkiy", "eğitim", "print", 
                "öğretmen", "sistem", "egitim", "çocuk", "turkiyenin", 
                "bilgi", "öğretmek", "türkiy", "türki", "yaaaa", "sınav", 
                "üniversite", "soru", "sadec", "ülke", "demek", "yıl", "sene", "sınıf", "kitap", "türkiye", "insan"]

#Get frequency of each word
frequency = defaultdict(int)
for text in corpus:
    for token_id, freq in text:
        frequency[dictionary[token_id]] += freq

#Filter out words that are too frequent or in the excluded list
dictionary.filter_tokens(bad_ids=[tokenid for tokenid, word in dictionary.items() if word in excluded_words or frequency[word] > 1000])

#Update the corpus to reflect the new dictionary
corpus = [dictionary.doc2bow(text) for text in documents2]

#Define the list of number of topics to try
num_topics_list = [5, 10, 15, 20, 25]

#Define the list of passes
passes_list = [1, 5, 10, 15, 20]

#Initialize a variable to store the maximum coherence value
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

documents2 = problem_df['text'].apply(lambda x: strip_multiple_whitespaces(strip_punctuation(x.lower())).split())

dictionary = corpora.Dictionary(documents2)

corpus = [dictionary.doc2bow(text) for text in documents2]

excluded_words = ["sistem", "print", "eğitim", "çocuk", "öğrenci", "üniversit", 
                "öğretmen","okul", "lise", "türkiy", "eğitim", "print", 
                "öğretmen", "sistem", "egitim", "çocuk", "turkiyenin", 
                "bilgi", "öğretmek", "türkiy", "türki", "yaaaa", "sınav", 
                "üniversite", "soru", "sadec", "ülke", "demek", "yıl", "sene", "sınıf", "kitap", "türkiye", "insan"]

frequency = defaultdict(int)
for text in corpus:
    for token_id, freq in text:
        frequency[dictionary[token_id]] += freq

dictionary.filter_tokens(bad_ids=[tokenid for tokenid, word in dictionary.items() if word in excluded_words or frequency[word] > 1000])

corpus = [dictionary.doc2bow(text) for text in documents2]

num_topics_list = [5, 10, 15, 20, 25]

passes_list = [1, 5, 10, 15, 20]

max_coherence = 0
best_model = None

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


# In[33]:


#Even though we do not care for scores here, pipeline is the important one, then an idea popped into my mind. 
#Since LDA is a probability based statistical model and you can only fine tune it rather than break up the model, 
#I decided to break up the principles of the model to try and learn if it could provide an interesting perspective. 

#Here, you cannot limit which words the model will take and will it repeat it or not, however, I did just that here if model provides something differen
#I used other hyper tuning agents such as TF-DIF or bag of words, again, the scores were low
#and these models only acted here as an experiment.

from gensim.models import Phrases, TfidfModel
from gensim.parsing.preprocessing import STOPWORDS, preprocess_string
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from collections import Counter

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


from gensim.models import Phrases, TfidfModel
from gensim.parsing.preprocessing import STOPWORDS, preprocess_string
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from collections import Counter

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


from gensim.models import Phrases, TfidfModel
from gensim.parsing.preprocessing import STOPWORDS, preprocess_string
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from collections import Counter

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

