
## Extractive Summarization based on sentence ranking using TextRank

This is a simple implementation of extractive summarization based on sentence ranking using [TextRank](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)

Here's a sample summarization executed by this implementation:

### ORIGINAL TEXT: 
[(Source)](https://www.reddit.com/r/autotldr/comments/31b9fm/faq_autotldr_bot/)

Autotldr is a bot that uses SMMRY to create a TL;DR/summary. I will put forth points that address the effects this bot has on the reddit community.

It doesn't create laziness, it only responds to it

For the users who click the article link first and then return back to the comments, they will have already given their best attempt of fully reading the article. If they read it fully, the tl;dr is unneeded and ignored. If they skimmed or skipped it, the bot will be useful to at least provide more context to the discussion, like an extension of the title. A large portion of users, especially in the defaulted mainstream subreddits like /r/politics, don't even go to the article and go straight to the comments section. Most of the time, if I skip to the comments, I'm able to illicit some sort of understanding of what the article was about from the title and discussion. However this bot is able to further improve my conjectured understanding. It did not make me skip it, it only helped me when I already decided to skip it. The scenario in which this bot would create a significantly lazy atmosphere is if the tl;dr were to be presented parallel to the main submission, in the same way the OP's tl;dr is presented right next to the long body of self post. Also, the tl;dr becomes more prevalent/hidden as it will get upvoted/downvoted depending on how much of a demand there was for a tl;dr in the first place. If it becomes the top voted comment than it has become more of a competitor to the original text for those who go to the comments first, but by then the thread has decided that a tl;dr was useful and the bot delivered.

It can make sophisticated topics more relevant to mainstream Reddit

Sophisticated and important topics are usually accompanied or presented by long detailed articles. By making these articles and topics relevant to a larger portion of the Reddit userbase (those who weren't willing to read the full article), it popularizes the topic and increases user participation. These posts will get more attention in the form of upvotes/downvotes, comments, and reposts. This will increase the prevalence of sophisticated topics in the mainstream subreddits and compete against cliched memes. This has the potential of re-sophisticating the topic discussion in the mainstream subreddits, as more hardcore redditors don't have to retreat to a safe haven like /r/TrueReddit. This is a loose approximation and the magnitude of this effect is questionable, but I'm not surprised if the general direction of the theory is correct. I'm not claiming this would improve reddit overnight, but instead very very gradually.

It decreases Reddit's dependency on external sites

The bot doubles as a context provider for when a submission link goes down, is removed, or inaccessible at work/school. The next time the article you clicked gives you a 404 error, you won't have to depend on the users to provide context as the bot will have been able to provide that service at a much faster and consistent rate than a person. Additionally, an extended summary is posted in /r/autotldr, which acts as a perpetual archive and decreases how much reddit gets broken by external sites.

Only useful tl;dr's are posted

There are several criteria for a bot to post a tl;dr. It posts the three most important sentences as decided by the core algorithm, and they must be within 450-700 characters total. The final tl;dr must also be 70% smaller than the original, that way there is a big gap between the original and the tl;dr, hence only very long articles get posted on. This way the likelihood of someone nonchalantly declaring "TL;DR" in a thread and the bot posting in the same one is high. Also my strategy is to tell the bot to post in default, mainstream subreddits were the demand for a TL;DR is much higher than /r/TrueReddit and /r/worldevents.

Feel free to respond to these concepts and to raise your own. Be polite, respectful, and clarify what you say. Any offending posts to this rule will be removed.


### GENERATED SUMMARY:

Autotldr is a bot that uses SMMRY to create a TL ; DR/summary.

The scenario in which this bot would create a significantly lazy atmosphere is if the tl ; dr were to be presented parallel to the main submission , in the same way the OP 's tl ; dr is presented right next to the long body of self post.

If it becomes the top voted comment than it has become more of a competitor to the original text for those who go to the comments first , but by then the thread has decided that a tl ; dr was useful and the bot delivered.

By making these articles and topics relevant to a larger portion of the Reddit userbase ( those who were n't willing to read the full article ) , it popularizes the topic and increases user participation.

Only useful tl ; dr 's are posted There are several criteria for a bot to post a tl ; dr.

The final tl ; dr must also be 70 % smaller than the original , that way there is a big gap between the original and the tl ; dr , hence only very long articles get posted on.


### Here starts the code:



```python
filename = 'summarytest.txt' #Enter Filename
```

### Loading the data into Python variable from the file. 


```python
file = open(filename,'r')
Text = ""
for line in file.readlines():
    Text+=str(line)
    Text+=" "
file.close()
```

### Removing non-printable characters from data and tokenizing the resultant test.


```python
import nltk
from nltk import word_tokenize
import string

def clean(text):
    printable = set(string.printable)
    text = filter(lambda x: x in printable, text) #filter funny characters, if any.
    return text

Cleaned_text = clean(Text)

text = word_tokenize(Cleaned_text)
case_insensitive_text = word_tokenize(Cleaned_text.lower())
```

### Sentence Segmentation

Senteces is a list of segmented sentences in the natural form with case sensitivity. This will be
necessary for displaying the summary later on.

Tokenized_sentences is a list of tokenized segmented sentences without case sensitivity. This will be
useful for text processing later on. 


```python
# Sentence Segmentation

sentences = []
tokenized_sentences = []
sentence = " "
for word in text:
    if word != '.':
        sentence+=str(word)+" "
    else:
        sentences.append(sentence.strip())
        tokenized_sentences.append(word_tokenize(sentence.lower().strip()))
        sentence = " "
        
```

### Lemmatization of tokenized words. 


```python
from nltk.stem import WordNetLemmatizer

def lemmatize(POS_tagged_text):
    
    wordnet_lemmatizer = WordNetLemmatizer()
    adjective_tags = ['JJ','JJR','JJS']
    lemmatized_text = []
    
    for word in POS_tagged_text:
        if word[1] in adjective_tags:
            lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0],pos="a")))
        else:
            lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0]))) #default POS = noun
    
    return lemmatized_text

#Pre_processing:

POS_tagged_text = nltk.pos_tag(case_insensitive_text)
lemmatized_text = lemmatize(POS_tagged_text)
```

### POS Tagging lemmatized words

This will be useful for generating stopwords. 


```python
Processed_text = nltk.pos_tag(lemmatized_text)
```

### Stopwords Generation

Based on the assumption that typically only nouns and adjectives are qualified as parts of keyword phrases, I will include any word that aren't tagged as a noun or adjective to the list of stopwords. (Note: Gerunds can often be important keywords or components of it. But including words tagged as 'VBG' (tag for present participles and gerunds) also include verbs of present continiuous forms which should be treated as stopwords. So I am not adding 'VBG' to list of POS that should not be treated as 'stopword-POSs'. Punctuations will be added to the same list (of stopwords). Additional the long list of stopwords from https://www.ranks.nl/stopwords are also added to the list. 


```python
def generate_stopwords(POS_tagged_text):
    stopwords = []
    
    wanted_POS = ['NN','NNS','NNP','NNPS','JJ','JJR','JJS','FW'] #may be add VBG too
    
    for word in POS_tagged_text:
        if word[1] not in wanted_POS:
            stopwords.append(word[0])
            
    punctuations = list(str(string.punctuation))
    stopwords = stopwords + punctuations
    
    stopword_file = open("long_stopwords.txt", "r")
    #Source = https://www.ranks.nl/stopwords

    for line in stopword_file.readlines():
        stopwords.append(str(line.strip()))

    return set(stopwords)

stopwords = generate_stopwords(Processed_text)
```

### Processing tokenized sentences.

Lemmatizing words in the sentences, and removing stopwords.



```python
processed_sentences = []

for sentence in tokenized_sentences:
    processed_sentence = []
    
    POS_tagged_sentence = nltk.pos_tag(sentence)
    lemmatized_sentence = lemmatize(POS_tagged_sentence)

    for word in lemmatized_sentence:
        if word not in stopwords:
            processed_sentence.append(word)
    processed_sentences.append(processed_sentence)

```

### Building Graph

TextRank is a graph based model, and thus it requires us to build a graph. Each sentence in the text will serve as a vertex for graph. The sentences will be represented in the vertices by their index in the list of sentences\processed_sentences. 

The weighted_edge matrix contains the information of edge connections among all the vertices. I am building a wieghted undirected edges.

weighted_edge[i][j] contains the weight of the connecting edge between the sentence vertex represented by sentence of index i and the sentence vertex represented by sentence of index j.

If weighted_edge[i][j] is zero, it means no edge connection is present between the sentences represented by index i and j.

There is a connection between the sentences (and thus between i and j which represents them) if the the value of the similarity between the two sentences is non-zero. The weight of the connection is the value of the similary between the connected vertices.

The value of the weighted_edge[i][j] is the determined by the similarity function. 

The similarity function is: 
(No. of overlapping words in sentence i and j)/(log(len(sentence_i)) + log(len(sentence_j))

The score of all vertices are intialized to one.
Self-connections are not considered, so weighted_edge[i][i] will be zero.


```python
import numpy as np
import math
from __future__ import division

sentence_len = len(processed_sentences)
weighted_edge = np.zeros((sentence_len,sentence_len),dtype=np.float32)

score = np.zeros((sentence_len),dtype=np.float32)

for i in xrange(0,sentence_len):
    score[i]=1
    for j in xrange(0,sentence_len):
        if j==i:
            weighted_edge[i][j]=0
        else:
            for word in processed_sentences[i]:
                if word in processed_sentences[j]:
                    weighted_edge[i][j] += processed_sentences[j].count(word)
            if weighted_edge[i][j]!=0:
                len_i = len(processed_sentences[i])
                len_j = len(processed_sentences[j])
                weighted_edge[i][j] = weighted_edge[i][j]/(math.log(len_i)+math.log(len_j))

```

### Calculating weighted summation of connections of a vertex

inout[i] will contain the sum of all the undirected connections\edges associated withe the vertex represented by i.


```python
inout = np.zeros((sentence_len),dtype=np.float32)

for i in xrange(0,sentence_len):
    for j in xrange(0,sentence_len):
        inout[i]+=weighted_edge[i][j]
```

### Scoring Vertices

The formula used for scoring a vertex represented by i is:

score[i] = (1-d) + d x [ Summation(j) ( (weighted_edge[i][j]/inout[j]) x score[j] ) ] where j belongs to the list of vertieces that has a connection with i. 

d is the damping factor.

The score is iteratively updated until convergence. 


```python
MAX_ITERATIONS = 50
d=0.85
threshold = 0.0001 #convergence threshold

for iter in xrange(0,MAX_ITERATIONS):
    prev_score = np.copy(score)
    
    for i in xrange(0,sentence_len):
        
        summation = 0
        for j in xrange(0,sentence_len):
            if weighted_edge[i][j] != 0:
                summation += (weighted_edge[i][j]/inout[j])*score[j]
                
        score[i] = (1-d) + d*(summation)
    
    if np.sum(np.fabs(prev_score-score)) <= threshold: #convergence condition
        print "Converging at iteration "+str(iter)+"...."
        break

```

    Converging at iteration 25....


### Displaying each sentence and its corresponding score:


```python
i=0
for sentence in sentences:
    print "Sentence:\n\n"+str(sentence)+"\nScore: "+str(score[i])+"\n\n"
    i+=1
```

    Sentence:
    
    Autotldr is a bot that uses SMMRY to create a TL ; DR/summary
    Score: 1.18796
    
    
    Sentence:
    
    I will put forth points that address the effects this bot has on the reddit community
    Score: 0.940897
    
    
    Sentence:
    
    It does n't create laziness , it only responds to it For the users who click the article link first and then return back to the comments , they will have already given their best attempt of fully reading the article
    Score: 1.05546
    
    
    Sentence:
    
    If they read it fully , the tl ; dr is unneeded and ignored
    Score: 0.828447
    
    
    Sentence:
    
    If they skimmed or skipped it , the bot will be useful to at least provide more context to the discussion , like an extension of the title
    Score: 0.834126
    
    
    Sentence:
    
    A large portion of users , especially in the defaulted mainstream subreddits like /r/politics , do n't even go to the article and go straight to the comments section
    Score: 0.990735
    
    
    Sentence:
    
    Most of the time , if I skip to the comments , I 'm able to illicit some sort of understanding of what the article was about from the title and discussion
    Score: 0.919072
    
    
    Sentence:
    
    However this bot is able to further improve my conjectured understanding
    Score: 0.727377
    
    
    Sentence:
    
    It did not make me skip it , it only helped me when I already decided to skip it
    Score: 0.15
    
    
    Sentence:
    
    The scenario in which this bot would create a significantly lazy atmosphere is if the tl ; dr were to be presented parallel to the main submission , in the same way the OP 's tl ; dr is presented right next to the long body of self post
    Score: 1.34784
    
    
    Sentence:
    
    Also , the tl ; dr becomes more prevalent/hidden as it will get upvoted/downvoted depending on how much of a demand there was for a tl ; dr in the first place
    Score: 1.13423
    
    
    Sentence:
    
    If it becomes the top voted comment than it has become more of a competitor to the original text for those who go to the comments first , but by then the thread has decided that a tl ; dr was useful and the bot delivered
    Score: 1.44883
    
    
    Sentence:
    
    It can make sophisticated topics more relevant to mainstream Reddit Sophisticated and important topics are usually accompanied or presented by long detailed articles
    Score: 1.19482
    
    
    Sentence:
    
    By making these articles and topics relevant to a larger portion of the Reddit userbase ( those who were n't willing to read the full article ) , it popularizes the topic and increases user participation
    Score: 1.43953
    
    
    Sentence:
    
    These posts will get more attention in the form of upvotes/downvotes , comments , and reposts
    Score: 0.367749
    
    
    Sentence:
    
    This will increase the prevalence of sophisticated topics in the mainstream subreddits and compete against cliched memes
    Score: 0.588198
    
    
    Sentence:
    
    This has the potential of re-sophisticating the topic discussion in the mainstream subreddits , as more hardcore redditors do n't have to retreat to a safe haven like /r/TrueReddit
    Score: 0.595447
    
    
    Sentence:
    
    This is a loose approximation and the magnitude of this effect is questionable , but I 'm not surprised if the general direction of the theory is correct
    Score: 0.15
    
    
    Sentence:
    
    I 'm not claiming this would improve reddit overnight , but instead very very gradually
    Score: 0.484023
    
    
    Sentence:
    
    It decreases Reddit 's dependency on external sites The bot doubles as a context provider for when a submission link goes down , is removed , or inaccessible at work/school
    Score: 1.00579
    
    
    Sentence:
    
    The next time the article you clicked gives you a 404 error , you wo n't have to depend on the users to provide context as the bot will have been able to provide that service at a much faster and consistent rate than a person
    Score: 1.0597
    
    
    Sentence:
    
    Additionally , an extended summary is posted in /r/autotldr , which acts as a perpetual archive and decreases how much reddit gets broken by external sites
    Score: 0.472735
    
    
    Sentence:
    
    Only useful tl ; dr 's are posted There are several criteria for a bot to post a tl ; dr
    Score: 1.64419
    
    
    Sentence:
    
    It posts the three most important sentences as decided by the core algorithm , and they must be within 450-700 characters total
    Score: 0.15
    
    
    Sentence:
    
    The final tl ; dr must also be 70 % smaller than the original , that way there is a big gap between the original and the tl ; dr , hence only very long articles get posted on
    Score: 1.34355
    
    
    Sentence:
    
    This way the likelihood of someone nonchalantly declaring `` TL ; DR '' in a thread and the bot posting in the same one is high
    Score: 1.15188
    
    
    Sentence:
    
    Also my strategy is to tell the bot to post in default , mainstream subreddits were the demand for a TL ; DR is much higher than /r/TrueReddit and /r/worldevents
    Score: 1.23764
    
    
    Sentence:
    
    Feel free to respond to these concepts and to raise your own
    Score: 0.15
    
    
    Sentence:
    
    Be polite , respectful , and clarify what you say
    Score: 0.15
    
    
    Sentence:
    
    Any offending posts to this rule will be removed
    Score: 0.15
    
    


### Summary Generation

Given some hyperparameters the program computes the summary_size. Sentences are then ranked in accordance to their corresponding scores. 

More precisely, the indices of the sentences are sorted based on the scores of their corresponding sentences. Based on size of the summary, indices of top 'summary_size' no. of highest scoring input sentences are chosen for generating the summary. 

The summary is then generated by presenting the sentences (whose indices were chosen) in a chronological order.

Note: I hardcoded the selection of the first statement (if the summary_size is computed to be more than 1) because the first sentence can usually serve as an introduction, and provide some context to the topic.


```python
Reduce_to_percent = 20
summary_size = int(((Reduce_to_percent)/100)*len(sentences))

if summary_size == 0:
    summary_size = 1

sorted_sentence_score_indices = np.flip(np.argsort(score),0)

indices_for_summary_results = sorted_sentence_score_indices[0:summary_size]

summary = "\n"

current_size = 0

if 0 not in indices_for_summary_results and summary_size!=1:
    summary+=sentences[0]
    summary+=".\n\n"
    current_size+=1


for i in xrange(0,len(sentences)):
    if i in indices_for_summary_results:
        summary+=sentences[i]
        summary+=".\n\n"
        current_size += 1
    if current_size == summary_size:
        break

print "\nSUMMARY: "
print summary

```

    
    SUMMARY: 
    
    Autotldr is a bot that uses SMMRY to create a TL ; DR/summary.
    
    The scenario in which this bot would create a significantly lazy atmosphere is if the tl ; dr were to be presented parallel to the main submission , in the same way the OP 's tl ; dr is presented right next to the long body of self post.
    
    If it becomes the top voted comment than it has become more of a competitor to the original text for those who go to the comments first , but by then the thread has decided that a tl ; dr was useful and the bot delivered.
    
    By making these articles and topics relevant to a larger portion of the Reddit userbase ( those who were n't willing to read the full article ) , it popularizes the topic and increases user participation.
    
    Only useful tl ; dr 's are posted There are several criteria for a bot to post a tl ; dr.
    
    The final tl ; dr must also be 70 % smaller than the original , that way there is a big gap between the original and the tl ; dr , hence only very long articles get posted on.
    
    
