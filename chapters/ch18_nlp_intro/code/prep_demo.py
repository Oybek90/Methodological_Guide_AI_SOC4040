# ci: skip
"""Tokenisation, stop words, stemming, lemmatisation with NLTK (needs nltk data)."""
import nltk
for pkg in ("punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger_eng"):
    nltk.download(pkg, quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer

text = "Dr. Smith isn't running; he's reading 3 papers. Great!"
print("whitespace :", text.split())
print("sentences  :", sent_tokenize(text))
print("word_tok   :", word_tokenize(text))
print("characters :", list("NLP"))

tw = "hopeless for tmr :( #sad @friend"
print("word_tok   :", word_tokenize(tw))
print("TweetTok   :", TweetTokenizer().tokenize(tw))

sw = set(stopwords.words("english"))
s = "This is an example of a sentence that has stop words."
toks = [w for w in word_tokenize(s.lower()) if w.isalpha()]
print("stop words :", [w for w in toks if w not in sw])
print("NLTK list size:", len(sw), "| 'not' in list:", "not" in sw)
neg = "this movie is not good".split()
print("negation   :", [w for w in neg if w not in sw])

ps, lem = PorterStemmer(), WordNetLemmatizer()
for w in ["running", "runner", "ran", "easily", "studies", "university", "universe"]:
    print(f"{w:11s} stem={ps.stem(w):9s} lemma(v)={lem.lemmatize(w, 'v'):9s} lemma(n)={lem.lemmatize(w)}")
print("better:", ps.stem("better"), lem.lemmatize("better"), lem.lemmatize("better", "a"))
print("feet  :", ps.stem("feet"), lem.lemmatize("feet"))
