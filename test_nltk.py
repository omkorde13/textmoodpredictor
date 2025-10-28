from nltk.tokenize import sent_tokenize, word_tokenize

text = "This is a test sentence."

# Use explicit sent_tokenize with correct language param
sentences = sent_tokenize(text, language='english')

tokens = []
for s in sentences:
    tokens.extend(word_tokenize(s))

print(tokens)
