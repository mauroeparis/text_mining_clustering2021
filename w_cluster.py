# %%
import spacy

nlp = spacy.load("es_core_news_sm")

# %%
DATASET_PATH = "./lavoztextodump.txt"
dataset = None

with open(DATASET_PATH, "r") as dataset_file:
    dataset = dataset_file.read()

# %%
doc = nlp(dataset)

# %%[markdown]
```python
doc = nlp("el la lo las los les le un una")

for token in doc:
    print(token.lemma_)
# out:
# el
# el
# él
# el
# él
# él
# él
# uno
# uno
```