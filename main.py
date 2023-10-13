import numpy as np

from faiss import IndexFlatIP, normalize_L2
from sentence_transformers import SentenceTransformer

from fastapi import FastAPI

embedder = SentenceTransformer("all-miniLM-L6-v2")
EMBEDDING_WIDTH = 384
index = IndexFlatIP(EMBEDDING_WIDTH)

def findIndices(s, cs): return [i for (i, c) in enumerate(s) if c in cs]
def staggerIndices(indices):
    l = indices[::2]
    return list(zip(l[::2], l[1::2]))

def normedEmbeds(arr):
    rv = np.array(arr, dtype="float32")
    normalize_L2(rv)
    return rv

with open("docs.md") as handle:
    corpus = handle.read()
indices = findIndices(corpus, ("\n", ".", "?", "!"))
spans = staggerIndices(indices) + staggerIndices(indices[1:])
sentences = [corpus[i:j] for (i, j) in spans]
index.add(normedEmbeds(embedder.encode(sentences)))

def mergeSpans(ss):
    ss.sort()
    rv = [ss[0]]
    for (i, j) in ss[1:]:
        ri, rj = rv.pop()
        if ri <= i <= rj: rv.append((ri, max(j, rj)))
        else:
            rv.append((ri, rj))
            rv.append((i, j))
    return rv

def searchFor(sentence):
    xq = embedder.encode([sentence])
    D, I = index.search(xq, 5)
    return mergeSpans([spans[i] for i in I[0]])

app = FastAPI()

@app.get("/")
def root(): return {}

@app.get("/search")
def search(sentence):
    ss = mergeSpans(searchFor(sentence))
    return [corpus[i:j].strip() for (i, j) in ss]
