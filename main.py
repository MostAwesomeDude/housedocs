import numpy as np

from faiss import IndexFlatIP, normalize_L2
from sentence_transformers import SentenceTransformer

from llm_rs import AutoModel, KnownModels
from llm_rs.config import GenerationConfig

from fastapi import FastAPI

print("Loading sentence transformer...")
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

print("Loading document...")
with open("docs.md") as handle:
    corpus = handle.read()
indices = findIndices(corpus, ("\n\n", ".", "?", "!"))
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

print("Loading generative model...")
model = AutoModel.from_pretrained("rustformers/mpt-7b-ggml",
                                  model_file="mpt-7b-instruct-q4_0-ggjt.bin")

print("Loading HTTP API...")
app = FastAPI()

@app.get("/")
def root(): return {}

@app.get("/search")
def search(sentence):
    ss = mergeSpans(searchFor(sentence))
    return [corpus[i:j].strip() for (i, j) in ss]

PROMPT_TEMPLATE = """Here is an example query and response. The response includes details from the excerpts.
Query: {q}
Excerpts: {spans}
Response:"""

CONFIG = GenerationConfig(
    max_new_tokens=200,
    stop_words=["\n"],
)

@app.get("/explain")
def explain(sentence):
    ss = mergeSpans(searchFor(sentence))
    spans = "\n * ".join([corpus[i:j].strip() for (i, j) in ss])
    prompt = PROMPT_TEMPLATE.format(q=sentence, spans=spans)
    return {
        "text": model.generate(prompt, generation_config=CONFIG).text,
    }
