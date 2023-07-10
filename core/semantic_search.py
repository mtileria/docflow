import pandas as pd
from sentence_transformers import SentenceTransformer, util

def parse_hits(hit_list):
    first = hit_list[0]['corpus_id'],hit_list[0]['score']
    second = hit_list[1]['corpus_id'],hit_list[1]['score']
    third = hit_list[2]['corpus_id'],hit_list[2]['score']
    return list(first + second + third)


def searc_in_corpus(corpus_embeddings,embedder, query, k=3 ):
    query_embedding = embedder.encode(query)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=k)
    return hits

if __name__ == '__main__':
    # use a 1-layer neural network as example
    learning = 'semantic-search'
