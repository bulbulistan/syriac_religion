import faulthandler
import logging
import os
import pickle
import time
import gensim.models
from gensim.test.utils import get_tmpfile
from gensim.similarities import Similarity
from gensim.models import word2vec
from constants import MODEL_PATH_OSX

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def train_gensim(alltexts: list, vector_size: int, window: int, min_count: int, epochs: int, *special_name: str) -> None:
    v = vector_size
    w = window
    m = min_count
    ep = epochs

    faulthandler.enable()
    t = time.process_time()
    
    print("Done, the processing took " + str(time.process_time() - t) + " seconds")

    print("Creating the model...")
    t = time.process_time()

    model_cbow = word2vec.Word2Vec(alltexts, vector_size=v, window=w, min_count=m, workers=4, epochs=ep)
    print("CBOW model created, the processing took " + str(time.process_time() - t) + " seconds")
    model_skipgram = word2vec.Word2Vec(alltexts, vector_size=v, window=w, min_count=m, \
                                   workers=4, epochs=ep, sg=1)
    print("Skipgram created, the processing took " + str(time.process_time() - t) + " seconds")

    print("Saving the model to disk")
    model_cbow_name = f"{special_name[0]}_simthovectors-cbow-size{v}-window{w}-min{m}.kv"
    model_skipgram_name = f"{special_name[0]}_simthovectors-skipgram-size{v}-window{w}-min{m}.kv"
    fname_model_cbow = get_tmpfile(os.path.join(MODEL_PATH_OSX, model_cbow_name))
    fname_model_skipgram = get_tmpfile(os.path.join(MODEL_PATH_OSX, model_skipgram_name))
    model_cbow.save(fname_model_cbow)
    model_skipgram.save(fname_model_skipgram)
    print("Models saved to the disk")


def train_fasttext(alltexts: list, vector_size: int, window: int, min_count: int, sg: int, epochs: int) -> None:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # _vs = 600 _mc = 3 _w = 64 _s = 1 _e = 25

    _vs = vector_size
    _mc = min_count
    _w = window
    _s = sg
    _e = epochs

    model_vectors_file = f"fasttext_{_vs}_{_mc}_{_w}_{_s}_{_e}.txt"
    model_binary_file = f"fasttext_{_vs}_{_mc}_{_w}_{_s}_{_e}.bin"
    model = gensim.models.FastText(sentences=alltexts, vector_size=_vs, min_count=_mc, window=_w, sg=_s, epochs=_e)
    model.wv.save_word2vec_format(os.path.join(MODEL_PATH_OSX, model_vectors_file), binary=False)
    print("Model saved as ", model_vectors_file)
    model.save(os.path.join(MODEL_PATH_OSX, model_binary_file))
