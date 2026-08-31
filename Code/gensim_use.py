"""
This module contains the functions for using the models generate with gensim_train.py.
"""
import logging
import natsort
import os
from constants import OUTPUT_DATA_OSX
from file_tools import parse_model_name
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def use_model(model_path: str, output_path: str, word: str, top: int) -> dict:
    """_summary_

    Args:
        model_path (str): Path to where models are stored.
        output_path (str): Path to where the results will be stored.
        word (str): A wordform.
        top (int): The number of most similar words to return.

    Returns:
        dict: A dictionary with the most similar words and their similarity scores.
    """
    all_models = parse_model_name(model_path)
    if len(all_models) == 0:
        print("No models!")
        return None
    else:
        all_results = {}
        vectors = {}
        print(all_models)
        print(f"Calculating the similarity of {word}\n")

        output_file = os.path.join(output_path, f"results_{word}_top{top}.csv")
        with open(output_file, "w", encoding="utf-8") as target:
            for idx, model in enumerate(natsort.os_sorted(all_models)):
                model_name = f"model_{'_'.join([str(i) for i in model[1].values()])}"
                model_vectors = get_tmpfile(model[0])
                used_model = KeyedVectors.load(model_vectors, mmap='r')
                results = used_model.wv.most_similar(word, topn=top)
                vectors[model_name] = (word, used_model.wv[word])
                print(results)
                all_results[model_name] = results
                print(model_name)
                target.write(f"{model_name}\n")
                for i in results:
                    print(f"{i[0]}\t{i[1]}")
                    target.write(f"{i[0]}\t{i[1]}\n")
                print("============\n")
                target.write("============\n")
        print(f"Save results to {target.name}")
    return all_results, vectors
