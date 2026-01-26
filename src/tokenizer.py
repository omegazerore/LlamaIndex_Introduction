from typing import List, Union, Callable
from tqdm.auto import tqdm

import jieba
from bm25s.tokenization import Tokenized
from nltk import corpus

def chinese_tokenize(
    texts: Union[str, List[str]],
    lower: bool = True,
    token_pattern: str = r"(?u)\b\w\w+\b",
    stopwords: Union[str, List[str]] = "zh",
    stemmer: Callable = None,  
    return_ids: bool = True,
    show_progress: bool = True,
    leave: bool = False,
    allow_empty: bool = True,
) -> Union[List[List[str]], Tokenized]:
    if isinstance(texts, str):
        texts = [texts]

    language_map = {
        "zh":"chinese",
        "en":"english"
    }
    
    if isinstance(stopwords, str):
        stopword_list = corpus.stopwords.words(language_map[stopwords])
    else:
        stopword_list = []
        for stopword in stopwords:
            stopword_list.extend(corpus.stopwords.words(language_map[stopword]))
    corpus_ids = []
    token_to_index = {}
    for text in tqdm(
        texts, desc="Split strings", leave=leave, disable=not show_progress
    ):  
        if lower:
            text = text.lower()

        splitted = jieba.lcut_for_search(text)
        if allow_empty is False and len(splitted) == 0:
            splitted = [""]

        doc_ids = []
        for token in splitted:
            if token in stopword_list:
                continue 
            token_to_index.setdefault(token, len(token_to_index))
            doc_ids.append(token_to_index[token])
        corpus_ids.append(doc_ids)
    if return_ids:
        return Tokenized(ids=corpus_ids, vocab=token_to_index)
    else:
        unique_tokens = list(token_to_index.keys())
        for i, token_ids in enumerate(
            tqdm(
                corpus_ids,
                desc="Reconstructing token strings",
                leave=leave,
                disable=not show_progress,
            )
        ):
            corpus_ids[i] = [unique_tokens[token_id] for token_id in token_ids]

        return corpus_ids