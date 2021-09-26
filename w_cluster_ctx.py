# %% [markdown]
# # Text Mining Práctico Clustering
# [Consigna](https://sites.google.com/unc.edu.ar/textmining2021/pr%C3%A1ctico/clustering?authuser=0)
# %%
# Setup inicial
import numpy as np
import pandas as pd
import plotly.express as px
import spacy
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.feature_extraction import DictVectorizer
from sklearn.manifold import TSNE
from spacy.tokens import DocBin
from tqdm import tqdm


nlp = spacy.load("es_core_news_sm")
nlp.Defaults.stop_words.add("e")
nlp.Defaults.stop_words.add("y")
nlp.Defaults.stop_words.add("a")
nlp.Defaults.stop_words.add("o")
nlp.Defaults.stop_words.add("u")
nlp.Defaults.stop_words.add("o")
nlp.Defaults.stop_words.add("etcétera")
nlp.Defaults.stop_words.add("etc")


DATASET_PATH = "./lavoztextodump.txt"
DOC_BIN_PATH = "./articles_doc_bin.spacy"
# 12936 es el número de documentos que vamos a generar.
TOTAL_ARTICLES = 12936

# %%
# Abrir dataset y crear librereia de objetos docs
# para no tener que crearla nuevamente sin necesidad.
# 
# EN CASO DE TENER EL ARCHIVO:
# 'articles_doc_bin.spacy'
# ESTA CELDA NO ES NECESARIA
articles_doc_bin = DocBin()

# Agregamos una barra de progreso tqdm
doc_bin_progress_bar = tqdm(total=TOTAL_ARTICLES)
with open(DATASET_PATH, "r") as dataset_file:
    while True:
        article_divider = dataset_file.readline()
        article_title = dataset_file.readline()
        article_content = dataset_file.readline()
        if not article_content:
            # Si no existe, llegamos al final del archivo
            break  # EOF
        # Agregar el objeto Doc compuesto por el titulo y contenido
        articles_doc_bin.add(
            nlp("{} {}".format(article_title, article_content)))
        # Actializar barra de progreso
        doc_bin_progress_bar.update()

articles_doc_bin.to_disk(DOC_BIN_PATH)

# %%
# Cargar archivo con los objetos Doc.
articles_doc_bin = DocBin().from_disk(DOC_BIN_PATH)

# %%
# Armar diccionario de tokens con features
words_feature_dict = dict()

word_feature_dict_progress_bar = tqdm(total=TOTAL_ARTICLES)
for doc in articles_doc_bin.get_docs(nlp.vocab):
    sents = [sent for sent in doc.sents]
    for sent in sents:
        for token in sent:
            # Si la palabra es stopword o no está compuesta unicamente
            # por letras 
            if token.is_stop or not token.is_alpha:
                continue
            # Obtener lemma de la palabra. Algunos lemmas son "VERBO él"
            # por lo cual vamos a tomar solo la parte del verbo.
            w_lemma = token.lemma_.split(" ")[0]
            # Obtener features de la palabra o devolver un diccionario vacío
            word_feature_dict = words_feature_dict.get(w_lemma, {})

            features = [
                "POS__" + token.pos_,
                "DEP__" + token.dep_,
                "TAG__" + token.tag_,
                "LEMM_" + w_lemma,
                "HEAD_" + token.head.lemma_.split(" ")[0],
                "count",
            ]
            for f in features:
                # si la feature está definida y sumar uno
                # si la feature no está definida, devolver 0 y sumar 1
                word_feature_dict[f] = word_feature_dict.get(f, 0) + 1
            
            right_t = next(token.rights, None)
            if right_t and not right_t.is_punct and not right_t.is_stop:
                if right_t.is_alpha:
                    r_lemm = right_t.lemma_.split(" ")[0]
                    feat_name = "RLEM_" + r_lemm
                    word_feature_dict[feat_name] = word_feature_dict.get(
                        feat_name, 0) + 1
                else:
                    r_lemm = "NUM__"
                    word_feature_dict[r_lemm] = word_feature_dict.get(
                        r_lemm, 0) + 1

            left_ts = [t for t in token.lefts]
            if left_ts:
                left_t = left_ts[-1]
                if not left_t.is_punct and not left_t.is_stop:
                    if left_t.is_alpha:
                        l_lemm = left_t.lemma_.split(" ")[0]
                        feat_name = "LLEM_" + l_lemm
                        word_feature_dict[feat_name] = word_feature_dict.get(
                            feat_name, 0) + 1
                    else:
                        l_lemm = "NUM__"
                        word_feature_dict[l_lemm] = word_feature_dict.get(
                            l_lemm, 0) + 1
            
            words_feature_dict[w_lemma] = word_feature_dict

    word_feature_dict_progress_bar.update()
# %%
filtered_words_feature_dict = dict()

for w, f in words_feature_dict.items():
    if f["count"] > 70:
        f.pop("count")
        filtered_words_feature_dict[w] = f

# %%
# Crear lista con las features de cada token y
# un diccionario que guarde la posición de cada token dentro
# de la lista
words_feature_list = []
words_ids = {}
wid = 0
for word in filtered_words_feature_dict:
    if len(word) > 0:
        words_ids[word] = wid
        wid += 1
        words_feature_list.append(filtered_words_feature_dict[word])

# %%
# Utilizar DictVectorizer para crear una matriz "scipy.sparse"
# para utilizar los modelos de sklearn.
v = DictVectorizer(sparse=False)
matrix = v.fit_transform(words_feature_list)

# %%
# normalizar la matriz
matrix_normed = preprocessing.normalize(matrix)

# %%
# Calcular varianza de cada columna
variances = np.square(matrix_normed).mean(axis=0) - \
    np.square(matrix_normed.mean(axis=0))

# %%
# Quitar las columnas con poca varianza
threshold_v = 0.001
red_matrix = np.delete(matrix_normed, np.where(
    variances < threshold_v), axis=1)

# %%
# Proyectar matriz n dimensional en 2 dimensiones
tsne = TSNE(n_components=2, random_state=0)
matrix_dicc2d = tsne.fit_transform(red_matrix)

# %%
# Crear DataFrame que contenga token, su posición en x y en y
pointsspacy = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in tqdm([
            (word, matrix_dicc2d[words_ids[word]])
            for word in words_ids
        ])
    ],
    columns=["word", "x", "y"]
)

# %%
# Mostrar scatterplot de cada valor
fig_matrix = px.scatter(pointsspacy, x="x", y="y", hover_data=['word'])
fig_matrix.show()
# %%
# Utilizamos Kmeans para crear clusters de palabras
kmeans = KMeans(n_clusters=6).fit(red_matrix)

# %%
# Creamos un DataFrame que contiene el token, las posiciones x e y y 
# el cluster de cada token.
pointscluster = pd.DataFrame(
    [
        (word, coords[0], coords[1], cluster)
        for word, coords, cluster in tqdm([
            (word, matrix_dicc2d[words_ids[word]],
             kmeans.labels_[words_ids[word]])
            for word in words_ids
        ])
    ],
    columns=["word", "x", "y", "c"]
)

# %%
# Mostrar scatterplot de cada valor y con diferentes colores de dependiendo
# del cluster al que pertenecen
fig_clusters = px.scatter(
    pointscluster, x="x", y="y", color="c", hover_data=['word'])
fig_clusters.show()

# %%
