# text_mining_clustering2021
Primer Práctico de la Materia Text Mining 2021

## Instalación

Primero clonar repo y abrir terminal dentro de la carpeta. Luego de esto seguir los pasos para conda o para pip.

### Conda

1. Crear y activar un conda environment con las configuraciones de `conda_requirements.txt`:

```bash
conda create --name textmining_clustering --file ./conda_requirements.txt
conda activate textmining_clustering
```

2. Profit


### Pip

1. Crear un virtualenv y activarlo utilizando la terminal
```bash
python -m venv venv
source venv/bin/activate
```

2. Instalar los requerimientos
```
pip install -r requirements.txt
```

3. Profit.

## Informe pt. 1

Nosotros decidimos trabajar sobre el dataset de LaVoz y como primer paso hicimos un análisis utilizando como feature los atriburos `POS`, `Tag` y `Dep` de los tokens dentro del cuerpo. Estas son características mayormente morfológicas de las palabras y se pueden ver ejemplos en [la documentación de spacy de linguistic features](https://spacy.io/usage/linguistic-features). Para poder generar los clusters utilizamos KMeans con 100 clusters.

<img width="1015" alt="imagen" src="https://user-images.githubusercontent.com/13922772/134432149-59ab1e17-2013-4987-b613-cfd7a7bb3df1.png">

Como era de esperar, los clusters formados generalmente eran un conjunto de adjetivos, verbos, sustantivos o nombres. Pero algo interesante que vimos en este paso es que el lematizador de spacy no funciona siempre como uno esperaría.

<img width="553" alt="imagen" src="https://user-images.githubusercontent.com/13922772/134432367-ba3a5f6f-2f64-434e-88e9-68985940e10b.png">

La imagen anterior es un recorte de un cluster que contiene palabras como "Aprender", "Hojear" y "Surfear" pero también contiene otros verbos conjugados erroneamente como "Informarar", "Ocurririr" y "Extraviarar". También contiene palabras que no fueron conjugadas al infinitivo como "Alcanzamos" o "Remitiendo".

Luego de algunas pruebas vimos que aparte de esto el lematizador también cambia todos los artículos al masculino, efectivamente no pasa todas las conjugaciones de los verbos al infinitivo y puede lematizar a dos palabras.

```python
doc = nlp("el la lo las los les le un una manejaría, manejarlo")

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
# manejaría
# manejar el
```

Para continuar con el análisis queríamos empezar a utilizar bigramas, trigramas e investigar la diferencia que podría hacer agregar la palabra (y/o atributos de la palabra) a la cual modifica utilizando [el atributo `head` de los tokens](https://spacy.io/usage/linguistic-features#navigating). Aparte de esto posiblemente utilizar [otro lemmatizer](https://github.com/pablodms/spacy-spanish-lemmatizer) y ver si este nos puede brindar mejores resultados.

Desgraciadamente, luego de horas de dejar diferentes computadoras y un colab corriendo por mucho tiempo sin resultado nos dimos cuenta de que era imposible hacer los análisis previamente mencionados sin realizar una fuerte reducción de tokens anteriormente. Para este momento ya era muy tarde para empezar el proceso de decidir e implementar el filtro de tokens.

Nuestro Jupyther funcional es `w_cluster_morph.ipynb` y `w_cluster_ctx.ipynb` contiene las pruebas donde intentamos agregar más contexto a cada token.

## Informe pt. 2

Luego de hacer un filtro por las palabras que tenian baja frecuencia, pudimos obtener clusters más claros. Pudimos obtener 4 clusters: Sustantivos Comunes (violeta), Sustantivos propios (Azul), Adjetivos (Naranja) y Verbos (Amarillo).

<img width="1025" alt="imagen" src="https://user-images.githubusercontent.com/13922772/135728723-c8f4406d-9512-45b0-b13a-888014d5edd0.png">

Luego de esto probamos con sumar bigramas, trigramas y el `text.head` de cada token para brindar un poco más de contexto pero no vimos mucha mejoría en la división de los clusters.


## Uso de `w_cluster.py`

El proyecto fué creado utilizando VS Code + [Su extención de Python](https://code.visualstudio.com/docs/languages/python). Esta última permite la creación de celdas de código o markdown utilizando "`# %%`" y `# %% [markdown]` respectivamente. Podemos ver en el siguiente ejemplo cómo se utilizan cada uno:

<img width="873" alt="imagen" src="https://user-images.githubusercontent.com/13922772/133910849-df831e64-e800-47b9-aba6-227444b6d0dd.png">

Y si corremos ambas celdas utilizando el botón de _"Run Cell"_, obtenemos lo siguiente en una nueva pestaña:

<img width="883" alt="imagen" src="https://user-images.githubusercontent.com/13922772/133911031-08863fc2-af2d-434c-bff1-ded3c2fef25f.png">

Utilizamos este método ya que brinda mejor claridad al momento de trabajar con el versionado del proyecto. 
