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

## Uso de `w_cluster.py`

El proyecto fué creado utilizando VS Code + [Su extención de Python](https://code.visualstudio.com/docs/languages/python). Esta última permite la creación de celdas de código o markdown utilizando "`# %%`" y `# %%[markdown]` respectivamente. 


## Disclaimer Sobre Lematización Con Spacy

Spacy lematiza todos los articulos al masculino y como no tenemos seguridad
que el comportamiento no se replique en otras palabras, no podemos confiar
en género de los lemas.

Aparte de esto, la lematización no funciona correctamente con algunas conjugaciones de verbos.

```python
doc = nlp("el la lo las los les le un una manejaría")

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
```