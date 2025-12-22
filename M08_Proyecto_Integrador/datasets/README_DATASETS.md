# Datasets para el Proyecto Capstone NLP

## Dataset Principal: Disaster Tweets

### Descarga

1. Ir a [Kaggle - NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started/data)
2. Descargar los archivos:
   - `train.csv` (7,613 tweets etiquetados)
   - `test.csv` (3,263 tweets sin etiqueta)
3. Colocar en esta carpeta

### Estructura del Dataset

```
train.csv columns:
- id: identificador único
- keyword: palabra clave del tweet (puede estar vacía)
- location: ubicación del usuario (puede estar vacía)
- text: texto del tweet
- target: 1 = desastre real, 0 = no desastre

test.csv columns:
- id, keyword, location, text (sin target)
```

### Estadísticas

| Métrica | Valor |
|---------|-------|
| Total tweets (train) | 7,613 |
| Clase 0 (no desastre) | 4,342 (57%) |
| Clase 1 (desastre) | 3,271 (43%) |
| Tweets con keyword | 7,552 (99.2%) |
| Tweets con location | 5,080 (66.7%) |

---

## Dataset Secundario: GloVe Embeddings

### Descarga

1. Ir a [Stanford GloVe](https://nlp.stanford.edu/projects/glove/)
2. Descargar `glove.6B.zip` (822 MB)
3. Extraer y usar `glove.6B.100d.txt` (100 dimensiones)

### Alternativa: Descargar con wget

```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
# Usar: glove.6B.100d.txt (100 dimensiones, 400K palabras)
```

---

## Nota sobre .gitignore

Los archivos `.csv` están en `.gitignore` para evitar subir datos a Git.
Debes descargar los datasets manualmente.
