"""
Módulo de Preprocesamiento de Texto para NLP Disaster Tweets.

Este módulo contiene funciones y clases para limpiar y preparar
texto de tweets para su uso en modelos de ML y Deep Learning.

Uso:
    from src.preprocessing import TextPreprocessor

    preprocessor = TextPreprocessor()
    clean_text = preprocessor.preprocess("BREAKING: Fire in LA! http://t.co/xyz @LAFD")
"""

import re

import nltk  # type: ignore[import-not-found]
from nltk.corpus import stopwords  # type: ignore[import-not-found]
from nltk.stem import PorterStemmer, WordNetLemmatizer  # type: ignore[import-not-found]
from nltk.tokenize import (  # type: ignore[import-not-found]
    TweetTokenizer,
    word_tokenize,
)


# Descargar recursos NLTK necesarios
def download_nltk_resources():
    """Descargar recursos NLTK si no están disponibles."""
    resources = ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]
    for resource in resources:
        try:
            nltk.data.find(
                f"tokenizers/{resource}"
                if resource == "punkt"
                else (
                    f"corpora/{resource}"
                    if resource in ["stopwords", "wordnet"]
                    else f"taggers/{resource}"
                )
            )
        except LookupError:
            nltk.download(resource, quiet=True)


download_nltk_resources()


class TextPreprocessor:
    """
    Pipeline completo de preprocesamiento para tweets.

    Parámetros:
    -----------
    remove_stopwords : bool
        Si True, elimina stopwords del texto.
    lemmatize : bool
        Si True, aplica lematización. Si False y stem=True, aplica stemming.
    stem : bool
        Si True y lemmatize=False, aplica stemming.
    lowercase : bool
        Si True, convierte texto a minúsculas.
    remove_urls : bool
        Si True, elimina URLs del texto.
    remove_mentions : bool
        Si True, elimina menciones (@usuario).
    remove_hashtag_symbol : bool
        Si True, elimina el símbolo # pero conserva la palabra.
    remove_html : bool
        Si True, elimina tags HTML.
    remove_special_chars : bool
        Si True, elimina caracteres especiales y números.
    min_word_length : int
        Longitud mínima de palabras a conservar.

    Ejemplo:
    --------
    >>> preprocessor = TextPreprocessor(lemmatize=True, remove_stopwords=True)
    >>> preprocessor.preprocess("BREAKING: Huge fire in downtown LA! http://t.co/xyz")
    'breaking huge fire downtown la'
    """

    def __init__(
        self,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        stem: bool = False,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_hashtag_symbol: bool = True,
        remove_html: bool = True,
        remove_special_chars: bool = True,
        min_word_length: int = 2,
    ):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stem = stem
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtag_symbol = remove_hashtag_symbol
        self.remove_html = remove_html
        self.remove_special_chars = remove_special_chars
        self.min_word_length = min_word_length

        # Inicializar herramientas
        self.stop_words: set[str] = set(stopwords.words("english"))
        self.lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
        self.stemmer: PorterStemmer = PorterStemmer()
        self.tweet_tokenizer: TweetTokenizer = TweetTokenizer(
            preserve_case=False, reduce_len=True, strip_handles=True
        )

        # Patrones regex compilados para eficiencia
        self.url_pattern = re.compile(r"http\S+|www\S+|https\S+", re.MULTILINE)
        self.mention_pattern = re.compile(r"@\w+")
        self.hashtag_pattern = re.compile(r"#(\w+)")
        self.html_pattern = re.compile(r"<.*?>")
        self.special_char_pattern = re.compile(r"[^a-zA-Z\s]")
        self.whitespace_pattern = re.compile(r"\s+")

    def clean_text(self, text: str) -> str:
        """
        Limpieza básica de texto.

        Parámetros:
        -----------
        text : str
            Texto a limpiar.

        Retorna:
        --------
        str
            Texto limpio.
        """
        if not isinstance(text, str):
            return ""

        # Minúsculas
        if self.lowercase:
            text = text.lower()

        # Eliminar URLs
        if self.remove_urls:
            text = self.url_pattern.sub("", text)

        # Eliminar menciones
        if self.remove_mentions:
            text = self.mention_pattern.sub("", text)

        # Procesar hashtags (conservar palabra)
        if self.remove_hashtag_symbol:
            text = self.hashtag_pattern.sub(r"\1", text)

        # Eliminar HTML tags
        if self.remove_html:
            text = self.html_pattern.sub("", text)

        # Eliminar caracteres especiales y números
        if self.remove_special_chars:
            text = self.special_char_pattern.sub(" ", text)

        # Eliminar espacios múltiples
        text = self.whitespace_pattern.sub(" ", text).strip()

        return text

    def tokenize(self, text: str, use_tweet_tokenizer: bool = False) -> list[str]:
        """
        Tokenización del texto.

        Parámetros:
        -----------
        text : str
            Texto a tokenizar.
        use_tweet_tokenizer : bool
            Si True, usa TweetTokenizer optimizado para tweets.

        Retorna:
        --------
        List[str]
            Lista de tokens.
        """
        if use_tweet_tokenizer:
            raw_tokens = self.tweet_tokenizer.tokenize(text)
        else:
            raw_tokens = word_tokenize(text)

        tokens: list[str] = [str(tok) for tok in raw_tokens]
        return tokens

    def remove_stops(self, tokens: list[str]) -> list[str]:
        """Eliminar stopwords de la lista de tokens."""
        filtered: list[str] = [
            token for token in tokens if token not in self.stop_words
        ]
        return filtered

    def lemmatize_tokens(self, tokens: list[str]) -> list[str]:
        """Aplicar lematización a los tokens."""
        lemmatized: list[str] = [self.lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized

    def stem_tokens(self, tokens: list[str]) -> list[str]:
        """Aplicar stemming a los tokens."""
        return [self.stemmer.stem(t) for t in tokens]

    def filter_by_length(self, tokens: list[str]) -> list[str]:
        """Filtrar tokens por longitud mínima."""
        return [t for t in tokens if len(t) >= self.min_word_length]

    def preprocess(self, text: str, return_tokens: bool = False):
        """
        Pipeline completo de preprocesamiento.

        Parámetros:
        -----------
        text : str
            Texto a preprocesar.
        return_tokens : bool
            Si True, retorna lista de tokens. Si False, retorna string.

        Retorna:
        --------
        str o List[str]
            Texto preprocesado o lista de tokens.
        """
        # Limpieza
        text = self.clean_text(text)

        # Tokenización
        tokens = self.tokenize(text)

        # Eliminar stopwords
        if self.remove_stopwords:
            tokens = self.remove_stops(tokens)

        # Lematización o Stemming
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        elif self.stem:
            tokens = self.stem_tokens(tokens)

        # Filtrar por longitud
        tokens = self.filter_by_length(tokens)

        if return_tokens:
            return tokens
        return " ".join(tokens)

    def preprocess_batch(self, texts: list[str], return_tokens: bool = False) -> list:
        """
        Preprocesar múltiples textos.

        Parámetros:
        -----------
        texts : List[str]
            Lista de textos a preprocesar.
        return_tokens : bool
            Si True, retorna lista de listas de tokens.

        Retorna:
        --------
        List
            Lista de textos preprocesados o listas de tokens.
        """
        return [self.preprocess(text, return_tokens) for text in texts]


def clean_tweet_simple(text: str) -> str:
    """
    Función simple de limpieza de tweets (sin dependencias pesadas).

    Útil para preprocesamiento rápido sin NLTK completo.

    Parámetros:
    -----------
    text : str
        Tweet a limpiar.

    Retorna:
    --------
    str
        Tweet limpio.
    """
    if not isinstance(text, str):
        return ""

    # Minúsculas
    text = text.lower()

    # Eliminar URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Eliminar menciones
    text = re.sub(r"@\w+", "", text)

    # Procesar hashtags
    text = re.sub(r"#(\w+)", r"\1", text)

    # Eliminar HTML
    text = re.sub(r"<.*?>", "", text)

    # Eliminar caracteres especiales
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Eliminar espacios múltiples
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de uso del preprocesador
    sample_tweets = [
        "BREAKING: Massive earthquake hits California! Stay safe everyone! http://t.co/xyz @CNN #earthquake",
        "My new mixtape is so fire it's causing earthquakes 🔥🔥🔥 @DJ_Fire",
        "Prayers for the victims of the flooding in Houston. #HoustonStrong",
        "I'm dying of laughter at this video 😂😂😂 #dead",
    ]

    preprocessor = TextPreprocessor(lemmatize=True, remove_stopwords=True)

    print("=" * 60)
    print("DEMO: TextPreprocessor")
    print("=" * 60)

    for tweet in sample_tweets:
        clean = preprocessor.preprocess(tweet)
        print(f"\nOriginal: {tweet}")
        print(f"Limpio:   {clean}")
