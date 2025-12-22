"""
Módulo LSTM para Clasificación de Texto - Proyecto Disaster Tweets
===================================================================

M08 - Proyecto Integrador: NLP con Deep Learning
Curso Alineado: CSCA 5642 - Deep Learning

Este módulo implementa un modelo LSTM Bidireccional para clasificación
de tweets como desastres o no desastres.

Dependencias:
    pip install tensorflow numpy

Autor: CU Boulder MS AI Program
Versión: 1.0.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# =============================================================================
# ⚠️ ADVERTENCIA IMPORTANTE: REQUISITOS DE HARDWARE
# =============================================================================
#
# Este modelo LSTM requiere recursos computacionales significativos:
#
# 🔴 ENTRENAMIENTO CON GPU (Recomendado):
#    - NVIDIA GPU con CUDA 11.x+ y cuDNN 8.x+
#    - Mínimo 4GB VRAM para batch_size=32
#    - Tiempo estimado: ~2-5 min/época en GPU moderna
#
# 🟡 ENTRENAMIENTO EN CPU (Posible pero lento):
#    - Tiempo estimado: ~15-30 min/época
#    - Reducir batch_size a 16 o menos
#    - Considerar usar LSTM unidireccional (no bidireccional)
#
# 🟢 ALTERNATIVA LIGERA: Si tu máquina no tiene GPU, considera:
#    1. Usar Google Colab (GPU gratuita)
#    2. Reducir vocab_size y embedding_dim
#    3. Usar modelo más simple (ver SimpleLSTM abajo)
#
# =============================================================================

# Verificar disponibilidad de GPU
try:
    import tensorflow as tf

    GPUS_AVAILABLE = len(tf.config.list_physical_devices("GPU"))
    if GPUS_AVAILABLE > 0:
        print(f"✅ GPU detectada: {GPUS_AVAILABLE} dispositivo(s)")
        for gpu in tf.config.list_physical_devices("GPU"):
            print(f"   - {gpu}")
    else:
        print("⚠️ No se detectó GPU. El entrenamiento será más lento.")
        print("   Considera usar Google Colab o reducir la complejidad del modelo.")

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    GPUS_AVAILABLE = 0
    print("❌ TensorFlow no instalado. Ejecutar: pip install tensorflow")


class DisasterTweetLSTM:
    """
    Modelo LSTM Bidireccional para clasificación de Disaster Tweets.

    Este modelo implementa una arquitectura de red neuronal recurrente
    con las siguientes capas:
    1. Embedding: Convierte tokens a vectores densos
    2. SpatialDropout1D: Regularización en dimensión temporal
    3. Bidirectional LSTM: Captura contexto en ambas direcciones
    4. GlobalMaxPooling1D: Extrae features más relevantes
    5. Dense + Dropout: Capas fully-connected con regularización
    6. Output: Sigmoid para clasificación binaria

    Attributes
    ----------
    vocab_size : int
        Tamaño del vocabulario (número de tokens únicos).
    embedding_dim : int
        Dimensionalidad de los embeddings de palabras.
    max_length : int
        Longitud máxima de secuencia (padding/truncation).
    lstm_units : int
        Número de unidades en la capa LSTM.
    dropout_rate : float
        Tasa de dropout para regularización.
    use_pretrained_embeddings : bool
        Si True, permite cargar embeddings pre-entrenados (GloVe).
    model : tf.keras.Model | None
        Modelo Keras compilado después de llamar a build().

    Examples
    --------
    >>> lstm = DisasterTweetLSTM(vocab_size=10000, max_length=128)
    >>> lstm.build()
    >>> lstm.compile(learning_rate=1e-3)
    >>> history = lstm.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

    Notes
    -----
    💡 Conexión con M03 - Cálculo:
        El optimizador Adam aplica descenso de gradiente adaptativo.
        La función de pérdida binary_crossentropy se deriva del MLE
        para distribuciones Bernoulli.

    💡 Conexión con M07 - Deep Learning:
        La arquitectura bidireccional permite que el modelo "vea"
        el contexto tanto anterior como posterior de cada palabra,
        crucial para entender el significado en NLP.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 100,
        max_length: int = 128,
        lstm_units: int = 64,
        dropout_rate: float = 0.3,
        use_pretrained_embeddings: bool = False,
    ) -> None:
        """
        Inicializa el modelo LSTM con los hiperparámetros especificados.

        Parameters
        ----------
        vocab_size : int, default=10000
            Tamaño del vocabulario. Tokens fuera del vocabulario se mapean a <UNK>.
        embedding_dim : int, default=100
            Dimensión de los word embeddings. Usar 100/200/300 para GloVe.
        max_length : int, default=128
            Longitud máxima de secuencia. Tweets típicamente < 280 caracteres.
        lstm_units : int, default=64
            Unidades en cada dirección del BiLSTM (total: 2*lstm_units).
        dropout_rate : float, default=0.3
            Probabilidad de dropout. Rango típico: 0.2-0.5.
        use_pretrained_embeddings : bool, default=False
            Si True, la capa de embedding espera matriz de pesos externos.
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.use_pretrained_embeddings = use_pretrained_embeddings

        self.model: tf.keras.Model | None = None
        self._embedding_matrix: NDArray[np.float32] | None = None
        self._history: dict[str, list[float]] | None = None

    def build(
        self,
        embedding_matrix: NDArray[np.float32] | None = None,
    ) -> tf.keras.Model:
        """
        Construye la arquitectura del modelo LSTM.

        Parameters
        ----------
        embedding_matrix : NDArray[np.float32] | None, default=None
            Matriz de embeddings pre-entrenados (vocab_size, embedding_dim).
            Si se proporciona, la capa de embedding se inicializa con estos pesos.

        Returns
        -------
        tf.keras.Model
            Modelo Keras construido (sin compilar).

        Raises
        ------
        ImportError
            Si TensorFlow no está instalado.
        ValueError
            Si embedding_matrix tiene dimensiones incorrectas.

        Notes
        -----
        💡 Conexión con M02 - Álgebra Lineal:
            La matriz de embeddings es una transformación lineal que mapea
            índices one-hot a vectores densos en un espacio de menor dimensión.
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow requerido. Instalar: pip install tensorflow")

        from tensorflow.keras.layers import (
            LSTM,
            Bidirectional,
            Dense,
            Dropout,
            Embedding,
            GlobalMaxPooling1D,
            Input,
            SpatialDropout1D,
        )
        from tensorflow.keras.models import Model

        # Validar embedding_matrix si se proporciona
        if embedding_matrix is not None:
            expected_shape = (self.vocab_size, self.embedding_dim)
            if embedding_matrix.shape != expected_shape:
                raise ValueError(
                    f"embedding_matrix debe tener shape {expected_shape}, "
                    f"pero tiene {embedding_matrix.shape}"
                )
            self._embedding_matrix = embedding_matrix

        # --- Arquitectura del Modelo ---
        # Input: secuencias de índices de tokens
        inputs = Input(shape=(self.max_length,), dtype="int32", name="input_tokens")

        # Capa de Embedding
        if self._embedding_matrix is not None:
            # Usar embeddings pre-entrenados (GloVe, Word2Vec, etc.)
            x = Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                weights=[self._embedding_matrix],
                input_length=self.max_length,
                trainable=False,  # Congelar embeddings pre-entrenados
                name="pretrained_embedding",
            )(inputs)
        else:
            # Entrenar embeddings desde cero
            x = Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_length,
                name="trainable_embedding",
            )(inputs)

        # SpatialDropout1D: dropout en la dimensión de features
        # Más efectivo que dropout estándar para secuencias
        x = SpatialDropout1D(self.dropout_rate, name="spatial_dropout")(x)

        # Bidirectional LSTM
        # 💡 Conexión M07: BiLSTM procesa la secuencia en ambas direcciones
        x = Bidirectional(
            LSTM(
                self.lstm_units,
                return_sequences=True,  # Retornar secuencia completa para pooling
                dropout=self.dropout_rate,
                recurrent_dropout=0.1,  # Dropout en conexiones recurrentes
            ),
            name="bidirectional_lstm",
        )(x)

        # GlobalMaxPooling1D: extrae el valor máximo de cada feature
        # Más robusto que flatten para secuencias de longitud variable
        x = GlobalMaxPooling1D(name="global_max_pool")(x)

        # Capas Dense con Dropout
        x = Dense(64, activation="relu", name="dense_1")(x)
        x = Dropout(self.dropout_rate, name="dropout_dense")(x)

        # Output: clasificación binaria
        outputs = Dense(1, activation="sigmoid", name="output")(x)

        # Construir modelo
        self.model = Model(inputs=inputs, outputs=outputs, name="DisasterTweetLSTM")

        return self.model

    def compile(
        self,
        learning_rate: float = 1e-3,
        optimizer: str = "adam",
    ) -> None:
        """
        Compila el modelo con el optimizador y métricas especificadas.

        Parameters
        ----------
        learning_rate : float, default=1e-3
            Tasa de aprendizaje para el optimizador.
        optimizer : str, default="adam"
            Nombre del optimizador ("adam", "rmsprop", "sgd").

        Raises
        ------
        RuntimeError
            Si el modelo no ha sido construido con build().

        Notes
        -----
        💡 Conexión con M03 - Cálculo:
            optimizer.apply_gradients() implementa: θ = θ - α∇L
            donde α es learning_rate y ∇L es el gradiente de la pérdida.
        """
        if self.model is None:
            raise RuntimeError("Modelo no construido. Llamar build() primero.")

        from tensorflow.keras.optimizers import SGD, Adam, RMSprop

        optimizers = {
            "adam": Adam(learning_rate=learning_rate),
            "rmsprop": RMSprop(learning_rate=learning_rate),
            "sgd": SGD(learning_rate=learning_rate, momentum=0.9),
        }

        if optimizer.lower() not in optimizers:
            raise ValueError(
                f"Optimizador '{optimizer}' no soportado. Usar: {list(optimizers.keys())}"
            )

        self.model.compile(
            optimizer=optimizers[optimizer.lower()],
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )

    def fit(
        self,
        X_train: NDArray[np.int32],
        y_train: NDArray[np.int32],
        epochs: int = 10,
        batch_size: int = 32,
        validation_data: tuple[NDArray[np.int32], NDArray[np.int32]] | None = None,
        early_stopping_patience: int = 3,
        verbose: int = 1,
    ) -> dict[str, list[float]]:
        """
        Entrena el modelo LSTM.

        Parameters
        ----------
        X_train : NDArray[np.int32]
            Secuencias de entrenamiento (n_samples, max_length).
        y_train : NDArray[np.int32]
            Labels de entrenamiento (n_samples,).
        epochs : int, default=10
            Número de épocas de entrenamiento.
        batch_size : int, default=32
            Tamaño del batch. Reducir a 16 si hay problemas de memoria.
        validation_data : tuple | None, default=None
            Tupla (X_val, y_val) para validación durante entrenamiento.
        early_stopping_patience : int, default=3
            Épocas sin mejora antes de detener entrenamiento.
        verbose : int, default=1
            Nivel de verbosidad (0=silencioso, 1=barra, 2=una línea por época).

        Returns
        -------
        dict[str, list[float]]
            Historial de entrenamiento con métricas por época.

        Raises
        ------
        RuntimeError
            Si el modelo no ha sido compilado.
        """
        if self.model is None:
            raise RuntimeError("Modelo no construido. Llamar build() primero.")

        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        callbacks = [
            EarlyStopping(
                monitor="val_loss" if validation_data else "loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss" if validation_data else "loss",
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose,
        )

        self._history = dict(history.history)
        return self._history

    def predict(
        self,
        X: NDArray[np.int32],
        batch_size: int = 32,
    ) -> NDArray[np.float32]:
        """
        Genera predicciones de probabilidad para las secuencias de entrada.

        Parameters
        ----------
        X : NDArray[np.int32]
            Secuencias tokenizadas (n_samples, max_length).
        batch_size : int, default=32
            Tamaño del batch para predicción.

        Returns
        -------
        NDArray[np.float32]
            Probabilidades de clase positiva (disaster=1) con shape (n_samples,).
        """
        if self.model is None:
            raise RuntimeError("Modelo no construido. Llamar build() primero.")

        predictions = self.model.predict(X, batch_size=batch_size, verbose=0)
        return predictions.flatten()

    def predict_classes(
        self,
        X: NDArray[np.int32],
        threshold: float = 0.5,
    ) -> NDArray[np.int32]:
        """
        Genera predicciones de clase binaria.

        Parameters
        ----------
        X : NDArray[np.int32]
            Secuencias tokenizadas.
        threshold : float, default=0.5
            Umbral de decisión. Ajustar según precision/recall deseado.

        Returns
        -------
        NDArray[np.int32]
            Clases predichas (0 o 1) con shape (n_samples,).
        """
        probas = self.predict(X)
        return (probas >= threshold).astype(np.int32)

    def summary(self) -> None:
        """Imprime el resumen de la arquitectura del modelo."""
        if self.model is None:
            print("⚠️ Modelo no construido. Llamar build() primero.")
            return
        self.model.summary()

    def save(self, filepath: str) -> None:
        """
        Guarda el modelo entrenado.

        Parameters
        ----------
        filepath : str
            Ruta para guardar el modelo (formato .keras o .h5).
        """
        if self.model is None:
            raise RuntimeError("Modelo no construido.")
        self.model.save(filepath)
        print(f"✅ Modelo guardado en: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> DisasterTweetLSTM:
        """
        Carga un modelo previamente guardado.

        Parameters
        ----------
        filepath : str
            Ruta al modelo guardado.

        Returns
        -------
        DisasterTweetLSTM
            Instancia con el modelo cargado.
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow requerido.")

        instance = cls()
        instance.model = tf.keras.models.load_model(filepath)
        print(f"✅ Modelo cargado desde: {filepath}")
        return instance


# =============================================================================
# MODELO SIMPLIFICADO PARA CPU
# =============================================================================


class SimpleLSTMCPU:
    """
    Versión simplificada del LSTM para entrenar en CPU.

    ⚠️ Usar esta clase si NO tienes GPU disponible.

    Diferencias con DisasterTweetLSTM:
    - LSTM unidireccional (no bidireccional)
    - Menos unidades LSTM (32 vs 64)
    - Embedding dimension reducida (50 vs 100)
    - Sin recurrent_dropout (más rápido en CPU)

    Examples
    --------
    >>> model = SimpleLSTMCPU(vocab_size=5000, max_length=100)
    >>> model.build()
    >>> model.compile()
    >>> model.fit(X_train, y_train, epochs=3, batch_size=16)
    """

    def __init__(
        self,
        vocab_size: int = 5000,
        embedding_dim: int = 50,
        max_length: int = 100,
        lstm_units: int = 32,
    ) -> None:
        """Inicializa modelo simplificado para CPU."""
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.lstm_units = lstm_units
        self.model: tf.keras.Model | None = None

    def build(self) -> tf.keras.Model:
        """Construye arquitectura simplificada."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow requerido.")

        from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Input
        from tensorflow.keras.models import Model

        inputs = Input(shape=(self.max_length,), dtype="int32")
        x = Embedding(self.vocab_size, self.embedding_dim)(inputs)
        x = LSTM(self.lstm_units)(x)  # Unidireccional, sin return_sequences
        x = Dropout(0.3)(x)
        x = Dense(32, activation="relu")(x)
        outputs = Dense(1, activation="sigmoid")(x)

        self.model = Model(inputs, outputs, name="SimpleLSTM_CPU")
        return self.model

    def compile(self, learning_rate: float = 1e-3) -> None:
        """Compila el modelo."""
        if self.model is None:
            raise RuntimeError("Llamar build() primero.")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

    def fit(self, X: NDArray, y: NDArray, **kwargs) -> dict[str, list[float]]:
        """Entrena el modelo."""
        if self.model is None:
            raise RuntimeError("Llamar build() y compile() primero.")
        history = self.model.fit(X, y, **kwargs)
        return dict(history.history)

    def predict(self, X: NDArray, batch_size: int = 32) -> NDArray[np.float32]:
        """Genera predicciones."""
        if self.model is None:
            raise RuntimeError("Llamar build() primero.")
        predictions = self.model.predict(X, batch_size=batch_size, verbose=0)
        return predictions.flatten()

    def summary(self) -> None:
        """Imprime el resumen del modelo."""
        if self.model is None:
            print("⚠️ Modelo no construido.")
            return
        self.model.summary()


# =============================================================================
# CELDA DE VALIDACIÓN (AUTOGRADER)
# =============================================================================


def validar_modelo_lstm(
    modelo: DisasterTweetLSTM | SimpleLSTMCPU,
    X_test: NDArray[np.int32],
    y_test: NDArray[np.int32],
    min_accuracy: float = 0.70,
    min_auc: float = 0.65,
) -> bool:
    """
    Celda de Validación para el modelo LSTM.

    Verifica que el modelo del estudiante:
    1. Está correctamente construido y compilado
    2. Puede hacer predicciones con la forma correcta
    3. Alcanza métricas mínimas de rendimiento

    Parameters
    ----------
    modelo : DisasterTweetLSTM | SimpleLSTMCPU
        Modelo LSTM entrenado.
    X_test : NDArray[np.int32]
        Datos de prueba tokenizados.
    y_test : NDArray[np.int32]
        Labels de prueba.
    min_accuracy : float, default=0.70
        Accuracy mínima requerida.
    min_auc : float, default=0.65
        AUC mínima requerida.

    Returns
    -------
    bool
        True si el modelo pasa todas las validaciones.

    Raises
    ------
    AssertionError
        Si alguna validación falla.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    print("🔍 Validando modelo LSTM...")

    # Test 1: Modelo construido
    assert (
        modelo.model is not None
    ), "❌ Error: El modelo no está construido. Llama build()."
    print("  ✅ Test 1: Modelo construido correctamente")

    # Test 2: Predicciones
    try:
        y_pred_proba = modelo.predict(X_test[:100])  # Subset pequeño
        assert y_pred_proba.shape == (
            100,
        ), f"❌ Shape incorrecto: {y_pred_proba.shape}"
        print("  ✅ Test 2: Predicciones con shape correcto")
    except Exception as e:
        raise AssertionError(f"❌ Error en predicción: {e}") from e

    # Test 3: Métricas
    y_pred_proba_full = modelo.predict(X_test)
    y_pred = (y_pred_proba_full >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba_full)

    assert (
        accuracy >= min_accuracy
    ), f"❌ Accuracy insuficiente: {accuracy:.2%} < {min_accuracy:.2%}"
    print(f"  ✅ Test 3: Accuracy = {accuracy:.2%} (mínimo: {min_accuracy:.2%})")

    assert auc >= min_auc, f"❌ AUC insuficiente: {auc:.3f} < {min_auc:.3f}"
    print(f"  ✅ Test 4: AUC = {auc:.3f} (mínimo: {min_auc:.3f})")

    print("\n✅ ¡EXCELENTE! Tu modelo LSTM pasa todas las validaciones.")
    return True


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DEMO: DisasterTweetLSTM")
    print("=" * 70)

    # Datos sintéticos para demo
    n_samples = 1000
    vocab_size = 5000
    max_length = 50

    rng = np.random.default_rng(42)
    X_demo = rng.integers(0, vocab_size, size=(n_samples, max_length))
    y_demo = rng.integers(0, 2, size=(n_samples,))

    # Crear y entrenar modelo
    print("\n📦 Creando modelo...")

    modelo: DisasterTweetLSTM | SimpleLSTMCPU
    if GPUS_AVAILABLE > 0:
        modelo = DisasterTweetLSTM(
            vocab_size=vocab_size,
            embedding_dim=50,
            max_length=max_length,
            lstm_units=32,
        )
    else:
        print("⚠️ Sin GPU - usando modelo simplificado")
        modelo = SimpleLSTMCPU(
            vocab_size=vocab_size,
            embedding_dim=50,
            max_length=max_length,
        )

    modelo.build()
    modelo.compile(learning_rate=1e-3)
    modelo.summary()

    print("\n🏋️ Entrenando modelo (2 épocas demo)...")
    history = modelo.fit(
        X_demo[:800],
        y_demo[:800],
        epochs=2,
        batch_size=32,
        validation_data=(X_demo[800:], y_demo[800:]),
        verbose=1,
    )

    print("\n✅ Demo completado!")
