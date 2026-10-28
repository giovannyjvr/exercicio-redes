"""MLP escrito do zero em NumPy — forward, perda, backprop e update.

Nada aqui usa autodiff. Cada gradiente é derivado à mão e escrito explicitamente;
o NumPy entra apenas como álgebra linear.

**O núcleo é genérico e é reutilizado literalmente** nos Exercises 2, 3 e 4:

* ``layer_sizes`` é um argumento — uma lista como ``[2, 16, 1]`` ou
  ``[4, 32, 32, 3]``. Nada é fixado no código;
* :meth:`MLP.forward`, :meth:`MLP.backward` e :meth:`MLP.update` são **o mesmo
  código** para classificação binária e multiclasse. O que muda é um argumento:
  ``output="sigmoid"`` com entropia cruzada binária, ou ``output="softmax"`` com
  entropia cruzada categórica.

A razão de isso funcionar é a identidade que o Exercise 3 pede para derivar: tanto
para sigmoid + BCE quanto para softmax + entropia cruzada categórica,

.. math:: \\frac{\\partial L}{\\partial u^{(L)}} = \\frac{1}{N}(p - y)

onde :math:`p` é a saída do modelo e :math:`y` o alvo (one-hot no caso
multiclasse). O gradiente na saída tem a mesma forma nos dois casos, então o
backward não precisa saber qual perda está sendo usada.
"""

from __future__ import annotations

from typing import Any

import numpy as np

EPS = 1e-12  # evita log(0) nas perdas


def tanh(z: np.ndarray) -> np.ndarray:
    """Ativação das camadas escondidas."""
    return np.tanh(z)


def tanh_derivative(a: np.ndarray) -> np.ndarray:
    """Derivada da tanh escrita em função da **ativação**: :math:`1 - \\tanh^2(z)`."""
    return 1.0 - a**2


def sigmoid(u: np.ndarray) -> np.ndarray:
    """Sigmoide numericamente estável, para os dois sinais de ``u``."""
    positive = u >= 0
    out = np.empty_like(u, dtype=float)
    out[positive] = 1.0 / (1.0 + np.exp(-u[positive]))
    exp_u = np.exp(u[~positive])
    out[~positive] = exp_u / (1.0 + exp_u)
    return out


def softmax(u: np.ndarray) -> np.ndarray:
    """Softmax por linha, deslocado pelo máximo para não estourar o expoente."""
    shifted = u - u.max(axis=1, keepdims=True)
    exp_u = np.exp(shifted)
    return exp_u / exp_u.sum(axis=1, keepdims=True)


class MLP:
    """Perceptron multicamada com número arbitrário de camadas escondidas.

    Parameters
    ----------
    layer_sizes:
        Larguras, da entrada à saída. ``[2, 16, 1]`` é uma camada escondida de 16
        neurônios; ``[4, 32, 32, 3]`` são duas de 32 com 3 saídas.
    rng:
        Gerador usado na inicialização dos pesos.
    output:
        ``"sigmoid"`` (binário, com BCE) ou ``"softmax"`` (multiclasse, com
        entropia cruzada categórica).
    """

    def __init__(
        self,
        layer_sizes: list[int],
        rng: np.random.Generator,
        *,
        output: str = "sigmoid",
    ) -> None:
        if output not in {"sigmoid", "softmax"}:
            raise ValueError(f"saída desconhecida: {output!r}")
        self.layer_sizes = list(layer_sizes)
        self.output = output

        # Inicialização de Xavier/Glorot: escala 1/sqrt(fan_in) mantém a variância
        # das pré-ativações estável ao longo da profundidade. Com escala fixa, uma
        # rede mais funda saturaria a tanh já no forward inicial.
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            self.weights.append(rng.normal(0.0, 1.0, size=(fan_in, fan_out)) / np.sqrt(fan_in))
            self.biases.append(np.zeros(fan_out))

    # ------------------------------------------------------------------ core
    # As três funções abaixo são o núcleo reutilizado sem alteração nos
    # Exercises 2, 3 e 4.

    def forward(self, X: np.ndarray) -> dict[str, Any]:
        """Passagem direta. Devolve as ativações de cada camada e a saída."""
        activations = [X]
        a = X
        # Camadas escondidas: pré-ativação afim seguida de tanh.
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            a = tanh(a @ W + b)
            activations.append(a)
        # Camada de saída: a pré-ativação é a mesma conta; só a não linearidade muda.
        u = a @ self.weights[-1] + self.biases[-1]
        probs = sigmoid(u) if self.output == "sigmoid" else softmax(u)
        return {"activations": activations, "u": u, "probs": probs}

    def backward(self, cache: dict[str, Any], Y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Retropropagação. Devolve ``(dW, db)``, um por camada.

        O gradiente na saída é ``(p - Y) / N`` nos dois casos — ver o docstring do
        módulo e a derivação no relatório. Daqui para trás o código é idêntico
        para binário e multiclasse.
        """
        activations, probs = cache["activations"], cache["probs"]
        n_samples = Y.shape[0]

        # Gradiente na pré-ativação da saída.
        delta = (probs - Y) / n_samples

        grads_w: list[np.ndarray] = [np.zeros_like(W) for W in self.weights]
        grads_b: list[np.ndarray] = [np.zeros_like(b) for b in self.biases]

        for layer in range(len(self.weights) - 1, -1, -1):
            a_prev = activations[layer]
            grads_w[layer] = a_prev.T @ delta
            grads_b[layer] = delta.sum(axis=0)
            if layer > 0:
                # Propaga para a camada anterior e atravessa a tanh.
                delta = (delta @ self.weights[layer].T) * tanh_derivative(activations[layer])

        return grads_w, grads_b

    def update(self, grads_w: list[np.ndarray], grads_b: list[np.ndarray], learning_rate: float) -> None:
        """Descida de gradiente simples, aplicada a todos os parâmetros."""
        for layer in range(len(self.weights)):
            self.weights[layer] -= learning_rate * grads_w[layer]
            self.biases[layer] -= learning_rate * grads_b[layer]

    # ------------------------------------------------------------------ perdas

    def loss(self, probs: np.ndarray, Y: np.ndarray) -> float:
        """Entropia cruzada binária ou categórica, média por amostra."""
        if self.output == "sigmoid":
            p = np.clip(probs, EPS, 1.0 - EPS)
            return float(-np.mean(Y * np.log(p) + (1.0 - Y) * np.log(1.0 - p)))
        p = np.clip(probs, EPS, 1.0)
        return float(-np.mean(np.sum(Y * np.log(p), axis=1)))

    # ------------------------------------------------------------------ uso

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        epochs: int,
        learning_rate: float,
    ) -> list[float]:
        """Treina em *full batch* e devolve a perda de treino de cada época."""
        history: list[float] = []
        for _ in range(epochs):
            cache = self.forward(X)
            history.append(self.loss(cache["probs"], Y))
            grads_w, grads_b = self.backward(cache, Y)
            self.update(grads_w, grads_b, learning_rate)
        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)["probs"]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Rótulos preditos: limiar de 0.5 no binário, argmax no multiclasse."""
        probs = self.predict_proba(X)
        if self.output == "sigmoid":
            return (probs[:, 0] >= 0.5).astype(int)
        return probs.argmax(axis=1)

    def n_parameters(self) -> int:
        """Número total de pesos e vieses treináveis."""
        return int(sum(W.size for W in self.weights) + sum(b.size for b in self.biases))

    def architecture(self) -> str:
        """Descrição curta da arquitetura, para o relatório."""
        return "–".join(str(size) for size in self.layer_sizes)


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Codifica rótulos inteiros como linhas one-hot."""
    encoded = np.zeros((y.shape[0], n_classes))
    encoded[np.arange(y.shape[0]), y] = 1.0
    return encoded
