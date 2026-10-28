"""Perceptron de camada única, escrito do zero.

Nada aqui vem de biblioteca de modelo: a ativação (`step`), a predição, a regra
de atualização e o laço de treino são implementados à mão, com NumPy usado
apenas como álgebra linear. É a mesma implementação usada nos dois exercícios —
o Exercise 2 a reaproveita **sem alterar uma linha**.

Convenção de rótulos: :math:`y \\in \\{0, 1\\}`. A regra de atualização é

.. math:: \\mathbf{w} \\leftarrow \\mathbf{w} + \\eta (y - \\hat{y}) \\mathbf{x},
          \\qquad b \\leftarrow b + \\eta (y - \\hat{y})

O erro :math:`(y - \\hat{y})` vale 0 quando a predição está certa — amostras
corretas não geram atualização — e :math:`\\pm 1` nos dois tipos de engano.
A forma ``w += eta * y * x``, comum em livros, pertence à convenção
:math:`y \\in \\{-1, +1\\}`: com rótulos 0/1 ela nunca atualizaria na classe 0 e
o perceptron jamais corrigiria um falso positivo.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def step(z: np.ndarray | float) -> np.ndarray | float:
    """Degrau de Heaviside: 1 se ``z >= 0``, 0 caso contrário."""
    return np.where(np.asarray(z) >= 0.0, 1, 0)


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Predição do perceptron: ``step(X @ w + b)``."""
    return step(X @ w + b)


def accuracy(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    """Fração de acertos de ``(w, b)`` sobre o conjunto inteiro."""
    return float(np.mean(predict(X, w, b) == y))


def train(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    *,
    eta: float = 0.01,
    max_epochs: int = 100,
) -> dict[str, Any]:
    """Treina o perceptron e devolve pesos finais, pesos *pocket* e histórico.

    Parâmetros seguem o enunciado: inicialização ``rng.normal(0, 0.01, size=2)``
    com ``b = 0`` (nunca do zero — o item D do Exercise 1 depende disso), taxa de
    aprendizado ``eta`` e parada quando uma época inteira não produz nenhuma
    atualização, ou ao atingir ``max_epochs``.

    O **pocket** é a única coisa acrescentada ao laço: sempre que uma atualização
    deixa a acurácia sobre o conjunto completo acima da melhor já vista, copia-se
    ``(w, b)``. Em dados separáveis ele converge para os pesos finais e não muda
    nada; em dados sobrepostos ele é o que salva o melhor hiperplano encontrado.
    Mantê-lo sempre ligado é o que permite usar esta função, sem modificação, nos
    dois exercícios.
    """
    n_features = X.shape[1]
    w = rng.normal(0.0, 0.01, size=n_features)
    b = 0.0

    pocket_w, pocket_b = w.copy(), b
    pocket_accuracy = accuracy(X, y, w, b)
    pocket_epoch = 0

    history: list[dict[str, float]] = []
    epochs_run = 0
    converged = False

    for epoch in range(1, max_epochs + 1):
        epochs_run = epoch
        # Embaralha a cada época: os dados chegam agrupados por classe (1000 da
        # classe 0, depois 1000 da classe 1) e percorrê-los nessa ordem faria o
        # perceptron ver uma classe inteira antes da outra. A permutação sai do
        # mesmo rng, então continua reprodutível.
        order = rng.permutation(len(X))
        updates = 0

        for index in order:
            x_i, y_i = X[index], y[index]
            y_hat = step(np.dot(w, x_i) + b)
            error = y_i - y_hat
            if error != 0:
                w = w + eta * error * x_i
                b = b + eta * error
                updates += 1
                # Pocket: só faz sentido medir depois de uma atualização, porque
                # sem atualização os pesos não mudaram.
                current = accuracy(X, y, w, b)
                if current > pocket_accuracy:
                    pocket_accuracy = current
                    pocket_w, pocket_b = w.copy(), b
                    pocket_epoch = epoch

        history.append(
            {
                "epoch": epoch,
                "updates": updates,
                "accuracy": accuracy(X, y, w, b),
                "pocket_accuracy": pocket_accuracy,
            }
        )

        if updates == 0:
            # Uma passagem completa sem nenhuma atualização: todo ponto está do
            # lado certo da fronteira e nada mais mudaria.
            converged = True
            break

    return {
        "w": w,
        "b": float(b),
        "accuracy": accuracy(X, y, w, b),
        "epochs": epochs_run,
        "converged": converged,
        "pocket_w": pocket_w,
        "pocket_b": float(pocket_b),
        "pocket_accuracy": pocket_accuracy,
        "pocket_epoch": pocket_epoch,
        "history": history,
        "eta": eta,
        "max_epochs": max_epochs,
    }
