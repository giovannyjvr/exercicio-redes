"""Exercise 1 — um passo de retropropagação, na mão.

Este módulo **não** usa autodiff nem biblioteca de rede neural: cada linha
corresponde a um passo da derivação feita no relatório, escrita explicitamente.
Ele existe para que os números citados no texto venham de uma execução real e
possam ser conferidos — a derivação em si está no `index.md`.

Rede: 2 entradas → 1 camada escondida de 2 neurônios → 1 saída, ``tanh`` nas duas
camadas, perda :math:`L = (y - \\hat{y})^2` sobre uma única amostra.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Valores dados pelo enunciado.
X = np.array([0.5, -0.2])
Y = 1.0
W1 = np.array([[0.3, -0.1], [0.2, 0.4]])
B1 = np.array([0.1, -0.2])
W2 = np.array([0.5, -0.3])
B2 = 0.2
ETA = 0.3


def run() -> dict[str, Any]:
    """Executa forward, backward e update, devolvendo todos os intermediários."""
    # --- A — Forward pass -------------------------------------------------
    z1 = W1 @ X + B1                     # pré-ativações da camada escondida
    h1 = np.tanh(z1)                     # ativações da camada escondida
    u2 = float(W2 @ h1 + B2)             # pré-ativação da saída
    y_hat = float(np.tanh(u2))           # saída
    loss = float((Y - y_hat) ** 2)       # L = (y - yhat)^2, com N = 1

    # --- B — Backward pass ------------------------------------------------
    # dL/dyhat = -2 (y - yhat)
    dL_dyhat = -2.0 * (Y - y_hat)
    # dyhat/du2 = 1 - tanh^2(u2) = 1 - yhat^2
    dL_du2 = dL_dyhat * (1.0 - y_hat**2)
    # Camada de saída: u2 = W2 · h1 + b2  =>  du2/dW2 = h1, du2/db2 = 1
    dL_dW2 = dL_du2 * h1
    dL_db2 = dL_du2
    # Propaga para trás: du2/dh1 = W2
    dL_dh1 = dL_du2 * W2
    # Atravessa a tanh da camada escondida: dh1/dz1 = 1 - h1^2
    dL_dz1 = dL_dh1 * (1.0 - h1**2)
    # Camada escondida: z1 = W1 x + b1  =>  dz1_i/dW1_ij = x_j, dz1_i/db1_i = 1
    dL_dW1 = np.outer(dL_dz1, X)
    dL_db1 = dL_dz1

    # --- C — Parameter update --------------------------------------------
    W2_new = W2 - ETA * dL_dW2
    b2_new = B2 - ETA * dL_db2
    W1_new = W1 - ETA * dL_dW1
    b1_new = B1 - ETA * dL_db1

    # Segundo forward, com os parâmetros atualizados, para confirmar a direção
    # da perda prevista na resposta do item C.
    z1_new = W1_new @ X + b1_new
    h1_new = np.tanh(z1_new)
    u2_new = float(W2_new @ h1_new + b2_new)
    y_hat_new = float(np.tanh(u2_new))
    loss_new = float((Y - y_hat_new) ** 2)

    return {
        "forward": {
            "z1": z1.tolist(), "h1": h1.tolist(),
            "u2": u2, "y_hat": y_hat, "loss": loss,
        },
        "backward": {
            "dL_dyhat": dL_dyhat, "dL_du2": dL_du2,
            "dL_dW2": dL_dW2.tolist(), "dL_db2": dL_db2,
            "dL_dh1": dL_dh1.tolist(), "dL_dz1": dL_dz1.tolist(),
            "dL_dW1": dL_dW1.tolist(), "dL_db1": dL_db1.tolist(),
        },
        "updated": {
            "W1": W1_new.tolist(), "b1": b1_new.tolist(),
            "W2": W2_new.tolist(), "b2": b2_new,
        },
        "second_forward": {
            "z1": z1_new.tolist(), "h1": h1_new.tolist(),
            "u2": u2_new, "y_hat": y_hat_new, "loss": loss_new,
            "loss_decreased": bool(loss_new < loss),
            "delta": loss_new - loss,
        },
    }


def report(result: dict[str, Any]) -> None:
    """Imprime todos os intermediários com 6 casas decimais."""
    f, b, u, s = result["forward"], result["backward"], result["updated"], result["second_forward"]
    print("A — Forward pass")
    print(f"  z1          = [{f['z1'][0]:.6f}, {f['z1'][1]:.6f}]")
    print(f"  h1          = [{f['h1'][0]:.6f}, {f['h1'][1]:.6f}]")
    print(f"  u2          = {f['u2']:.6f}")
    print(f"  y_hat       = {f['y_hat']:.6f}")
    print(f"  L           = {f['loss']:.6f}")
    print("\nB — Backward pass")
    print(f"  dL/dy_hat   = {b['dL_dyhat']:.6f}")
    print(f"  dL/du2      = {b['dL_du2']:.6f}")
    print(f"  dL/dW2      = [{b['dL_dW2'][0]:.6f}, {b['dL_dW2'][1]:.6f}]")
    print(f"  dL/db2      = {b['dL_db2']:.6f}")
    print(f"  dL/dh1      = [{b['dL_dh1'][0]:.6f}, {b['dL_dh1'][1]:.6f}]")
    print(f"  dL/dz1      = [{b['dL_dz1'][0]:.6f}, {b['dL_dz1'][1]:.6f}]")
    print(f"  dL/dW1      = [[{b['dL_dW1'][0][0]:.6f}, {b['dL_dW1'][0][1]:.6f}],"
          f" [{b['dL_dW1'][1][0]:.6f}, {b['dL_dW1'][1][1]:.6f}]]")
    print(f"  dL/db1      = [{b['dL_db1'][0]:.6f}, {b['dL_db1'][1]:.6f}]")
    print("\nC — Parameter update (eta = 0.3)")
    print(f"  W1          = [[{u['W1'][0][0]:.6f}, {u['W1'][0][1]:.6f}],"
          f" [{u['W1'][1][0]:.6f}, {u['W1'][1][1]:.6f}]]")
    print(f"  b1          = [{u['b1'][0]:.6f}, {u['b1'][1]:.6f}]")
    print(f"  W2          = [{u['W2'][0]:.6f}, {u['W2'][1]:.6f}]")
    print(f"  b2          = {u['b2']:.6f}")
    print(f"\n  segundo forward: y_hat = {s['y_hat']:.6f}, L = {s['loss']:.6f}"
          f" (variação {s['delta']:+.6f}; caiu: {s['loss_decreased']})")
