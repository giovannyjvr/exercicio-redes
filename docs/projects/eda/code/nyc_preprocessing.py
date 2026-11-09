"""Pipeline de pré-processamento do *New York City Airbnb Open Data*.

Módulo **importável**: carrega a base, separa treino/teste e devolve um
``Pipeline`` do scikit-learn pronto para a etapa de regressão.

Uso típico::

    from nyc_preprocessing import load_dataset, split_data, build_preprocessor, verify

    df = load_dataset()
    X_train, X_test, y_train, y_test = split_data(df)

    prep = build_preprocessor(X_train)
    Z_train = prep.fit_transform(X_train)   # aprende E transforma — só o treino
    Z_test = prep.transform(X_test)         # só transforma

    print(verify(prep, Z_train, Z_test))

Regras que o módulo garante:

* ``price`` **nunca** entra em ``X`` — :func:`split_data` o remove antes de devolver;
* todo parâmetro aprendido (mediana, quantil de corte, média/desvio da escala,
  categorias do one-hot) sai **apenas** do treino;
* categorias inéditas no teste não quebram o ``transform``;
* a saída não contém ``NaN``, e treino e teste têm as mesmas colunas, na mesma ordem.

A semente é :data:`RANDOM_STATE` = 42, como pede a rubrica.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.20

KAGGLE_DATASET = "dgomonov/new-york-city-airbnb-open-data"
CSV_NAME = "AB_NYC_2019.csv"

#: Esquema esperado do arquivo bruto. ``load_dataset`` recusa um CSV diferente —
#: é a verificação que impede a análise de rodar sobre a base errada.
EXPECTED_COLUMNS = [
    "id", "name", "host_id", "host_name", "neighbourhood_group", "neighbourhood",
    "latitude", "longitude", "room_type", "price", "minimum_nights",
    "number_of_reviews", "last_review", "reviews_per_month",
    "calculated_host_listings_count", "availability_365",
]
EXPECTED_SHAPE = (48895, 16)

TARGET = "price"

#: Descartadas antes da modelagem. O relatório justifica uma a uma; o motivo
#: curto fica aqui para quem lê só o código.
DROPPED_COLUMNS = {
    "id": "identificador do anúncio — único por linha, sem conteúdo preditivo",
    "host_id": "identificador do anfitrião — 37 457 valores; como feature vira memorização",
    "name": "texto livre do anúncio — 47 905 valores distintos; exige NLP, fora do escopo tabular",
    "host_name": "nome próprio do anfitrião — identificador pessoal, sem relação causal com preço",
}

#: Numéricas de cauda pesada: ganham winsorização + log1p antes da escala.
NUM_HEAVY = [
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
]
#: Numéricas de escala bem-comportada: só imputação e escala.
NUM_PLAIN = ["latitude", "longitude", "availability_365", "days_since_last_review"]
#: Indicador construído por :class:`ReviewRecencyAdder`; já é 0/1, não precisa de escala.
FLAG_COLUMNS = ["never_reviewed"]
#: Categóricas de baixa cardinalidade: one-hot direto.
CAT_LOW = ["neighbourhood_group", "room_type"]
#: Categórica de alta cardinalidade (221 bairros): one-hot com agrupamento de raras.
CAT_HIGH = ["neighbourhood"]

#: Bairros com menos ocorrências que isto no treino caem num nível "infrequent".
#: 50 linhas ≈ 0,13 % do treino — o corte está justificado na seção 2B do relatório.
NEIGHBOURHOOD_MIN_FREQUENCY = 50

#: Quantis de corte da winsorização, aprendidos no treino.
WINSOR_LOWER, WINSOR_UPPER = 0.01, 0.99


# --------------------------------------------------------------------------- dados


def find_csv() -> pathlib.Path:
    """Localiza ``AB_NYC_2019.csv``, baixando pelo KaggleHub se preciso.

    Ordem de busca:

    1. a variável de ambiente ``NYC_AIRBNB_CSV``, se apontar para um arquivo;
    2. o cache/download do KaggleHub.

    O CSV **não** é versionado no repositório: tem 6,8 MB e o KaggleHub o
    reconstrói em qualquer clone limpo.
    """
    override = os.environ.get("NYC_AIRBNB_CSV")
    if override:
        path = pathlib.Path(override)
        if path.is_file():
            return path
        raise FileNotFoundError(f"NYC_AIRBNB_CSV aponta para um arquivo inexistente: {path}")

    import kagglehub  # importado aqui para o módulo carregar sem a dependência

    root = pathlib.Path(kagglehub.dataset_download(KAGGLE_DATASET))
    # O arquivo não fica necessariamente na raiz do diretório devolvido.
    matches = sorted(root.rglob("*.csv"))
    if not matches:
        raise FileNotFoundError(f"nenhum CSV em {root}")
    for candidate in matches:
        if candidate.name == CSV_NAME:
            return candidate
    return matches[0]


def load_dataset(*, validate: bool = True) -> pd.DataFrame:
    """Lê o CSV bruto e confere nome, colunas, dimensão e tipos.

    Com ``validate=True`` (padrão), levanta ``ValueError`` se o esquema não for
    o esperado — assim um dataset trocado falha alto, em vez de produzir uma
    análise silenciosamente errada.
    """
    csv = find_csv()
    df = pd.read_csv(csv)

    if validate:
        if list(df.columns) != EXPECTED_COLUMNS:
            raise ValueError(
                f"colunas inesperadas em {csv.name}:\n"
                f"  esperado: {EXPECTED_COLUMNS}\n"
                f"  obtido:   {list(df.columns)}"
            )
        if df.shape != EXPECTED_SHAPE:
            raise ValueError(f"dimensão inesperada: {df.shape}, esperado {EXPECTED_SHAPE}")
        if not pd.api.types.is_numeric_dtype(df[TARGET]):
            raise ValueError(f"{TARGET} deveria ser numérico, veio {df[TARGET].dtype}")

    df.attrs["source_file"] = str(csv)
    return df


def drop_invalid_target(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove as linhas com ``price <= 0``.

    Diária de zero dólar não é uma observação da grandeza que se quer prever —
    é anúncio inativo, erro de cadastro ou preço combinado fora da plataforma.
    Mantê-las puxaria a métrica de erro para baixo sem que o modelo tivesse
    aprendido nada. São 11 linhas (0,022 % da base); o relatório documenta o efeito.
    """
    invalid = df[TARGET] <= 0
    report = {
        "rows_before": int(len(df)),
        "rows_dropped": int(invalid.sum()),
        "share_dropped": float(invalid.mean()),
        "rows_after": int(len(df) - invalid.sum()),
    }
    return df.loc[~invalid].reset_index(drop=True), report


def split_data(
    df: pd.DataFrame, *, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa treino/teste **antes** de qualquer ajuste de pré-processamento.

    Devolve ``(X_train, X_test, y_train, y_test)`` com ``price`` fora de ``X`` e
    com as colunas puramente identificadoras já descartadas.
    """
    features = df.drop(columns=[TARGET, *DROPPED_COLUMNS])
    target = df[TARGET]
    return train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )


# --------------------------------------------------------------- transformadores


class ReviewRecencyAdder(BaseEstimator, TransformerMixin):
    """Converte ``last_review`` em dias de recência e marca quem nunca foi avaliado.

    ``last_review`` é uma data em texto: inútil para a rede como está, e ausente
    em exatamente as 10 052 linhas com ``number_of_reviews == 0``. A ausência é
    **estrutural**, não aleatória — não existe "última avaliação" de quem não tem
    avaliação nenhuma.

    Duas colunas saem daqui:

    * ``days_since_last_review`` — dias entre a última avaliação e a data de
      referência, que é a avaliação mais recente **vista no treino**
      (``reference_date_``). Aprender essa data no treino é o que impede o
      vazamento: o teste é medido contra o relógio do treino.
    * ``never_reviewed`` — 1 para quem nunca foi avaliado. O indicador preserva a
      informação que a imputação de ``days_since_last_review`` apagaria.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ReviewRecencyAdder":
        dates = pd.to_datetime(X["last_review"], errors="coerce")
        self.reference_date_ = dates.max()
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        dates = pd.to_datetime(out["last_review"], errors="coerce")
        out["days_since_last_review"] = (self.reference_date_ - dates).dt.days
        out["never_reviewed"] = out["last_review"].isna().astype(int)
        return out.drop(columns=["last_review"])


class Winsorizer(BaseEstimator, TransformerMixin):
    """Corta cada coluna nos quantis aprendidos **no treino**.

    Winsorizar em vez de remover: a linha continua no conjunto, só o valor
    extremo é puxado para o limite. Remover linhas de treino por causa de uma
    feature extrema descartaria o preço que as acompanha, e remover linhas do
    teste seria escolher em quais exemplos o modelo é avaliado.
    """

    def __init__(self, lower: float = WINSOR_LOWER, upper: float = WINSOR_UPPER) -> None:
        self.lower = lower
        self.upper = upper

    def fit(self, X: Any, y: Any = None) -> "Winsorizer":
        values = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(values, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(values, self.upper, axis=0)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X: Any) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(input_features, dtype=object)

    def count_clipped(self, X: Any) -> dict[str, int]:
        """Quantos valores cada coluna teria cortado — usado no relatório."""
        values = np.asarray(X, dtype=float)
        below = (values < self.lower_bounds_).sum(axis=0)
        above = (values > self.upper_bounds_).sum(axis=0)
        return {"below": below.astype(int), "above": above.astype(int)}


def _log1p_transformer() -> FunctionTransformer:
    """``log1p`` como passo nomeado, preservando os nomes das features."""
    return FunctionTransformer(np.log1p, feature_names_out="one-to-one", validate=False)


# ------------------------------------------------------------------- a pipeline


def build_preprocessor(
    X_train: pd.DataFrame | None = None,
    *,
    neighbourhood_min_frequency: int = NEIGHBOURHOOD_MIN_FREQUENCY,
) -> Pipeline:
    """Monta a pipeline completa de pré-processamento.

    ``X_train`` é aceito só para checagem defensiva das colunas esperadas; nada
    é aprendido aqui — o aprendizado acontece em ``fit``/``fit_transform``.

    Estrutura::

        ReviewRecencyAdder            (last_review -> days_since + never_reviewed)
          └── ColumnTransformer
                ├── heavy : imputa 0 -> winsoriza -> log1p -> padroniza
                ├── plain : imputa mediana -> padroniza
                ├── flag  : passa direto (já é 0/1)
                ├── cat   : imputa moda -> one-hot (handle_unknown="ignore")
                └── hicard: imputa moda -> one-hot com agrupamento de raras
    """
    if X_train is not None:
        required = set(NUM_HEAVY) | set(NUM_PLAIN) | set(CAT_LOW) | set(CAT_HIGH)
        required = (required - {"days_since_last_review"}) | {"last_review"}
        missing = required - set(X_train.columns)
        if missing:
            raise ValueError(f"faltam colunas em X_train: {sorted(missing)}")

    # Ausente em reviews_per_month significa "nenhuma avaliação por mês" = 0.
    # Mediana aqui seria errado: inventaria atividade onde não houve nenhuma.
    heavy = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("win", Winsorizer()),
        ("log", _log1p_transformer()),
        ("esc", StandardScaler()),
    ])

    # days_since_last_review é NaN para quem nunca foi avaliado; a mediana entra
    # como valor neutro e o indicador never_reviewed diz ao modelo que o valor
    # foi imputado. Latitude/longitude/availability não têm ausentes.
    plain = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("esc", StandardScaler()),
    ])

    categorical = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("cod", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # min_frequency agrupa bairros raros num nível "infrequent_sklearn";
    # handle_unknown="infrequent_if_exist" manda um bairro inédito no teste para
    # esse mesmo nível, em vez de uma linha de zeros.
    high_cardinality = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("cod", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=neighbourhood_min_frequency,
            sparse_output=False,
        )),
    ])

    column_transformer = ColumnTransformer(
        [
            ("heavy", heavy, NUM_HEAVY),
            ("plain", plain, NUM_PLAIN),
            ("flag", "passthrough", FLAG_COLUMNS),
            ("cat", categorical, CAT_LOW),
            ("hicard", high_cardinality, CAT_HIGH),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return Pipeline([("features", ReviewRecencyAdder()), ("prep", column_transformer)])


def feature_names(prep: Pipeline) -> np.ndarray:
    """Nomes das colunas produzidas, na ordem da matriz de saída."""
    return prep.named_steps["prep"].get_feature_names_out()


# ------------------------------------------------------------------ verificação


def verify(prep: Pipeline, Z_train: np.ndarray, Z_test: np.ndarray) -> dict[str, Any]:
    """Checagens automáticas da saída da pipeline.

    Levanta ``AssertionError`` no primeiro problema; devolve um dicionário com os
    números quando tudo passa, para o relatório citar sem recalcular.
    """
    names = feature_names(prep)

    assert Z_train.ndim == Z_test.ndim == 2, "a saída deveria ser uma matriz 2D"
    assert Z_train.shape[1] == Z_test.shape[1], (
        f"treino e teste com larguras diferentes: {Z_train.shape[1]} vs {Z_test.shape[1]}"
    )
    assert len(names) == Z_train.shape[1], (
        f"{len(names)} nomes para {Z_train.shape[1]} colunas"
    )
    assert not np.isnan(Z_train).any(), "há NaN na matriz de treino"
    assert not np.isnan(Z_test).any(), "há NaN na matriz de teste"
    assert np.isfinite(Z_train).all(), "há valores infinitos no treino"
    assert np.isfinite(Z_test).all(), "há valores infinitos no teste"

    # As colunas padronizadas têm média ~0 no treino por construção. No teste,
    # NÃO: uma média exatamente zero no teste seria sinal de vazamento.
    n_scaled = len(NUM_HEAVY) + len(NUM_PLAIN)
    train_mean_abs = float(np.abs(Z_train[:, :n_scaled].mean(axis=0)).max())
    test_mean_abs = float(np.abs(Z_test[:, :n_scaled].mean(axis=0)).max())
    assert train_mean_abs < 1e-10, f"escala não centrou o treino: {train_mean_abs}"

    return {
        "shape_train": list(Z_train.shape),
        "shape_test": list(Z_test.shape),
        "n_features_out": int(Z_train.shape[1]),
        "feature_names": [str(n) for n in names],
        "nan_train": int(np.isnan(Z_train).sum()),
        "nan_test": int(np.isnan(Z_test).sum()),
        "max_abs_mean_train_scaled": train_mean_abs,
        "max_abs_mean_test_scaled": test_mean_abs,
        "columns_match": True,
    }


__all__ = [
    "RANDOM_STATE", "TEST_SIZE", "TARGET", "DROPPED_COLUMNS",
    "NUM_HEAVY", "NUM_PLAIN", "FLAG_COLUMNS", "CAT_LOW", "CAT_HIGH",
    "ReviewRecencyAdder", "Winsorizer",
    "find_csv", "load_dataset", "drop_invalid_target", "split_data",
    "build_preprocessor", "feature_names", "verify",
]
