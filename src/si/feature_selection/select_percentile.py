import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset

class SelectPercentile(Transformer):

    def __init__(self, score_func, percentile):
        """
        Inicializa o seletor de percentil.

        Parâmetros
        ----------
        score_func : função,
            Função usada para calcular os valores F.
        percentile : int,
            Percentual de características a serem selecionadas
                """

        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None


    def _fit(self, X, y):

        """
        Estima os valores F e p para cada característica.

        Retorna
        -------
        self
            retorna a instância do objeto
        """

        self.F, self.p = self.score_func(X, y)
        return self

    def _transform(self, X):

        """
        Seleciona as características com os maiores valores F até o percentil especificado.

        Retorna
        -------
        X_transformed
            conjunto de dados transformado com as melhores características
        """

        num_features_to_select = int(np.ceil(self.percentile / 100 * X.shape[1]))
        top_features = np.argsort(self.F)[-num_features_to_select:]

        return X[:, top_features]