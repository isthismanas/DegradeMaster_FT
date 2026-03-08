import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mae = np.mean(np.abs(y_pred - y_true), axis=0)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))

    r2_dc50 = r2_score(y_true[:, 0], y_pred[:, 0]) if np.unique(y_true[:, 0]).size > 1 else float('nan')
    r2_dmax = r2_score(y_true[:, 1], y_pred[:, 1]) if np.unique(y_true[:, 1]).size > 1 else float('nan')

    return {
        'mae_dc50_nm': float(mae[0]),
        'mae_dmax_pct': float(mae[1]),
        'rmse_dc50_nm': float(rmse[0]),
        'rmse_dmax_pct': float(rmse[1]),
        'r2_dc50_nm': float(r2_dc50),
        'r2_dmax_pct': float(r2_dmax),
        'mean_rmse': float(np.mean(rmse)),
    }


class TabularRegressionBaseline:
    """
    Simple industry baseline on precomputed feature vectors.

    backend:
      - 'hist_gb': HistGradientBoostingRegressor per target
      - 'rf': RandomForestRegressor (multi-output native)
    """

    def __init__(self, backend='hist_gb', random_state=111, **kwargs):
        self.backend = backend
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = self._build_model()

    def _build_model(self):
        if self.backend == 'hist_gb':
            base = HistGradientBoostingRegressor(
                learning_rate=self.kwargs.get('learning_rate', 0.05),
                max_depth=self.kwargs.get('max_depth', 6),
                max_iter=self.kwargs.get('max_iter', 500),
                random_state=self.random_state,
            )
            return MultiOutputRegressor(base)

        if self.backend == 'rf':
            return RandomForestRegressor(
                n_estimators=self.kwargs.get('n_estimators', 500),
                max_depth=self.kwargs.get('max_depth', None),
                min_samples_split=self.kwargs.get('min_samples_split', 2),
                n_jobs=self.kwargs.get('n_jobs', -1),
                random_state=self.random_state,
            )

        raise ValueError(f'Unsupported backend: {self.backend}')

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return self

    def predict(self, x):
        return self.model.predict(x)

    def score(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return regression_metrics(y_test, y_pred)
