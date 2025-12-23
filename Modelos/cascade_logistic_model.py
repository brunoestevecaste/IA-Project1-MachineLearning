from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
import numpy as np

class CascadedLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    First model: 0 vs (1,2,3,4)
    Second model: 1 vs 2 vs 3 vs 4 (trained only on y > 0)
    """
    def __init__(self, C_zero=1.0, C_multi=1.0,
                 solver='lbfgs', max_iter=1000, random_state=42):
        self.C_zero = C_zero
        self.C_multi = C_multi
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)

        # -------- Model 1: 0 vs (1,2,3,4) --------
        y_binary = (y > 0).astype(int)
        self.zero_vs_rest_ = LogisticRegression(
            C=self.C_zero,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.zero_vs_rest_.fit(X, y_binary)

        # -------- Model 2: 1 vs 2 vs 3 vs 4 --------
        mask_pos = y > 0
        X_pos = X[mask_pos]
        y_pos = y[mask_pos]

        self.multi_pos_ = LogisticRegression(
            C=self.C_multi,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.multi_pos_.fit(X_pos, y_pos)

        return self

    def predict(self, X):
        # First decide 0 vs >0
        y_bin_pred = self.zero_vs_rest_.predict(X)

        # Initialize all as 0
        y_pred = np.zeros(len(X), dtype=int)

        # For those predicted >0, use the multiclass model (1–4)
        idx_pos = np.where(y_bin_pred == 1)[0]
        if len(idx_pos) > 0:
            y_pred[idx_pos] = self.multi_pos_.predict(X[idx_pos])

        return y_pred
    
    def get_intermediate_predictions(self, X):
        """
        Returns predictions from both intermediate models for confusion matrix analysis.
        
        Returns:
        --------
        dict with keys:
            'binary_pred': predictions from binary model (0 vs >0)
            'multiclass_pred': predictions from multiclass model (1 vs 2 vs 3 vs 4) for positive samples
            'final_pred': final cascaded predictions
        """
        # Binary predictions
        y_bin_pred = self.zero_vs_rest_.predict(X)
        
        # Initialize final predictions
        y_pred = np.zeros(len(X), dtype=int)
        y_multi_pred = np.full(len(X), -1, dtype=int)  # -1 for samples not passed to multiclass model
        
        # For those predicted >0, get multiclass predictions
        idx_pos = np.where(y_bin_pred == 1)[0]
        if len(idx_pos) > 0:
            multi_preds = self.multi_pos_.predict(X[idx_pos])
            y_pred[idx_pos] = multi_preds
            y_multi_pred[idx_pos] = multi_preds
        
        return {
            'binary_pred': y_bin_pred,
            'multiclass_pred': y_multi_pred,
            'final_pred': y_pred
        }
    

class ThresholdedCascadedLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Two-step model:

    - Model 1 (binary): 0 vs (1,2,3,4)  -> used to get a strong signal for class 0.
    - Model 2 (multiclass): 0,1,2,3,4   -> base classifier for all classes.

    At prediction time:
    - Use multiclass model to get base prediction (argmax of P(0..4)).
    - Use binary model's P(y=0) and a threshold to force more zeros when we are confident.
    """

    def __init__(
        self,
        C_zero=1.0,
        C_multi=1.0,
        solver='lbfgs',
        max_iter=1000,
        random_state=42,
        zero_threshold=0.6
    ):
        """
        zero_threshold: if P_binary(y=0) >= zero_threshold, force prediction = 0,
                        even if multiclass argmax says otherwise.
        """
        self.C_zero = C_zero
        self.C_multi = C_multi
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        self.zero_threshold = zero_threshold

    def fit(self, X, y):
        y = np.asarray(y)

        # --------- Model 1: Binary 0 vs rest ---------
        # y_binary = 0 if y==0, 1 if y>0
        y_binary = (y > 0).astype(int)

        self.zero_vs_rest_ = LogisticRegression(
            C=self.C_zero,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.zero_vs_rest_.fit(X, y_binary)

        # --------- Model 2: Multiclass 0..4 ---------
        self.multi_all_ = LogisticRegression(
            C=self.C_multi,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.multi_all_.fit(X, y)

        return self

    def predict(self, X):
        # Probabilities from binary model: columns correspond to classes [0,1]
        proba_bin = self.zero_vs_rest_.predict_proba(X)
        # P(y=0) is column whose class_ == 0 (it should be column 0, but let's be explicit)
        idx_zero_bin = np.where(self.zero_vs_rest_.classes_ == 0)[0][0]
        p_zero_bin = proba_bin[:, idx_zero_bin]

        # Base multiclass prediction
        proba_multi = self.multi_all_.predict_proba(X)
        classes_multi = self.multi_all_.classes_
        base_pred_idx = np.argmax(proba_multi, axis=1)
        y_pred = classes_multi[base_pred_idx]

        # Force zeros where binary model is confident enough
        mask_force_zero = p_zero_bin >= self.zero_threshold
        y_pred[mask_force_zero] = 0

        return y_pred

    def predict_proba(self, X):
        """
        Return multiclass probabilities (0..4) from the multiclass model.
        We don't modify probabilities with the threshold here; thresholding
        is only applied in predict().
        """
        return self.multi_all_.predict_proba(X)
    
    def get_intermediate_predictions(self, X):
        """
        Returns predictions from both intermediate models for confusion matrix analysis.
        
        Returns:
        --------
        dict with keys:
            'binary_pred': predictions from binary model (0 vs >0)
            'binary_proba_zero': P(y=0) from binary model
            'multiclass_pred': base predictions from multiclass model (before threshold)
            'final_pred': final predictions (after applying zero threshold)
        """
        # Binary predictions and probabilities
        proba_bin = self.zero_vs_rest_.predict_proba(X)
        idx_zero_bin = np.where(self.zero_vs_rest_.classes_ == 0)[0][0]
        p_zero_bin = proba_bin[:, idx_zero_bin]
        y_bin_pred = self.zero_vs_rest_.predict(X)
        
        # Multiclass predictions (base, before threshold)
        proba_multi = self.multi_all_.predict_proba(X)
        classes_multi = self.multi_all_.classes_
        base_pred_idx = np.argmax(proba_multi, axis=1)
        y_multi_pred = classes_multi[base_pred_idx]
        
        # Final predictions (after threshold)
        y_pred = y_multi_pred.copy()
        mask_force_zero = p_zero_bin >= self.zero_threshold
        y_pred[mask_force_zero] = 0
        
        return {
            'binary_pred': y_bin_pred,
            'binary_proba_zero': p_zero_bin,
            'multiclass_pred': y_multi_pred,
            'final_pred': y_pred
        }
    


from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
import numpy as np

class ZeroVsRestLogisticRegression(BaseEstimator, ClassifierMixin):

    """
    Binary model: 0 vs (1,2,3,4)

    - Internamente convierte y a binaria:
        y_bin = 0 si y == 0
        y_bin = 1 si y > 0

    - zero_threshold: umbral sobre P(y=0).
      Si P(y=0) >= zero_threshold  -> predicción = 0
      Si P(y=0) <  zero_threshold  -> predicción = 1
    """

    def __init__(
        self,
        C_zero=0.1,
        solver='lbfgs',
        max_iter=1000,
        random_state=42,
        zero_threshold=0.5
    ):
        self.C_zero = C_zero
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        self.zero_threshold = zero_threshold

    def fit(self, X, y):
        y = np.asarray(y)
        # 0 vs >0
        y_binary = (y > 0).astype(int)

        self.zero_vs_rest_ = LogisticRegression(
            C=self.C_zero,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.zero_vs_rest_.fit(X, y_binary)

        # Guardamos las clases binarias explícitamente (0,1)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        """
        Devuelve P(y=0) y P(y>0).
        """
        proba_bin = self.zero_vs_rest_.predict_proba(X)
        # Reordenamos columnas para que correspondan a clases_ = [0,1]
        idx_zero = np.where(self.zero_vs_rest_.classes_ == 0)[0][0]
        idx_one  = np.where(self.zero_vs_rest_.classes_ == 1)[0][0]
        proba_ordered = proba_bin[:, [idx_zero, idx_one]]
        return proba_ordered

    def predict(self, X):
        """
        Aplica el umbral sobre P(y=0):
        - Si P(y=0) >= zero_threshold -> 0
        - Si no -> 1
        """
        proba = self.predict_proba(X)
        p_zero = proba[:, 0]
        y_pred = np.where(p_zero >= self.zero_threshold, 0, 1)
        return y_pred

    def get_binary_outputs(self, X):
        """
        Devuelve info útil para análisis:

        - 'binary_pred': predicción 0/1 tras umbral
        - 'p_zero': P(y=0)
        """
        proba = self.predict_proba(X)
        p_zero = proba[:, 0]
        y_pred = np.where(p_zero >= self.zero_threshold, 0, 1)
        return {
            'binary_pred': y_pred,
            'p_zero': p_zero,
        }

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

from xgboost import XGBClassifier


class ZeroVsRestXGBoost(BaseEstimator, ClassifierMixin):
    """
    Binary model: 0 vs (1,2,3,4) using XGBoost.

    - Internally converts y to binary:
        y_bin = 0 if y == 0
        y_bin = 1 if y > 0

    - zero_threshold: threshold on P(y=0).
      If P(y=0) >= zero_threshold  -> prediction = 0
      If P(y=0) <  zero_threshold  -> prediction = 1
    """

    def __init__(
        self,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=42,
        zero_threshold=0.5,
        n_jobs=-1,
        # extra_kwargs to keep it flexible
        **xgb_kwargs
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.random_state = random_state
        self.zero_threshold = zero_threshold
        self.n_jobs = n_jobs
        self.xgb_kwargs = xgb_kwargs

    def _build_model(self):
        return XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            objective="binary:logistic",
            use_label_encoder=False,
            eval_metric="logloss",
            **self.xgb_kwargs
        )

    def fit(self, X, y):
        y = np.asarray(y)
        # 0 vs >0
        y_binary = (y > 0).astype(int)

        self.zero_vs_rest_ = self._build_model()
        self.zero_vs_rest_.fit(X, y_binary)

        # explicitly define binary classes [0, 1]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        """
        Returns P(y=0) and P(y>0) with columns aligned to classes_ = [0, 1].
        """
        # XGBClassifier returns proba for [class 0, class 1] by default
        proba = self.zero_vs_rest_.predict_proba(X)

        # make sure order matches [0,1] in case something weird happens
        idx_zero = np.where(self.zero_vs_rest_.classes_ == 0)[0][0]
        idx_one = np.where(self.zero_vs_rest_.classes_ == 1)[0][0]
        proba_ordered = proba[:, [idx_zero, idx_one]]

        return proba_ordered

    def predict(self, X):
        """
        Applies threshold on P(y=0):
        - If P(y=0) >= zero_threshold -> predicts 0
        - Otherwise -> predicts 1
        """
        proba = self.predict_proba(X)
        p_zero = proba[:, 0]
        y_pred = np.where(p_zero >= self.zero_threshold, 0, 1)
        return y_pred

    def get_binary_outputs(self, X):
        """
        Convenience method for analysis:

        Returns dict with:
        - 'binary_pred': 0/1 predictions after thresholding
        - 'p_zero': P(y=0)
        """
        proba = self.predict_proba(X)
        p_zero = proba[:, 0]
        y_pred = np.where(p_zero >= self.zero_threshold, 0, 1)

        return {
            "binary_pred": y_pred,
            "p_zero": p_zero,
        }

