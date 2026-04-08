import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    recall_score, precision_score, f1_score, 
    balanced_accuracy_score, roc_auc_score, 
    precision_recall_curve, auc
)

class FistaLogisticRegression:
    """
    Logistic Regression with L1 penalty using FISTA algorithm.
    """
    def __init__(self, max_iter=1000, tol=1e-4, 
                 lambda_range=None, measure='roc_auc',
                 X_valid=None, y_valid=None):
        self.max_iter = max_iter
        self.tol = tol
        
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.measure = measure
        self.lambda_range = lambda_range if lambda_range is not None else np.logspace(-4, 2, 50)
        
        # State variables
        self.coefficients_ = None
        self.intercept_ = None
        self.best_lambda_ = None
        self.validation_scores_ = {}
        self.coefs_paths_ = [] # To store coefficients for plotting
        
    #Helper functions for FISTA algorithm
    def _sigmoid(self, z):
        # Sigmoid activation function, maps real-valued input to (0, 1)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    
    def _soft_threshold(self, x, threshold):
        # Soft thresholding operator (proximal operator for L1 norm)
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def _compute_gradient(self, X_obs, y_obs, x, b):
        # Computes gradient of logistic loss for binary classification
        n = X_obs.shape[0]
        y_signed = 2.0 * y_obs - 1.0 # Convert {0,1} to {-1,1} for logistic loss
        z = X_obs @ x + b
        
        pred = self._sigmoid(y_signed * z)
        errors = (pred - 1.0) * y_signed
        
        grad_x = (X_obs.T @ errors) / n
        grad_b = np.mean(errors)
        return grad_x, grad_b
    
    def _fit_single_lambda(self, X, y, lambda_, learning_rate):
        """
        Internal method to run FISTA for a single lambda value.
        """
        n, p = X.shape
        x_prev = np.zeros(p)
        b_prev = 0.0
        
        t_k = 1.0
        y_k = x_prev.copy()
        y_k_b = b_prev
        
        for iteration in range(self.max_iter):
            # Step 1: Compute gradient of smooth part (logistic loss) at momentum point y_k
            grad_x, grad_b = self._compute_gradient(X, y, y_k, y_k_b)
            
            # Step 2: Gradient descent on smooth part
            x_temp = y_k - learning_rate * grad_x
            
            # Step 3: Proximal step - soft thresholding for L1 penalty
            x_k = self._soft_threshold(x_temp, learning_rate * lambda_)
            b_k = y_k_b - learning_rate * grad_b
            
            # Step 4: Nesterov acceleration - compute new momentum coefficient
            t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_k**2)) / 2.0
            momentum_coeff = (t_k - 1.0) / t_next
            
            # Step 5: Update momentum variables for next iteration
            y_k = x_k + momentum_coeff * (x_k - x_prev)
            y_k_b = b_k + momentum_coeff * (b_k - b_prev)

            # Convergence check:
            if np.linalg.norm(x_k - x_prev) < self.tol:
                break
                
            x_prev = x_k.copy()
            b_prev = b_k
            t_k = t_next
            
        return x_k, b_k

    def fit(self, X_train, y_train):
        """
        Fit model using cross-validation over lambda values.
        
        This method performs L-curve validation:
        - For each lambda in [lambda_min, lambda_max], solve the regularized problem
        - Evaluate validation score for each lambda
        - Select lambda with best validation performance
        """
        # Filter mask for missing data (marked as -1)
        mask = y_train != -1
        X_obs = X_train[mask]
        y_obs = y_train[mask]
        n = X_obs.shape[0]

        # Calculate safe Lipschitz-based learning rate
        L = np.linalg.norm(X_obs, ord=2)**2 / (4.0 * n)
        learning_rate = 1.0 / L if L > 0 else 0.01

        # Cross-validation over lambda_range
        self.validation_scores_[self.measure] = []
        self.coefs_paths_ = []
        best_score = -np.inf
        
        best_x, best_b = None, None
        
        # Grid search over lambda values
        for lambda_ in self.lambda_range:
            # Solve regularized problem for current lambda
            x_cur, b_cur = self._fit_single_lambda(X_obs, y_obs, lambda_, learning_rate)
            
            # Temporarily set coefficients to validate
            self.coefficients_ = x_cur
            self.intercept_ = b_cur
            self.coefs_paths_.append(x_cur.copy())
            
            # Evaluate on validation set
            score = self.validate(self.X_valid, self.y_valid, self.measure)
            self.validation_scores_[self.measure].append(score)
            
            # Keep track of best lambda
            if score > best_score:
                best_score = score
                self.best_lambda_ = lambda_
                best_x = x_cur.copy()
                best_b = b_cur
                
                
        # Set final model to the best found
        self.coefficients_ = best_x
        self.intercept_ = best_b

            
        return self
    
    def predict_proba(self, X):
        # Compute probability estimates using sigmoid
        z = X @ self.coefficients_ + self.intercept_
        proba = self._sigmoid(z)
        return np.column_stack([1 - proba, proba])
    
    def predict(self, X, threshold=0.5):
        # Hard classification using threshold
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
    
    def validate(self, X_valid, y_valid, measure):
        mask = y_valid != -1
        if np.sum(mask) == 0:
            return -np.inf
        
        X_obs = X_valid[mask]
        y_obs = y_valid[mask]
        
        proba = self.predict_proba(X_obs)[:, 1]
        y_pred = self.predict(X_obs)
        
        has_both_classes = len(np.unique(y_obs)) == 2
        
        if measure == 'recall':
            return recall_score(y_obs, y_pred, zero_division=0)
        elif measure == 'precision':
            return precision_score(y_obs, y_pred, zero_division=0)
        elif measure == 'f1':
            return f1_score(y_obs, y_pred, zero_division=0)
        elif measure == 'balanced_accuracy':
            return balanced_accuracy_score(y_obs, y_pred)
        elif measure == 'roc_auc':
            return roc_auc_score(y_obs, proba) if has_both_classes else np.mean(y_pred == y_obs)
        elif measure == 'pr_auc':
            if has_both_classes:
                precision_vals, recall_vals, _ = precision_recall_curve(y_obs, proba)
                return auc(recall_vals, precision_vals)
            else:
                return np.mean(y_pred == y_obs)
        else:
            raise ValueError(f"Unknown measure: {measure}")
            
    def plot(self, measure, ax=None, label='FISTA', **kwargs):
        # Plot validation scores vs lambda on log scale
        if measure not in self.validation_scores_:
            raise ValueError(f"No scores for measure: {measure}")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
            
        scores = self.validation_scores_[measure]
        ax.semilogx(self.lambda_range, scores, 'o-', linewidth=2, markersize=5, label=label, **kwargs)
        if self.best_lambda_ is not None:
            ax.axvline(self.best_lambda_, color='r', linestyle='--', label=f'Best lambda={self.best_lambda_:.4f}')
        
        ax.set_xlabel('Lambda')
        ax.set_ylabel(measure.replace('_', ' ').title())
        ax.set_title('Validation Score vs Lambda')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax
    
    def plot_coefficients(self, ax=None, top_n=20):
        # Plot coefficient paths (regularization paths) vs lambda
        if not self.coefs_paths_:
            raise ValueError("Model must be fitted with lambda_range first")
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
            
        coefs_matrix = np.array(self.coefs_paths_)
        
        for i in range(min(top_n, coefs_matrix.shape[1])):
            ax.semilogx(self.lambda_range, coefs_matrix[:, i], linewidth=1.5, alpha=0.8, label=f'Feature {i}')
            
        ax.set_xlabel('Lambda')
        ax.set_ylabel('Coefficient Value')
        ax.set_title('Coefficient Paths (Lasso)')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linestyle='-', linewidth=0.5)

        return ax