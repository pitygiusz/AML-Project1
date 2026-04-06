import numpy as np


class DataGenerator:
    """
    A class for generating synthetic binary classification data together with
    different missing-label mechanisms.

    The class first generates a feature matrix X from a multivariate normal-like
    independent distribution and then constructs a binary target variable y
    using a logistic model:
        P(Y = 1 | X) = sigmoid(X @ beta)

    After generating fully labeled data (X, y), the class can simulate four
    missing-label mechanisms and return observed labels y_obs, where:
        - y_obs[i] = -1 means the label is missing,
        - y_obs[i] in {0, 1} means the label is observed.

    Implemented missingness mechanisms:
        - MCAR: Missing Completely at Random
        - MAR1: Missing at Random depending on one explanatory variable
        - MAR2: Missing at Random depending on all explanatory variables
        - MNAR: Missing Not at Random depending on X and y

    Parameters
    ----------
    p : int, default=2
        Number of explanatory variables (features).

    n : int, default=1000
        Number of observations.

    mean : float, default=0
        Mean of the normal distribution used to generate X.

    std : float, default=1
        Standard deviation of the normal distribution used to generate X.

    c : float, default=0.1
        Probability of missingness in the MCAR mechanism.

    seed : int, default=42
        Random seed used for reproducibility.

    Attributes
    ----------
    n : int
        Number of observations.

    p : int
        Number of features.

    mean : float
        Mean of the feature-generating distribution.

    std : float
        Standard deviation of the feature-generating distribution.

    c : float
        Missingness probability for MCAR.

    rng : numpy.random.Generator
        NumPy random number generator.

    X : ndarray of shape (n, p)
        Generated feature matrix.

    beta : ndarray of shape (p,)
        Coefficient vector used to generate logistic probabilities.

    y : ndarray of shape (n,)
        Generated binary target vector.
    """

    def __init__(self, p=2, n=1000, mean=0, std=1, c=0.1, seed=42):
        """
        Initialize the data generator and create fully labeled synthetic data.

        Parameters
        ----------
        p : int, default=2
            Number of explanatory variables.

        n : int, default=1000
            Number of observations.

        mean : float, default=0
            Mean of the normal distribution used for feature generation.

        std : float, default=1
            Standard deviation of the normal distribution used for feature generation.

        c : float, default=0.1
            Probability of missingness in the MCAR mechanism.

        seed : int, default=42
            Random seed for reproducibility.
        """
        self.n = n
        self.p = p
        self.mean = mean
        self.std = std
        self.c = c
        self.rng = np.random.default_rng(seed=seed)

        self.X = self.rng.normal(loc=mean, scale=std, size=(n, p))
        self.beta = self.rng.normal(size=self.p)
        self.y = self.generate_target_value()

    def generate_target_value(self) -> np.ndarray:
        """
        Generate the binary target vector y using a logistic model.

        The target is generated according to:
            logits = X @ beta
            probs = sigmoid(logits)
            y ~ Bernoulli(probs)

        Returns
        -------
        ndarray of shape (n,)
            Binary target vector containing values 0 or 1.
        """
        logits = self.X @ self.beta
        probs = 1 / (1 + np.exp(-logits))
        return self.rng.binomial(n=1, p=probs, size=self.n)

    def MCAR(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate observed labels under the MCAR mechanism.

        In the MCAR (Missing Completely at Random) setting, missingness is
        independent of both X and y:
            P(S = 1 | X, Y) = c

        Here:
            - S = 1 means the label is missing,
            - S = 0 means the label is observed.

        Returns
        -------
        X : ndarray of shape (n, p)
            Copy of the feature matrix.

        y_obs : ndarray of shape (n,)
            Observed target vector, where missing labels are encoded as -1.
        """
        S = self.rng.binomial(n=1, p=self.c, size=self.n)
        y_obs = np.where(S == 1, -1, self.y)
        return self.X.copy(), y_obs

    def MAR1(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate observed labels under the MAR1 mechanism.

        In the MAR1 setting, missingness depends only on one randomly selected
        explanatory variable:
            P(S = 1 | X, Y) = P(S = 1 | X_j)

        A single feature column is randomly selected, then transformed into
        missingness probabilities through the logistic function.

        Returns
        -------
        X : ndarray of shape (n, p)
            Copy of the feature matrix.

        y_obs : ndarray of shape (n,)
            Observed target vector, where missing labels are encoded as -1.
        """
        idx = self.rng.integers(0, self.p)
        scores = self.X[:, idx]
        probs = 1 / (1 + np.exp(-scores))
        S = self.rng.binomial(n=1, p=probs, size=self.n)
        y_obs = np.where(S == 1, -1, self.y)
        return self.X.copy(), y_obs

    def MAR2(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate observed labels under the MAR2 mechanism.

        In the MAR2 setting, missingness depends on all explanatory variables:
            P(S = 1 | X, Y) = P(S = 1 | X)

        A combined score based on all features is computed and transformed into
        missingness probabilities through the logistic function.

        Returns
        -------
        X : ndarray of shape (n, p)
            Copy of the feature matrix.

        y_obs : ndarray of shape (n,)
            Observed target vector, where missing labels are encoded as -1.
        """
        scores = np.sum(self.X, axis=1) / np.sqrt(self.p)
        probs = 1 / (1 + np.exp(-scores))
        S = self.rng.binomial(n=1, p=probs, size=self.n)
        y_obs = np.where(S == 1, -1, self.y)
        return self.X.copy(), y_obs

    def MNAR(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate observed labels under the MNAR mechanism.

        In the MNAR (Missing Not at Random) setting, missingness depends on
        both explanatory variables X and the true target y:
            P(S = 1 | X, Y)

        A combined score based on X and y is computed and transformed into
        missingness probabilities through the logistic function.

        Returns
        -------
        X : ndarray of shape (n, p)
            Copy of the feature matrix.

        y_obs : ndarray of shape (n,)
            Observed target vector, where missing labels are encoded as -1.
        """
        scores = np.sum(self.X, axis=1) / np.sqrt(self.p) + 0.8 * self.y
        probs = 1 / (1 + np.exp(-scores))
        S = self.rng.binomial(n=1, p=probs, size=self.n)
        y_obs = np.where(S == 1, -1, self.y)
        return self.X.copy(), y_obs