from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from src.config import CONFIG


def get_baseline():
    """Majority-class baseline — the dumb model to beat."""
    return DummyClassifier(
        strategy="most_frequent",
        random_state=CONFIG["seed"]
    )


def get_model():
    """Binary Logistic Regression model."""
    return LogisticRegression(
        random_state=CONFIG["seed"],
        max_iter=1000
    )

