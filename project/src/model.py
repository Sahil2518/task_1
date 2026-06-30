from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from src.config import CONFIG


def get_baseline():
    """Majority-class baseline — the dumb model to beat."""
    return DummyClassifier(
        strategy="most_frequent",
        random_state=CONFIG["seed"]
    )


def get_model():
    """Non-linear Gradient Boosting model."""
    return GradientBoostingClassifier(
        random_state=CONFIG["seed"]
    )

