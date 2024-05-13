import common
from common.data import X, y
from trials import run_trial

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


def run_all_trials(name, classifier, params):
    local_untuned_clf = run_trial(name, classifier, X, y, with_fetched=False)
    fetched_untuned_clf = run_trial(name, classifier, X, y, with_fetched=True)
    fim = None
    if hasattr(local_untuned_clf, "feature_importances_"):
        fim = local_untuned_clf
    run_trial(
        name,
        classifier,
        X,
        y,
        with_fetched=False,
        params=params,
        feature_importance_model=fim,
    )
    fim = None
    if hasattr(local_untuned_clf, "feature_importances_"):
        fim = fetched_untuned_clf
    run_trial(
        name,
        classifier,
        X,
        y,
        with_fetched=True,
        params=params,
        feature_importance_model=fim,
    )


random_state = 42

name = "Decision Tree"
classifier = DecisionTreeClassifier(random_state=random_state)
params = {
    "criterion": ["gini", "entropy"],
    "max_depth": [4, 16, None],
    "min_samples_split": range(2, 41, 4),
    "min_samples_leaf": range(2, 21, 4),
}
run_all_trials(name, classifier, params)

name = "Random Forest"
classifier = RandomForestClassifier(random_state=random_state)
params["n_estimators"] = range(100, 1000, 200)
run_all_trials(name, classifier, params)

name = "Hist Gradient Boosting"
classifier = HistGradientBoostingClassifier(random_state=random_state)
params = None
run_all_trials(name, classifier, params)

name = "Logistic Regression"
classifier = make_pipeline(
    SimpleImputer(missing_values=-1),
    StandardScaler(),
    LogisticRegression(random_state=random_state),
)
params = None
run_all_trials(name, classifier, params)
