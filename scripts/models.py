import common
from common import X, y
from trials import run_trial

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import randint

random_state = 42

name = "Decision Tree"
classifier = DecisionTreeClassifier(random_state=random_state)
params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [4, 8, 16, 32, 64, None],
    'min_samples_split': randint(2, 41),
    'min_samples_leaf': randint(2, 21),
}

run_trial(name, classifier, X, y, with_fetched=False)
run_trial(name, classifier, X, y, with_fetched=True)
run_trial(name, classifier, X, y, with_fetched=False, params=params)
run_trial(name, classifier, X, y, with_fetched=True, params=params)

name = "Random Forest"
classifier = RandomForestClassifier(random_state=random_state)
params = None

run_trial(name, classifier, X, y, with_fetched=False)
run_trial(name, classifier, X, y, with_fetched=True)
run_trial(name, classifier, X, y, with_fetched=False, params=params)
run_trial(name, classifier, X, y, with_fetched=True, params=params)

name = "Hist Gradient Boosting"
classifier = HistGradientBoostingClassifier(random_state=random_state)
params = None

run_trial(name, classifier, X, y, with_fetched=False)
run_trial(name, classifier, X, y, with_fetched=True)
run_trial(name, classifier, X, y, with_fetched=False, params=params)
run_trial(name, classifier, X, y, with_fetched=True, params=params)

name = "Logistic Regression"
classifier = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
params = None

run_trial(name, classifier, X, y, with_fetched=False)
run_trial(name, classifier, X, y, with_fetched=True)
run_trial(name, classifier, X, y, with_fetched=False, params=params)
run_trial(name, classifier, X, y, with_fetched=True, params=params)
