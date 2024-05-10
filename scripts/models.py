import common
from common import X, y
from trials import run_trial

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

random_state = 42

name = "Decision Tree"
classifier = DecisionTreeClassifier(random_state=random_state)
param_grid = {
    'criterion': ['gini', 'entropy'],
    'min_samples_split': range(2, 41, 2),
    'min_samples_leaf': range(2, 21, 2),
}

run_trial(name, classifier, X, y, with_fetched=False)
run_trial(name, classifier, X, y, with_fetched=True)
run_trial(name, classifier, X, y, with_fetched=False, param_grid=param_grid)
run_trial(name, classifier, X, y, with_fetched=True, param_grid=param_grid)

name = "Random Forest"
classifier = RandomForestClassifier(random_state=random_state)
param_grid = None

run_trial(name, classifier, X, y, with_fetched=False)
run_trial(name, classifier, X, y, with_fetched=True)
run_trial(name, classifier, X, y, with_fetched=False, param_grid=param_grid)
run_trial(name, classifier, X, y, with_fetched=True, param_grid=param_grid)

name = "Hist Gradient Boosting"
classifier = HistGradientBoostingClassifier(random_state=random_state)
param_grid = None

run_trial(name, classifier, X, y, with_fetched=False)
run_trial(name, classifier, X, y, with_fetched=True)
run_trial(name, classifier, X, y, with_fetched=False, param_grid=param_grid)
run_trial(name, classifier, X, y, with_fetched=True, param_grid=param_grid)

name = "Logistic Regression"
classifier = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
param_grid = None

run_trial(name, classifier, X, y, with_fetched=False)
run_trial(name, classifier, X, y, with_fetched=True)
run_trial(name, classifier, X, y, with_fetched=False, param_grid=param_grid)
run_trial(name, classifier, X, y, with_fetched=True, param_grid=param_grid)
