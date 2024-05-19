import common
from common.data import X, y, fetched_features
from trials import run_trial
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import matplotlib.pyplot as plt
from paths import output

class_names = [ "legitimate", "Phishing" ]
feature_names = list(X.drop(fetched_features, axis=1))

common_trial_args = {
    'X': X,
    'y': y,
    'fit_all_finally': True,
}

common_clf_args = {
    'random_state': 42,
    'max_depth': 2,
}

local_clf = run_trial("Decision Tree", DecisionTreeClassifier(**common_clf_args), with_fetched=False, **common_trial_args)
fetched_clf = run_trial("Decision Tree", DecisionTreeClassifier(**common_clf_args), with_fetched=True, **common_trial_args)

common_plot_args = {
    'class_names': class_names,
    'rounded': True,
    'filled': True
}

export_graphviz(local_clf, out_file=str(output.DIR.joinpath("dtree_local.dot")), feature_names=feature_names,  **common_plot_args)
export_graphviz(fetched_clf, out_file=str(output.DIR.joinpath("dtree_fetched.dot")), feature_names=feature_names + fetched_features, **common_plot_args)

# ❯ -Tpng -Gdpi=300 -Gbgcolor=transparent -Efontcolor=white output/dtree_??????.dot -o ??????.png
