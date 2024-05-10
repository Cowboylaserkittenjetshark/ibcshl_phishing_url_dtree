from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report
from joblib import parallel_backend
from common import X, y, fetched_features
from trials import run_trial
from paths import output
import logging

run_trial("Decision Tree", DecisionTreeClassifier(random_state=42), X, y)
exit(1)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(message)s'
)

def feature_importance(model):
    def sortSecond(val):
        return val[1]

    values = model.feature_importances_
    features = list(X)
    importances = [(features[i], values[i]) for i in range(len(features))]
    importances.sort(reverse=True, key=sortSecond)
    return importances

with parallel_backend('threading', n_jobs=-1):
    logging.debug('Splitting data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
    test_size=0.2)

    logging.info('==== Decision Tree ====')
    dt_clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)

    dt_score = dt_clf.score(X_valid, y_valid)
    dt_cross_val_score = cross_val_score(dt_clf, X_train, y_train, cv=7)
    dt_report = classification_report(y_valid, dt_clf.predict(X_valid))
    dt_feat_importance = feature_importance(dt_clf)

    logging.info('Score: %s\n', dt_score)
    logging.info('Cross val score: %s\n', dt_cross_val_score)
    logging.info('Classification report:\n%s\n', dt_report)
    logging.debug('Feature importance: %s', str(dt_feat_importance))
    
    dt_test_report = classification_report(y_test, dt_clf.predict(X_test))
    logging.info('Test classification report:\n%s\n', dt_report)

    logging.info('==== Decision Tree with feature selection')
    X_train_cut = X_train[[col[0] for col in dt_feat_importance[:15]]]
    X_test_cut  = X_test[[col[0] for col in dt_feat_importance[:15]]]
    X_valid_cut = X_valid[[col[0] for col in dt_feat_importance[:15]]]

    dt_clf = DecisionTreeClassifier(random_state=42).fit(X_train_cut, y_train)

    dt_score = dt_clf.score(X_valid_cut, y_valid)
    dt_cross_val_score = cross_val_score(dt_clf, X_train_cut, y_train, cv=7)
    dt_report = classification_report(y_test, dt_clf.predict(X_test_cut))

    logging.info('Score: %s\n', dt_score)
    logging.info('Cross val score: %s\n', dt_cross_val_score)
    logging.info('Classification report:\n%s\n', dt_report)
    export_graphviz(dt_clf, out_file=str(output.DIR.joinpath("dtree_cut.dot")), feature_names=list(X_train_cut), class_names=True, rounded=True, filled=True)


    logging.info('==== Decision Tree tuned ====')
    # param_dist = {
    #     'criterion': optuna.distributions.CategoricalDistribution(['gini', 'entropy']),
    #     'min_samples_split': optuna.distributions.IntDistribution(2, 40),
    #     'min_samples_leaf': optuna.distributions.IntDistribution(2,20),
    #     'max_depth': optuna.distributions.IntDistribution(10,80)
    # }

    # optuna_search = optuna.integration.OptunaSearchCV(
    #     DecisionTreeClassifier(random_state=42), param_dist, n_trials=100, timeout=600, verbose=2
    # )

    # optuna_search.fit(X, y)

    # print("Best trial:")
    # trial = optuna_search.study_.best_trial

    # print("  Value: ", trial.value)
    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))
    
    dt_clf = DecisionTreeClassifier(random_state=42, criterion='entropy', min_samples_leaf=2, min_samples_split=2, max_depth=44).fit(X_train, y_train)

    dt_score = dt_clf.score(X_valid, y_valid)
    dt_cross_val_score = cross_val_score(dt_clf, X_train, y_train, cv=7)
    dt_report = classification_report(y_test, dt_clf.predict(X_test))

    logging.info('Score: %s\n', dt_score)
    logging.info('Cross val score: %s\n', dt_cross_val_score)
    logging.info('Classification report:\n%s\n', dt_report)
    
    logging.info('==== Decision Tree without fetched features  ====')
    X_train = X_train.drop(fetched_features, axis = 1)
    X_test = X_test.drop(fetched_features, axis = 1)
    X_valid = X_valid.drop(fetched_features, axis = 1)
    dt_clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)

    dt_score = dt_clf.score(X_valid, y_valid)
    dt_cross_val_score = cross_val_score(dt_clf, X_train, y_train, cv=7)
    dt_report = classification_report(y_valid, dt_clf.predict(X_valid))

    logging.info('Score: %s\n', dt_score)
    logging.info('Cross val score: %s\n', dt_cross_val_score)
    logging.info('Classification report:\n%s\n', dt_report)
    
    dt_test_report = classification_report(y_test, dt_clf.predict(X_test))
    logging.info('Test classification report:\n%s\n', dt_report)
