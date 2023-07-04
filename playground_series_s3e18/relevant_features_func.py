def get_relevant_variables(X_processed, y, seeds, classifier=None, params=None):
    X = X_processed.copy()

    X["RANDOM_1"] = np.random.normal(size=len(X))
    X["RANDOM_2"] = np.random.normal(size=len(X))
    X["RANDOM_3"] = np.random.normal(size=len(X))
    X["RANDOM_4"] = np.random.normal(size=len(X))
    X["RANDOM_5"] = np.random.normal(size=len(X))

    feature_names = X.columns.to_list()
    results = pd.DataFrame(index=feature_names)

    if not classifier:
        classifier = LGBMClassifier

    if not params:
        params = {}

    for seed in seeds:
        classifier_on_seed = classifier(random_state=seed, **params).fit(X, y)
        if hasattr(classifier_on_seed, "feature_importances_"):
            results[seed] = pd.Series(classifier_on_seed.feature_importances_, index=feature_names)
        elif hasattr(classifier_on_seed, "coef_"):
            results[seed] = pd.Series(np.abs(classifier_on_seed.coef_[0]), index=feature_names)
        else:
            raise AttributeError("Classifier does not have importance attribute!")

    results["SUM"] = results.sum(axis=1)
    results.sort_values(by="SUM", ascending=False, inplace=True)
    highest_random_id = results[results.index.str.startswith("RANDOM")].index[0]
    value_at_random = results.loc[highest_random_id, "SUM"]  # type: ignore

    return results  # results.query(f"SUM > {value_at_random}").index.to_list()
