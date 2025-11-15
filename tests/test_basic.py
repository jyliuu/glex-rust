import glex_rust
import numpy as np

import xgboost as xgb


def test_extract_trees_from_xgboost_smoke():
    # Generate 100 data points with 2 larger trees for a more comprehensive test.
    np.random.seed(0)
    X = np.random.rand(100, 2) * 10  # 100 points with 2 features
    y = np.random.randint(0, 2, size=100)  # Binary classification labels

    model = xgb.XGBClassifier(
        n_estimators=2,
        max_depth=5,
        random_state=0,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X, y)

    trees = glex_rust.extract_trees_from_xgboost(model)
    assert len(trees) == 2

    # Test tree object methods for both trees
    for i, tree in enumerate(trees):
        assert tree.num_nodes() > 0
        assert tree.root() == 0

        # Test tree structure printing
        tree_str = tree.format_tree()
        assert isinstance(tree_str, str)
        assert len(tree_str) > 0
        print(f"\nTree {i + 1} structure:")
        print(tree_str)
