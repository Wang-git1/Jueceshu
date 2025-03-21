class SimpleDecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    class Node:
        def __init__(self, feature=None, value=None, left=None, right=None, label=None):
            self.feature = feature  # 分裂特征索引
            self.value = value  # 分裂阈值
            self.left = left  # 左子树（<=value）
            self.right = right  # 右子树（>value）
            self.label = label  # 叶节点的类别

    def fit(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return self.Node(label=np.argmax(np.bincount(y)))

        best_gini = float('inf')
        best_feature = None
        best_value = None

        # 随机选择sqrt(n_features)个特征
        n_features = X.shape[1]
        selected_features = np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)

        for feature in selected_features:
            values = np.unique(X[:, feature])
            for value in values:
                left_mask = X[:, feature] <= value
                if np.sum(left_mask) == 0 or np.sum(~left_mask) == 0:
                    continue
                curr_gini = self._calc_weighted_gini(y[left_mask], y[~left_mask])
                if curr_gini < best_gini:
                    best_gini = curr_gini
                    best_feature = feature
                    best_value = value

        if best_feature is None:
            return self.Node(label=np.argmax(np.bincount(y)))

        left_mask = X[:, best_feature] <= best_value
        left = self.fit(X[left_mask], y[left_mask], depth + 1)
        right = self.fit(X[~left_mask], y[~left_mask], depth + 1)
        return self.Node(feature=best_feature, value=best_value, left=left, right=right)

    def predict(self, x):
        node = self.tree
        while node.label is None:
            if x[node.feature] <= node.value:
                node = node.left
            else:
                node = node.right
        return node.label