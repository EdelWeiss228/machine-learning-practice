import numpy as np
import heapq

class KDTree:
    class Node:
        __slots__ = ['axis', 'split', 'left', 'right', 'indices']
        def __init__(self, axis=None, split=None, left=None, right=None, indices=None):
            self.axis = axis
            self.split = split
            self.left = left
            self.right = right
            self.indices = indices

    def __init__(self, X, leaf_size=30, p=2):
        self.X = np.asarray(X, dtype=np.float32)
        self.leaf_size = leaf_size
        self.p = p
        self.root = self._build_tree(X, np.arange(X.shape[0]), 0)
    
    def _build_tree(self, X, indices, depth):
        if len(indices) <= self.leaf_size:
            return self.Node(indices=indices)
        
        axis = depth % X.shape[1]
        sorted_indices = indices[np.argsort(X[indices, axis])]
        mid = len(sorted_indices) // 2
        
        return self.Node(
            axis=axis,
            split=X[sorted_indices[mid], axis],
            left=self._build_tree(X, sorted_indices[:mid], depth+1),
            right=self._build_tree(X, sorted_indices[mid+1:], depth+1)
        )

    def _distance(self, a, b):
        diff = a - b
        if self.p == 1:
            return np.sum(np.abs(diff))
        elif self.p == 2:
            return np.sqrt(np.sum(diff**2))
        else:
            raise ValueError("p must be 1 or 2")

    def query(self, queries, k=1):
        queries = np.asarray(queries, dtype=np.float32)
        distances = np.empty((queries.shape[0], k), dtype=np.float32)
        indices = np.empty((queries.shape[0], k), dtype=np.int32)
        
        for i, x in enumerate(queries):
            heap = []
            stack = [self.root]
            
            while stack:
                node = stack.pop()
                
                if node.indices is not None:
                    for idx in node.indices:
                        dist = self._distance(x, self.X[idx])
                        if len(heap) < k:
                            heapq.heappush(heap, (-dist, idx))
                        elif dist < -heap[0][0]:
                            heapq.heappushpop(heap, (-dist, idx))
                else:
                    if x[node.axis] <= node.split:
                        stack.extend([node.right, node.left])
                    else:
                        stack.extend([node.left, node.right])
            
            heap.sort(reverse=True)
            distances[i] = [-d for d, _ in heap]
            indices[i] = [idx for _, idx in heap]
        
        return distances, indices

class KNeighborsClassifier:
    def __init__(self, n_neighbors=5, weights='uniform', leaf_size=30, p=2):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.leaf_size = leaf_size
        self.p = p
        self.tree = None
        self.classes_ = None
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=np.float32)
        self.y_train = np.asarray(y, dtype=np.int32)
        self.classes_ = np.unique(y)
        self.tree = KDTree(X, self.leaf_size, self.p)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        distances, indices = self.tree.query(X, self.n_neighbors)
        
        if self.weights == 'distance':
            weights = 1 / (distances + 1e-8)
        else:
            weights = np.ones_like(distances)
        
        votes = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float32)
        rows = np.repeat(np.arange(X.shape[0])[:, None], self.n_neighbors, axis=1).ravel()
        cols = self.y_train[indices].ravel()
        np.add.at(votes, (rows, cols), weights.ravel())
        
        return self.classes_[np.argmax(votes, axis=1)]

if __name__ == "__main__":
    with np.load('mnist.npz') as data:
        X_train = data['x_train'].reshape(-1, 784).astype(np.float32) / 255.0
        y_train = data['y_train'].astype(np.int32)
        X_test = data['x_test'].reshape(-1, 784).astype(np.float32) / 255.0
        y_test = data['y_test'].astype(np.int32)

    knn = KNeighborsClassifier(n_neighbors=5, weights='distance', p=2)
    knn.fit(X_train, y_train)  

    test_samples = 10000
    y_pred = knn.predict(X_test[:test_samples])

    accuracy = np.mean(y_pred == y_test[:test_samples])
    print(f"Accuracy on {test_samples} test samples: {accuracy:.3f}")