from sklearn.linear_model import LinearRegression

from libuplift.classifiers import MemoizedClassifier

def test_memoized_classifier(tmp_path):
    cache = tmp_path / "cache"
    lr = LinearRegression()
    mlr = MemoizedClassifier(lr, str(cache))
    X = [[1.0,0], [2,1], [2,3]]
    y = [1.0, 2.0, 3.0]
    mlr.fit(X, y)
    
    lr2 = LinearRegression()
    mlr2 = MemoizedClassifier(lr, str(cache))
    mlr2.fit(X, y)
