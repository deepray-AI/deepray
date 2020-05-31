import pandas as pd
from sklearn.preprocessing import LabelEncoder


class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode

    def fit(self, X, y=None):
        return self  # not relevant here

    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


if __name__ == '__main__':
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [6, 6, 5]})
    print(df)
    MultiColumnLabelEncoder(columns=['a', 'b']).fit_transform(df)
    test = df.apply(LabelEncoder().fit_transform)
    print(test)
