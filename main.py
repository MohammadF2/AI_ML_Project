from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics
import pandas as pd
import collections


def getData():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    col_names = ['NPG', 'PGL', 'DIA', 'TSF', 'INS', 'BMI', 'DPF', 'AGE', 'Diabetic']
    dataRes = pd.read_csv('Data.csv', header=None, names=col_names)
    return dataRes


data = getData()


def get_train_test(size):
    feature_cols = ['NPG', 'PGL', 'DIA', 'TSF', 'INS', 'BMI', 'DPF', 'AGE']
    features = data[feature_cols]
    target = data.Diabetic

    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=size,
                                                                                random_state=1)
    return features_train, features_test, target_train, target_test


def getModel(features_train, target_train):
    model = DecisionTreeClassifier()

    model = model.fit(features_train, target_train)

    return model


def getAccuracy(model, features_test, target_test):
    target_pred = model.predict(features_test)
    return metrics.accuracy_score(target_test, target_pred)


def printStatistics():
    statistics = data.describe()
    statistics = statistics.loc[['mean', '50%', 'std', 'min', 'max'], :]
    print(statistics)


def printDistribution():
    target = data.Diabetic
    distribution = collections.Counter(target)

    # Calculate the proportions
    total = len(target)
    proportions = {k: (v / total) * 100 for k, v in distribution.items()}

    # Print the percentages
    for k, v in proportions.items():
        print(f'{k}: {v}%')


# print Statistics
print("Statistics:")
printStatistics()

# print distribution
print("\nDistribution:")
printDistribution()

# split data:
features_train_M1, features_test_M1, target_train_M1, target_test_M1 = get_train_test(0.3)
features_train_M2, features_test_M2, target_train_M2, target_test_M2 = get_train_test(0.5)

M1 = getModel(features_train_M1, target_train_M1)
M2 = getModel(features_train_M2, target_train_M2)

print(f'\nModel 1 (30%): {getAccuracy(M1, features_test_M1, target_test_M1)}')
print(f'Model 2 (50%): {getAccuracy(M2, features_test_M2, target_test_M2)}')



