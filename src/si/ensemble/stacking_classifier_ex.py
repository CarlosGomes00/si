from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.ensemble.stacking_classifier import StackingClassifier


csv_file = '/Users/carla/PycharmProjects/Mestrado/SI_ML/Sistemas_Inteligentes_ML/si/datasets/breast_bin/breast-bin.csv'
dataset = read_csv(filename = csv_file, features = True, label = True)

train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=1)

knn1 = KNNClassifier(k=3)
logistic_regression = LogisticRegression()
decision_tree = DecisionTreeClassifier()
knn2 = KNNClassifier(k=3)

models = [knn1, logistic_regression, decision_tree]
final_model = knn2

stacking_classifier = StackingClassifier(models, final_model)

stacking_classifier._fit(train_dataset)

predictions = stacking_classifier._predict(test_dataset)
accuracy = stacking_classifier._score(test_dataset, predictions)

print("Predictions:", predictions)
print("Accuracy:", accuracy)


