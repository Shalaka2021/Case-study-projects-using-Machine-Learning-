from sklearn.datasets import load_iris;
import numpy as np;
from sklearn import tree;
from sklearn.model_selection import train_test_split;
from sklearn.metrics import accuracy_score;
import pickle;


def DecisionTree():
	data=load_iris();

	print(data.feature_names);
	print(data.target_names);

	train_features,test_features,train_target,test_target=train_test_split(data.data,data.target,test_size=0.3);

	clf=tree.DecisionTreeClassifier();

	clf.fit(train_features,train_target);
	predicted=clf.predict(test_features);

	accuracy=accuracy_score(test_target,predicted);

	print("Accuracy is ",accuracy*100);

	"""fobj=open("model_pickle","wb");
	pickle.dump(clf,fobj);"""


def main():
	DecisionTree();



if __name__=="__main__":
	main();

