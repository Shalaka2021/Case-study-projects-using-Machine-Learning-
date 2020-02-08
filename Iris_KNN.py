from sklearn.datasets import load_iris;
from sklearn.metrics import accuracy_score;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.model_selection import train_test_split;

def KNN():
	data=load_iris();

	print(data.feature_names);
	print(data.target_names);

	train_features,test_features,train_target,test_target=train_test_split(data.data,data.target,test_size=0.3,shuffle=True,
		random_state=10);

	clf=KNeighborsClassifier();

	clf.fit(train_features,train_target);

	predicted=clf.predict(test_features);

	accuracy=accuracy_score(test_target,predicted);

	return accuracy;


def main():
	accuracy=KNN();
	print("Accuracy is ",accuracy*100);

if __name__=="__main__":
	main();