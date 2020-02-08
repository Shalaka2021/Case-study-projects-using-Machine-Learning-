from sklearn.datasets import load_iris;
from sklearn.model_selection import train_test_split;
from sklearn.metrics import accuracy_score;
from scipy.spatial import distance;

def euc(a,b):
	return distance.euclidean(a,b);

class MyKNeighborsClassifier:
	def fit(self,train_data,train_target):
		self.train_data=train_data;
		self.train_target=train_target;

	def predict(self,test_data):
		predictions=[];
		for row in test_data:
			label=self.closest(row);
			predictions.append(label);
		return predictions;

	def closest(self,row):
		bestdistance=euc(row,self.train_data[0]);
		bestindex=0;
		for i in range(1,len(self.train_data)):
			dist=euc(row,self.train_data[i]);
			if dist<bestdistance:
				bestdistance=dist;
				bestindex=i;
		return self.train_target[bestindex];


def MyKNNAccuracy():
	dataset=load_iris();

	data=dataset.data;
	target=dataset.target;

	train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.5);

	clf=MyKNeighborsClassifier();

	clf.fit(train_data,train_target);
	predictions=clf.predict(test_data);

	accuracy=accuracy_score(test_target,predictions);

	print(accuracy);

	return accuracy;

def main():
	Accuracy=MyKNNAccuracy();
	print("Accuracy is ",Accuracy*100,"%");

if __name__=="__main__":
	main();