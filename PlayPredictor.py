import pandas;
import sys;
from sklearn import preprocessing;
from sklearn.model_selection import train_test_split;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.metrics import accuracy_score;

def PlayPredictor(file_path):
	data=pandas.read_csv(file_path);

	#print(data);

	weather=data.Wether;
	temperature=data.Temperature;
	play=data.Play;

	#print(weather);
	#print(temperature);
	#print(play);

	le=preprocessing.LabelEncoder();
	weather=le.fit_transform(weather);
	temperature=le.fit_transform(temperature);
	play=le.fit_transform(play);

	#print(weather," ",temperature," ",play);

	features=zip(weather,temperature);

	features=list(features);

	#print(features);

	train_data,test_data,train_target,test_target=train_test_split(features,play,test_size=0.3);

	clf=KNeighborsClassifier(n_neighbors=3);

	clf.fit(train_data,train_target);

	predicted=clf.predict(test_data);

	return predicted,test_target;

def Accuracy(predicted,test_target):
	accuracy=accuracy_score(test_target,predicted);

	return accuracy;


def main():

	predicted,test_target=PlayPredictor(sys.argv[1]);
	accuracy=Accuracy(predicted,test_target);
	print("Accuracy is ",accuracy*100);

if __name__=="__main__":
	main();