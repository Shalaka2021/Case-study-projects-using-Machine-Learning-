import pandas;
import sys;
from sklearn.model_selection import train_test_split;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.metrics import accuracy_score;
from sklearn import preprocessing;

def WinePredictor(file_path):
	data=pandas.read_csv(file_path);

	#print(data);

	alcohol=data.Alcohol;
	malic_acid=data.Malicacid;
	ash=data.Ash;
	alcalinityofash=data.Alcalinityofash;
	magnesium=data.Magnesium;
	totalphenols=data.Totalphenols;
	flavanoids=data.Flavanoids;
	nonflavanoidphenols=data.Nonflavanoidphenols;
	proanthocyanins=data.Proanthocyanins;
	colorintensity=data.Colorintensity;
	hue=data.Hue;
	dilutedwines=data.dilutedwines;
	proline=data.Proline;

	target=data.Class;

	le=preprocessing.LabelEncoder();


	alcohol=le.fit_transform(alcohol);
	malic_acid=le.fit_transform(malic_acid);
	ash=le.fit_transform(ash);
	alcalinityofash=le.fit_transform(alcalinityofash);
	magnesium=le.fit_transform(magnesium);
	totalphenols=le.fit_transform(totalphenols);
	flavanoids=le.fit_transform(flavanoids);
	nonflavanoidphenols=le.fit_transform(nonflavanoidphenols);
	proanthocyanins=le.fit_transform(proanthocyanins);
	colorintensity=le.fit_transform(colorintensity);
	hue=le.fit_transform(hue);
	dilutedwines=le.fit_transform(dilutedwines);
	proline=le.fit_transform(proline);

	target=le.fit_transform(target);

	#print(alcohol);


	features=zip(alcohol,malic_acid,ash,alcalinityofash,magnesium,totalphenols,flavanoids,nonflavanoidphenols,proanthocyanins,
		colorintensity,hue,dilutedwines,proline);

	features=list(features);

	train_data,test_data,train_target,test_target=train_test_split(features,target,test_size=0.3);

	clf=KNeighborsClassifier(n_neighbors=2);

	clf.fit(train_data,train_target);

	predicted=clf.predict(test_data);

	return predicted,test_target;

def Accuracy(predicted,test_target):

	return accuracy_score(test_target,predicted);

def main():
	predicted,test_target=WinePredictor(sys.argv[1]);

	accuracy=Accuracy(predicted,test_target);

	print("Accuracy is ",accuracy*100);

if __name__=="__main__":
	main();