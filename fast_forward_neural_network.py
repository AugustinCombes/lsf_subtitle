import csv
import numpy as np
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from enhanced_vocab_making.py import infos

def main():
	labels, number_imgs = infos()

	data = np.zeros((len(labels), number_imgs, 42*3))

	count_labels, count_imgs = 0,0

	with open('mp_data.csv', newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		for row in reader :
			if count_imgs==25:
				count_labels+=1
				count_imgs=0
			data[count_labels,count_imgs]=np.array(row)
			count_imgs+=1

	data = data.reshape(len(labels)*number_imgs,-1)

	X = [list(data[i]) for i in range(len(data))]
	y = []
	for it in range(len(labels)):
		y+=[it]*(number_imgs)


	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1) #we can test different solvers, differentes sizes

	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=69)

	scaler = preprocessing.StandardScaler().fit(X_train)
	X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

	clf.fit(X_train, y_train)

	y_pred=clf.predict(X_test)

def predict(mp_array):
	return clf.predict([mp_array])

def preprocess(mp_array):
	return scaler.transform([mp_array])

#print(accuracy_score(y_test,y_pred))

if __name__ == "__main__":
	main()