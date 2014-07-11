from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt, int64, float64, asarray
import csv

def main():
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open('Data/train.csv','r'), delimiter=',', dtype= float64, names=True)
    colnames = dataset.dtype.names[1:]
    dataset = dataset.view(float64).reshape(dataset.shape + (-1,))
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    test = genfromtxt(open('Data/test.csv','r'), delimiter=',', dtype='f8')[1:]
    
    #create and train the random forest
    #multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(train, target)
    predicted_probs = [x[1] for x in rf.predict_proba(test)]
    moleculeID = list(xrange(1,len(predicted_probs)+1))
    table = zip(asarray(moleculeID, dtype = int64),asarray(predicted_probs,dtype = float64))
    with open('Data/submission.csv','wb') as fout:
        writer = csv.writer(fout)
        writer.writerow(("MoleculeID","PredictedProbability"))
        writer.writerows(table)

if __name__=="__main__":
    main()