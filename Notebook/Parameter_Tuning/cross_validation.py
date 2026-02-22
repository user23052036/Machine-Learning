from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# creating a template to compare models on the basis of stratified K-Fold cross Validation

def my_CV(models,X,Y):
    for model in models:

        cv_score = cross_val_score(model,X,Y,cv=5)

        mean_accuracy = sum(cv_score)/len(cv_score)
        mean_accuracy *= 100
        mean_accuracy = round(mean_accuracy,2)

        print('Cross validation accuracy for = ',model,'=',cv_score)
        print('Accuracy percentage for the model = ',model,'=',mean_accuracy)
        print('-----------------------------------------------------------------------')