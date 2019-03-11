import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

class fusion_vote_maj_binary():
    """Cette classe implémente un vote majoritaire entre différents classifieurs;
    implémente un interface scikitlearn
    """
    def __init__(self, list_classifieur, is_train = False):
        """
        parametres:
        list_class : Liste des classifieurs qui compose la fusion
        is_train : booléen qui indique si les classifieurs de la liste sont déjà entraîné"""
        self.list_class = list_classifieur
        self.n = len(list_classifieur)
        self.is_train = is_train
        
    def fit(self, X, y):
        """entraîne les différents classifieurs de la fusion"""
        if self.is_train:
            pass
        for i in range(self.n):
            self.list_class[i].fit(X, y)
        self.is_train = True
            
    def predict(self, X):
        """ prédit la classe des données en entrées selon un vote majoritaire"""
        assert self.is_train, "is not trained"
        prediction = np.zeros(X.shape[0])
        # On additionne les vote
        for clf in self.list_class:
            prediction += clf.predict(X)
        return ((prediction/self.n) > .5).astype(int) # On test si la majorité absolue est atteinte
    def score(self, X, y):
        """score de précision"""
        return (self.predict(X)==y).mean()
        

def generer_plis(n,n_plis):
    """
    paramètres :
    n : nombre de données au total
    n_plis : nombre de plis
    return:
    array : donne à chaque élément le plis auquel il appartient"""
    plis = np.zeros(n)
    for i in range(n_plis):
        plis[n//n_plis*i:n//n_plis*(i+1)] = i
    return plis

def calcul_score(classifieurs, plis, Y, c, with_fusion = False, fusion_class = None):
    """ Calcul la précision de différents classifieurs selon le principe de la validation croisée
    Paramètres :
    - classifieurs : liste des classifieurs
    - plis : mapping entre les données et les plis
    - Y : features
    - c : labels
    - with_fusion : Indique si on réalise une fusion des classifieurs
    - fusion_class: indique le typre de fusion réalisé
    Return:
    -Matrice des scores (plis en lignes, classifieurs en colonnes)"""
    n_classifieurs = len(classifieurs)
    n_plis = int(plis[-1] + 1)
    scores = np.zeros((n_plis, n_classifieurs + with_fusion))
    for i in range(n_plis):
        for j in range(n_classifieurs):
            classifieurs[j].fit(Y[plis!=i],c[plis!=i])
            scores[i,j] = classifieurs[j].score(Y[plis==i],c[plis==i])
        if with_fusion:
            fusion = fusion_class(classifieurs, is_train=True)
            scores[i,n_classifieurs] = fusion.score(Y[plis!=i],c[plis!=i])
        
    return scores

def afficher(clf,X):
    """ Réalise l'affichage des frontière de décision d'un classifieur"""
    h = .02
    x1_min, x1_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    x2_min, x2_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    x11, x22 = np.meshgrid(np.arange(x1_min, x1_max, h),np.arange(x2_min, x2_max, h))
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[x11.ravel(), x22.ravel()])
    else:
        Z = clf.predict(np.c_[x11.ravel(), x22.ravel()]).reshape(x11.shape)
    plt.contourf(x11, x22, Z, cmap=plt.cm.bwr, alpha=.8)

def afficher_crossval(classifieurs, Y, c, plis, with_fusion = False, fusion_class = None , fusion_name = "Fusion"):
    """ Réalise l'affichage des frontières de décision de plusieurs classifieurs"""
    n_classifieurs = len(classifieurs)
    n_plis = int(plis[-1] + 1)
    plt.figure(figsize=(20,20))
    # parcourt des plis
    for i in range(n_plis):
        # On affiche les données d'entraînement
        plt.subplot(n_plis, n_classifieurs +1 +with_fusion,np.ravel_multi_index((i,0),(n_plis,n_classifieurs +1 +with_fusion)) + 1)
        plt.scatter(Y[:,0][plis==i],Y[:,1][plis==i],c=c[plis==i])
        # parcourt des classifieurs
        for j in range(n_classifieurs):
            classifieurs[j].fit(Y[plis!=i],c[plis!=i])
            plt.subplot(n_plis, n_classifieurs +1+with_fusion,np.ravel_multi_index((i,j+1),(n_plis,n_classifieurs +1+with_fusion)) + 1)
            afficher(classifieurs[j], Y)
            if i == 0:
                plt.title(str(classifieurs[j]).split('(')[0])
        # frontière de décision de la fusion
        if with_fusion:
            fusion = fusion_class(list(classifieurs), is_train=True)
            plt.subplot(n_plis, n_classifieurs +1 +with_fusion,np.ravel_multi_index((i,n_classifieurs +1),(n_plis,n_classifieurs +1 +with_fusion)) + 1)
            afficher(fusion, Y)
            if i == 0:
                    plt.title(fusion_name)
