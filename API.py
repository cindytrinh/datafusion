import numpy as np
import pandas as pd
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
    def __init__(self, list_classifieur, pondere = False, ponderations = None, is_train = False):
        """
        parametres:
        list_class : Liste des classifieurs qui compose la fusion
        is_train : booléen qui indique si les classifieurs de la liste sont déjà entraîné"""
        self.list_class = list_classifieur
        self.n = len(list_classifieur)
        self.is_train = is_train
        if not pondere:
            self.ponde = np.ones(self.n)
        else :
            self.ponde = ponderations
        
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
        
        prediction = np.sum(list(map((lambda pond,clf : pond*clf.predict(X)), self.ponde, self.list_class)), axis = 0)/np.sum(self.ponde)
        return (prediction > .5).astype(int) # On test si la majorité absolue est atteinte
    
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

def calcul_score(classifieurs, plis, Y, c, names, with_fusion = False, fusion_class = None):
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
            scores[i,n_classifieurs] = fusion.score(Y[plis==i],c[plis==i])
     
        
    return pd.DataFrame(scores,columns=names,index=[i+1 for i in range(n_plis)])

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
    """ Réalise l'affichage des frontières de décision de plusieurs classifieurs 
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
    plt.figure(figsize=(20,20))
    # parcourt des plis
    for i in range(n_plis):
        # On affiche les données d'entraînement
        plt.subplot(n_plis, n_classifieurs +1 +with_fusion,np.ravel_multi_index((i,0),(n_plis,n_classifieurs +1 +with_fusion)) + 1)
        plt.scatter(Y[:,0][plis==i],Y[:,1][plis==i],c=c[plis==i],cmap=plt.cm.bwr)
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
                    
def score_deux_etages(classifieurs, Y, c, n_plis, names, is_pondere=True, weight_function = lambda x: x):
    """ Cette fonction calcul le score de précision des différents classifieurs 
    en réalisant une cross validation à deux étages pour calculer les poids associés aux classifieurs lors du vote pondéré
    paramètres :
    - classifieurs : liste des classifieurs
    - n_plis :nombres de plis dans la première et la deuxième cross validation
    - Y : features
    - c : labels
    - names : nom des classifieurs
    - is_pondere : indique si le vote est pondere
    - weight_function : fonction de modification des pondérations
    Return:
    -Matrice des scores (plis en lignes, classifieurs en colonnes)"""

    plis_niv1 = generer_plis(c.shape[0],n_plis)
    n_classifieurs = len(classifieurs)
    scores = np.zeros((n_plis, n_classifieurs + 1))
    for i in range(n_plis):
    ### Deuxième etage 
        plis_niv2 = generer_plis((plis_niv1!=i).sum(),n_plis)
        precision = np.zeros((n_plis, n_classifieurs))
        for j in range(n_plis):
            for k in range(n_classifieurs):
                clf = classifieurs[k]
                clf.fit(Y[plis_niv1!=i][plis_niv2!=j],c[plis_niv1!=i][plis_niv2!=j])
                precision[j,k] = clf.score(Y[plis_niv1!=i][plis_niv2==j],c[plis_niv1!=i][plis_niv2==j])
        wi = precision.mean(axis = 0)

        ### Premier Etage
        for k in range(n_classifieurs):
                classifieurs[k].fit(Y[plis_niv1!=i],c[plis_niv1!=i])
                scores[i,k] = classifieurs[k].score(Y[plis_niv1==i],c[plis_niv1==i])
        fusion = fusion_vote_maj_binary(classifieurs, pondere = is_pondere, ponderations = weight_function(wi), is_train=True)
        scores[i,n_classifieurs] = fusion.score(Y[plis_niv1==i],c[plis_niv1==i])

    return pd.DataFrame(scores,columns=names,index=[i+1 for i in range(n_plis)])



def affichage_deux_etages(classifieurs, Y, c, n_plis, 
                          fusion_name = "vote pondere", is_pondere=True, weight_function = lambda x: x):
    """ 
    Cette fonction affiche les frontières de décision des différents classifieurs en réalisant 
    une cross validation à deux étages pour calculer les poids associés aux classifieurs lors du vote pondéré
    paramètres :
    - classifieurs : liste des classifieurs
    - n_plis :nombres de plis dans la première et la deuxième cross validation
    - Y : features
    - c : labels
    - names : nom des classifieurs
    - is_pondere : indique si le vote est pondere
    - weight_function : fonction de modification des pondérations
    """

    n_classifieurs = len(classifieurs)
    n = c.shape[0]
    plis_niv1 =  generer_plis(n,n_plis)
    plt.figure(figsize=(20,20))
    for i in range(n_plis):
        plt.subplot(n_plis, n_classifieurs +1 +1,np.ravel_multi_index((i,0),(n_plis,n_classifieurs +1 +1)) + 1)
        plt.scatter(Y[:,0][plis_niv1==i],Y[:,1][plis_niv1==i],c=c[plis_niv1==i],cmap=plt.cm.bwr)
        ### Deuxième etage 
        plis_niv2 = generer_plis((plis_niv1!=i).sum(),n_plis)
        errors = np.zeros((n_plis, n_classifieurs))
        for j in range(n_plis):
            for k in range(n_classifieurs):
                clf = classifieurs[k]
                clf.fit(Y[plis_niv1!=i][plis_niv2!=j],c[plis_niv1!=i][plis_niv2!=j])
                error = clf.score(Y[plis_niv1!=i][plis_niv2==j],c[plis_niv1!=i][plis_niv2==j])
                errors[j, k] = error
        wi = errors.mean(axis = 0)

        ### Premier Etage
        for k in range(n_classifieurs):
                clf = classifieurs[k]
                clf.fit(Y[plis_niv1!=i],c[plis_niv1!=i])
        fusion = fusion_vote_maj_binary(classifieurs, pondere = False, is_train=True)
        # parcourt des plis
            # On affiche les données d'entraînement

        # parcourt des classifieurs
        for j in range(n_classifieurs):
            classifieurs[j].fit(Y[plis_niv1!=i],c[plis_niv1!=i])
            plt.subplot(n_plis, n_classifieurs +1+1,np.ravel_multi_index((i,j+1),(n_plis,n_classifieurs +1+1)) + 1)
            afficher(classifieurs[j], Y)
            if i == 0:
                plt.title(str(classifieurs[j]).split('(')[0])
        # frontière de décision de la fusion
        fusion = fusion_vote_maj_binary(classifieurs, pondere = is_pondere, 
                                        ponderations = weight_function(wi), is_train=True)
        plt.subplot(n_plis, n_classifieurs +1 +1,np.ravel_multi_index((i,n_classifieurs +1),
                                                                      (n_plis,n_classifieurs +1 +1)) + 1)
        afficher(fusion, Y)
        if i == 0:
                plt.title(fusion_name)
