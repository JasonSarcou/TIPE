### diament aztèque
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from copy import deepcopy


#construction du damier à l'ordre n
def trace_damier(n):
    assert n%2 == 0, "Veuillez entrer une nombre pair"    # les diamants aztèques sont construits à partir de damiers d'ordre pair
    noir = [0,0,0]
    blanc = [255,255,255]
    L1 = [noir, blanc]
    for i in range(n//2-1):                               # on construit d'abord la première ligne du tableau
        L1 += [noir, blanc]                               # il n'y a qu'un motif qui se répète sur toutes la ligne
    L2 = L1[::-1]                                         # la deuxième ligne est la première retournée
    L3 = np.concatenate([np.array([L1]), np.array([L2])]) # on les concatène pour obtenir les deux premières lignes de la matrice
    L4 = L3.copy()
    for j in range(n//2-1):                               # le tableau est constitué des deux première lignes répétée n//2 fois
        L3 = np.concatenate([L3, L4])                     # on cacatène les deux premières lignes le bon nombre de fois pour obtenir le tableau
    """extent = 0,n,0,n
    plt.figure()
    plt.axis()
    plt.imshow(L3, cmap = 'gray', extent = extent)
    plt.show()"""
    return L3




#obtention du diamant aztèque
def trace_diamant(n):
    B = trace_damier(n)
    noir = [0,0,0]
    blanc = [255,255,255]
    for k in range(n//2-1):                               # on parcours le tableau ligne par ligne
        Lk1 = B[k]                                        # le tableau du diamant est symétrique par rapport à la ligne du milieu
        Lk2 = Lk1[::-1]
        p = n-(n//2)-k-1
        for i in range(p):                                # on parcourt les lignes jusqu'au début du diamant
            Lk1[i], Lk2[i] = blanc, blanc                 # toutes les cases avant le début du diamant sont blanches
        B[k] = Lk1
        B[n - k - 1] = Lk2
    """extent = 0,n,0,n
    plt.imshow(B , cmap = 'gray', extent = extent)
    plt.legend()
    plt.show()"""
    return B

### pavage du diamant aztèque
# recherche des cellules actives d'un diamant d'ordre N/2
def cellules_actives(N):
    """
    Paramètres
    ----------
    N : double de l'odre du diamant dont on recherche les cellules actives

    Résultat
    --------
    renvoie toutes les cellules actives du diamant d'ordre N/2
    """

    noir = [0,0,0]
    blanc = [255,255,255]
    B = trace_diamant(N)
    liste_des_cellules = []
    ordre = N//2
    s_pair = [ordre//2,(ordre//2)-1]              # compteur de position pour parcourir le diamant en diagonale en partant de celle du milieu et vers le bas
    s_impair = [ordre//2,ordre//2]
    S_pair = [(ordre//2)-1,ordre//2]              # compteur de position pour parcourir le diamant d'ordre pair en diagonale vers le haut
    S_impair = [(ordre//2)-1,(ordre//2)+1]        # compteur de position pour parcourir le diamant d'ordre impair en diagonale vers le haut
    if ordre%2 == 0:
        plt.plot(ordre//2-1, N-ordre//2,marker='o',color = 'blue', label = "point de départ de du parcours vers le bas")
        plt.plot(ordre//2, N-(ordre//2-1),marker='o',color = 'red', label = "point de départ de du parcours vers le haut")
        while s_pair[1] >= 0:                     # jusqu'à ce qu'on arrive au bord gauche
            for k in range(ordre):                # on parcourt le diamant depuis la diagonale juste en dessous de celle du milieu en descendant
                if all(B[s_pair[0]+k,s_pair[1]+k]) == all(blanc):   # si l'on rencontre une case noire on l'ajoute de la liste des cellules actives
                    liste_des_cellules.append((s_pair[0]+k,s_pair[1]+k))
            s_pair[0] += 1
            s_pair[1] -= 1
        while S_pair[0]>= 0:                      # on parcourt le diamant depuis la diagonale juste au dessus de celle du milieu en montant
            for k in range(ordre):
                if all(B[S_pair[0]+k,S_pair[1]+k]) == all(blanc):
                    liste_des_cellules.append((S_pair[0]+k,S_pair[1]+k))
            S_pair[1] += 1
            S_pair[0] -= 1
    if ordre%2 == 1:                              # même principe que pour l'odre pair hormis que les cellules actives sont celles qui ont une case noire en haut à gauche
        plt.plot(ordre//2,N-ordre//2,marker='o',color = 'blue', label = "point de départ de du parcours vers le bas")
        plt.plot(ordre//2+1,N-(ordre//2-1),marker='o',color = 'red', label = "point de départ de du parcours vers le haut")
        while s_impair[1] >= 0:
            for k in range(ordre):
                if all(B[s_impair[0]+k,s_impair[1]+k]) == all(noir):
                    liste_des_cellules.append((s_impair[0]+k,s_impair[1]+k))
            s_impair[0] += 1
            s_impair[1] -= 1
        while S_impair[0] >= 0:
            for k in range(ordre):
                if all(B[S_impair[0]+k,S_impair[1]+k]) == all(noir):
                    liste_des_cellules.append((S_impair[0]+k,S_impair[1]+k))
            S_impair[0] -= 1
            S_impair[1] += 1
    assert len(liste_des_cellules) == ordre**2
    return liste_des_cellules



### pavage
# obtention du diamant d'ordre 1
rouge = [255,0,0]
vert = [0,255,0]
bleu = [0,0,255]
jaune = [255,255,0]
noir = [0,0,0]
blanc = [255,255,255]
carre_1 = np.array([[bleu, bleu],[vert, vert]])
carre_2 = np.array([[vert, vert],[bleu, bleu]])
carre_3 = np.array([[jaune, rouge],[jaune, rouge]])
carre_4 = np.array([[rouge, jaune],[rouge, jaune]])
carre = [carre_1, carre_4]                                  # le diamant d'ordre 1 a une case en haut à gauche
diamant_ordre1 = rd.choice(carre)
extent = 0,2,0,2
plt.imshow(diamant_ordre1, extent = extent)
plt.axis('off')
plt.show()


def egalite(L,M):
    """
    Paramètres
    ----------
    L, M : deux array de même taille

    Résultat
    --------
    renvoie True si les arrray sont égaux, False sinon
    """
    return np.array_equal(L,M)

def inversement(tableau1,tableau2):
    """
    Paramètres
    ----------
    tableau1, tableau2 : tableaux RGB de taille 2x2x3

    Effet de bord
    -------------
    Décale le domino de l'autre côté de la cellule

    """
    if egalite(tableau1[0], [bleu, bleu]):
        tableau2[1] = [vert , vert]
    elif egalite(tableau1[1], [bleu, bleu]):
        tableau2[0] = [vert , vert]
    elif egalite(tableau1[0], [vert, vert]):
        tableau2[1] = [bleu, bleu]
    elif egalite(tableau1[1], [vert, vert]):
        tableau2[0] = [bleu, bleu]
    elif egalite(tableau1[0][0] , rouge) and egalite(tableau1[1][0], rouge):
        tableau2[0][1] = jaune
        tableau2[1][1] = jaune
    elif egalite(tableau1[0][1], rouge) and egalite(tableau1[1][1], rouge):
        tableau2[0][0] = jaune
        tableau2[1][0] = jaune
    elif egalite(tableau1[0][0], jaune) and egalite(tableau1[1][0], jaune):
        tableau2[0][1] = rouge
        tableau2[1][1] = rouge
    elif egalite(tableau1[0][1], jaune) and egalite(tableau1[1][1], jaune):
        tableau2[0][0] = rouge
        tableau2[1][0] = rouge
    return tableau2

def configuration_int(cellule, support):
    """
    Paramètres
    ----------
    cellule : cellue active que l'on étudie
    support : diamant aztèque pavé

    Résultat
    --------
    renvoie True si la cellule active contient exactement un domino entier
    """

    config = False
    if egalite(cellule[0], [bleu, bleu]) and not configuration_plein(cellule, support) :    # ATTENTION : il faut bien vérifier que la cellule ne contient qu'un domino entier
        config = True
    elif egalite(cellule[1], [bleu, bleu]) and not configuration_plein(cellule, support):
        config = True
    elif egalite(cellule[0], [vert, vert]) and not configuration_plein(cellule, support):
        config = True
    elif egalite(cellule[1], [vert, vert]) and not configuration_plein(cellule, support):
        config = True
    elif egalite(cellule[0][0], rouge) and egalite(cellule[1][0], rouge) and not configuration_plein(cellule, support):
        config = True
    elif egalite(cellule[0][1], rouge) and egalite(cellule[1][1], rouge) and not configuration_plein(cellule, support):
        config = True
    elif egalite(cellule[0][0], jaune) and egalite(cellule[1][0], jaune) and not configuration_plein(cellule, support):
        config = True
    elif egalite(cellule[0][1], jaune) and egalite(cellule[1][1], jaune) and not configuration_plein(cellule, support):
        config = True
    return config

def configuration_plein(cellule, support):
    """
    Paramètres
    ----------
    cellule : cellue active que l'on étudie
    support : diamant aztèque pavé

    Résultat
    --------
    renvoie True si la cellule active est recouverte par deux dominos entiers
    """
    config = False
    if np.array_equal(cellule, carre_1) and egalite(support[0][0], noir):               # ATTENTION : il faut s'assurer que les dominos soient bien entiers, cf exemple
        config = True
    elif np.array_equal(cellule, carre_2) and egalite(support[0][0], blanc):
        config = True
    elif np.array_equal(cellule, carre_3) and egalite(support[0][0], blanc):
        config = True
    elif np.array_equal(cellule, carre_4) and egalite(support[0][0], noir):
        config = True
    return config



# passage du pavage de l'ordre  1 à l'ordre 2
def passage():
    """
    Résultat
    --------
    pavage du diamant d'ordre 2
    """
    diamant_intermediaire = trace_diamant(4)
    n = 2
    for i in range(n):
        for j in range(n):
            diamant_intermediaire[i+1,j+1] = diamant_ordre1[i,j]
    diamant_final = deepcopy(diamant_intermediaire)                           #  on créé une nouveau tableau pour conserver ces couleurs lors de la construction du diamant d’ordre suivant, dont les nouvelles couleurs vont dépendre de celles du tableau diamant_intermediaire, et il ne doit pas y avoir écrasement des anciennes couleurs par les nouvelles tant que le nouveau pavage n’est pas complètement construit.
    coordonnee_cellule = cellules_actives(4)                                  # on récupère les cellules actives du diamant qu'on veut paver
    for coordonnee in coordonnee_cellule:
        i, j = coordonnee[0], coordonnee[1]
        L = diamant_intermediaire[i:i+2, j:j+2].copy()
        vide = True
        if configuration_int(L, configuration_int[k]):
            L = inversement(L, diamant_final[i:i+2, j:j+2])
            diamant_final[i:i+2, j:j+2] = L
            vide = False
        if vide:
            for conf in configuration_plein:
                if np.array_equal(diamant_intermediaire[i:i+2, j:j+2], conf):
                    vide = False
            if vide:
                diamant_final[i:i+2, j:j+2] = rd.choice([carre_2, carre_3])
    extent = 0,4,0,4
    plt.imshow(diamant_final, extent = extent)
    plt.grid()
    plt.show()




def pavage(N):
    """
    Résultat
    --------
    Pavage du diamant d'ordre N/2
    """
    if N == 2:
        return diamant_ordre1
    else:
        diamant_precedent = pavage(N-2)
    ordre = N/2
    diamant_intermediaire = trace_diamant(N)
    support = trace_diamant(N)
    n = np.shape(diamant_precedent)[0]
    for i in range(n):                                                       # la première étape est d'inscrire le diamant pavé de l'ordre précédent dans le diamant qu'on veut paver
        for j in range(n):
            if not(egalite(diamant_precedent[i,j],blanc)) and not(egalite(diamant_precedent[i,j],noir)):
                diamant_intermediaire[i+1,j+1] = diamant_precedent[i,j]
    diamant_final = deepcopy(diamant_intermediaire)                          #  on créé une nouveau tableau pour conserver ces couleurs lors de la construction du diamant d’ordre suivant, dont les nouvelles couleurs vont dépendre de celles du tableau diamant_intermediaire, et il ne doit pas y avoir écrasement des anciennes couleurs par les nouvelles tant que le nouveau pavage n’est pas complètement construit.

    coordonnee_cellule = cellules_actives(N)                                 # on récupère les cellules actives du diamant qu'on veut paver
    for coordonnee in coordonnee_cellule:                                    # on parcours toutes les cullules actives
        i, j = coordonnee[0], coordonnee[1]
        vide = True                                                          # booléen qui indique si la cellules actives ne contient aucun domino
        L = diamant_intermediaire[i:i+2, j:j+2]                              # pour alléger les notations
        if configuration_int(L, support):                                    # si la cellule active contient exactement un domino : on le décale
                diamant_final[i:i+2, j:j+2] = inversement(L, diamant_final[i:i+2, j:j+2])
                vide = False                                                 # il contient alors un domino complet
        if configuration_plein(L, support[i:i+2, j:j+2]):                    # on test si la cellule active contient exactement deux dominos, si c'est le cas on ne fait rien : les dominos anciens vont être écrasés par de nouveaux provenant des pavés voisins

            vide = False
        if vide:                                                             # si la cellule active est vide on la recouvre avec deux dominos en tirant à pile ou face leur orientation
            if (ordre)%2 == 1:
                diamant_final[i:i+2, j:j+2] = np.array(rd.choice([carre_1, carre_4]))
            else:
                diamant_final[i:i+2, j:j+2] = np.array(rd.choice([carre_2, carre_3]))
        """extent = 0,N,0,N
        plt.imshow(diamant_final, extent = extent)
        plt.grid()
        plt.show()"""
    extent = 0,N,0,N
    plt.imshow(diamant_final, extent = extent)
    plt.grid()
    plt.show()
    return diamant_final