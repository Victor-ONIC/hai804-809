import math

class Modele :
    type = "null"
    C = []
    w = []
    sw = 1/16
    lw = 1/4
    
    def __init__(self, data) :
        if isinstance(data, str) :      # ex : Modele("i") avec "i" comme type de modèle 
            self.type = data
        elif isinstance(data, list) :
            if (len(data)>0 and isinstance(data[0], str)) :
                self.type = data[0]
                if (len(data)>2 and (isinstance(data[1], int) or isinstance(data[1], float)) and (isinstance(data[2], int) or isinstance(data[2], float))) :    # ex : Modele(["i", 0.1, 0.2]) avec "i" comme type,
                    self.sw = data[1]                                                                                                                           # 0.1 comme petit angle et 0.2 comme grand angle
                    self.lw = data[2]
            elif (len(data)>0) :
                for i in data :
                    if (isinstance(i, int) or isinstance(i, float)) :       # ex : Modele([0, 0.2, 0.4]) où chaque valeur est le centre d'un secteur
                        self.C.append(i - int(i))
                if len(data)>1 and isinstance(data[0], list) and isinstance(data[1], list) :        # ex : Modele([[0, 0.2], [0.1, 0.4]]) où la première liste correspond aux centres des secteurs
                    for i in data[0] :                                                              # et la deuxième liste correspond à leur largeur
                        if (isinstance(i, int) or isinstance(i, float)) :                           # !!!ATENTION!!! C'est bien une liste de 2 sous listes
                            self.C.append(i - int(i))
                    for i in data[1] :
                        if (isinstance(i, int) or isinstance(i, float)) :
                            self.w.append(i)
        self.compileCw()
    
    def compileCw(self) :
        match self.type :
            case "i" :
                self.C = [1/4]
                self.w = [self.sw]
            case "V" :
                self.C = [1/4]
                self.w = [self.lw]
            case "L" :
                self.C = [0., 1/4]
                self.w = [self.lw, self.sw]
            case "I" :
                self.C = [1/4, 3/4]
                self.w = [self.sw, self.sw]
            case "T" :
                self.C = [0.]
                self.w = [1/2]
            case "Y" :
                self.C = [1/4, 3/4]
                self.w = [self.lw, self.sw]
            case "X" :
                self.C = [1/4, 3/4]
                self.w = [self.lw, self.lw]
            case "N" :
                self.C = []
                self.w = []
            case "null" :
                for i in range(len(self.C) - len(self.w)) :
                    self.w.append(self.sw)
        
    def rotate(self, alpha) :                                       # fait pivoter l'ensemble des secteurs suivant l'angle
        for i in range(len(self.C)) :                               # entre 0 et 1 dans le sens anti-horaire (ou trigonométrique)
            self.C[i] = Modele.congru(self.C[i]+alpha)
    def radRotate(self, alpha) :
        for i in range(len(self.C)) :
            self.C[i] = Modele.congru(self.C[i]+alpha/2/math.pi)
    def degRotate(self, alpha) :
        for i in range(len(self.C)) :
            self.C[i] = Modele.congru(self.C[i]+alpha/360)
    
    def bord(self, n=-1) :      # bord(n) retourne la liste des deux bords de la nième section
        if n == -1 :            # bord() retourne la liste de toutes les listes des deux bords de chaque section
            return [[Modele.congru(self.C[i] + self.w[i]/2), Modele.congru(self.C[i] - self.w[i]/2)] for i in range(len(self.C))]
        elif n < len(self.C) :
            return [Modele.congru(self.C[n] + self.w[n]/2), Modele.congru(self.C[n] - self.w[n]/2)]
    
    def distance_secteur(self, h, c) :    # h est la valeur de la couleur, c est l'indice dans la liste des secteurs
        if Modele.distance_congru(h, self.C[c]) <= self.w[c]/2 :
            return -1               
        else :
            b = self.bord(c)        # cas où on est hors du secteur (renvoie 0 si on est sur un bord)
            return min(Modele.distance_congru(h, b[0]), Modele.distance_congru(h, b[1]))
            

    def congru(n) :             # Permet de faire la congruence
        return n-int(n)         # si on obtient un angle supérieur à 1 par exempl
    def radCongru(n) :          # ça renvoie l'angle équivalent entre 0 et 1
        res = n/2/math.pi       # plus une version avec des radians et avec des degrés au cas où
        res -= int(res)
        return 2*math.pi*res
    def degCongru(n) :
        res = n/360
        res -= int(res)
        return 360*res
    
    def distance_congru(a, b) :     # Retourne la distance congru entre deux angles
        d = abs(a-b)                # ex : entre 0 et 0.9, la distance et 0.1 car 0 est congru à 1
        return min(d, 1-d)          # il trouve la distance la plus courte entre les deux sens de rotation possible
    
    def get_liste_modeles():
        return ["i", "V", "L", "I", "T", "Y", "X"] # J'ai enlevé "N" puisque c'est un modèle vide