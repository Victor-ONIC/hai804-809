from colorsys import rgb_to_hsv as r2h

class Histogramme :
    """
    Créez un histogramme vide avec le constructeur et une "nbValue".
    Ensuite, chargez une image avec load().
    Les données de l'histogramme, arrondies sur les valeurs permises, sont stokées dans "data".
    """
    def __init__(self, nbValue) :
        self.nbValue = nbValue                      # "nbValue" correspond aux nombre de valeurs différentes que peut prendre les pixels de l'image qui sera donnée à l'histogramme,
                                                    # les autres valeurs seront arrondies à la plus proche.
        self.data = [0 for i in range(nbValue)]     # "data" stocke à la position "n", le nombre de pixels ayant sa valeur arrondie à "n/nbValue", les valeurs étant supposées aller de 0 à 1.
        self.trueData = []                          # "trueData" stock les vraies valeurs des pixels, pour recalculer les bon arrondies si l'on veut changer de "nbValue".
        
    def load(self, image) :
        nbx = image.size[0]
        nby = image.size[1]
        data = image.load()
        self.trueData = [0 for i in range(nbx*nby)]
        for i in range(nbx) :
            for j in range(nby) :
                r, g, b = [c/255.0 for c in data[i, j]]
                h, s, v = r2h(r, g, b)
                self.trueData[j*nbx+i] = h
        sorted(self.trueData)
        self.changeNbValue(self.nbValue)
    
    def changeNbValue(self, nbValue) :
        self.nbValue = nbValue
        self.data = [0 for i in range(nbValue)]
        for h in self.trueData :
            index = round(h*self.nbValue) % (self.nbValue)
            self.data[index] += 1