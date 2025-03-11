from colorsys import rgb_to_hsv as r2h

class Histogramme :
    def __init__(self, nbValue) :
        self.nbValue = nbValue
        self.data = [0 for i in range(nbValue)]
        self.trueData = []
        
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