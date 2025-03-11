class Modele :
    C = []
    w = []
    sr = 1/16
    lr = 1/4
    def __init__(self, data) :
        if isinstance(data, str) :      # ex : Modele("i") avec "i" comme type de modèle 
            self.type = data
        elif isinstance(data, list) :
            if (len(data)>0 and isinstance(data[0], str)) :
                self.type = data[0]
                if (len(data)>2 and (isinstance(data[1], int) or isinstance(data[1], float)) and (isinstance(data[2], int) or isinstance(data[2], float))) :    # ex : Modele(["i", 0.1, 0.2]) avec "i" comme type,
                    self.sr = data[1]                                                                                                                           # 0.1 comme petit angle et 0.2 comme grand angle
                    self.lr = data[2]
            elif (len(data)>0) :
                for i in data :
                    if (isinstance(data[i], int) or isinstance(data[i], float)) :       # ex : Modele([0, 0.2, 0.4]) où chaque valeur est le centre d'un secteur
                        self.type = "null"
                        self.C.append(data[i] - int(data[i]))
                if len(data)>1 and isinstance(data[0], list) and isinstance(data[1], list) :        # ex : Modele([0, 0.2], [0.1, 0.4]) où la première liste correspond aux centres des secteurs
                    self.type = "null"
                    for i in data[0] :                                                              # et la deuxième liste correspond à leur largeur
                        if (isinstance(data[0][i], int) or isinstance(data[0][i], float)) :
                            self.C.append(data[0][i] - int(data[0][i]))
                    for i in data[1] :
                        if (isinstance(data[1][i], int) or isinstance(data[1][i], float)) :
                            self.w.append(data[1][i] - int(data[1][i]))