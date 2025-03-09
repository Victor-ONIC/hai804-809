## Harmonization.py
# Color Harmonization based on Cohen-Or et al. 2006

import math

#J'ai pas fait une classe puisque ca m'avait pas l'air nécessaire

HARMONY_TEMPLATES = {
    "I": [(0, math.pi/6)],
    "V": [(0, math.pi/3), (math.pi, 4*math.pi/3)],
    "L": [(0, math.pi/3), (2*math.pi/3, math.pi)],
    "T": [(0, math.pi)],
    "Y": [(0, math.pi/6), (2*math.pi/3, math.pi), (4*math.pi/3, 3*math.pi/2)],
    "X": [(0, math.pi/3), (2*math.pi/3, math.pi), (4*math.pi/3, 5*math.pi/3)],
}

def rotate_template(template, alpha):
    return [((start + alpha) % (2*math.pi), (end + alpha) % (2*math.pi)) for (start, end) in template]

def to_degrees(alpha):
    return alpha * 180 / math.pi

def to_radians(alpha):
    return alpha * math.pi / 180


#Input image (obv) and both template and alpha 
def harmonize(image, template , alpha):
    #TODO
    pass

#Input image and template (auto calculate angle)
def harmonize(image, template):
    #TODO
    pass

#Input image (auto calculate template and angle)
def harmonize(image):
    #TODO
    pass


#tests
template = HARMONY_TEMPLATES["I"]
print(rotate_template(template, math.pi/6))