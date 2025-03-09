## Harmonization.py
# Color Harmonization based on Cohen-Or et al. 2006

import math
from PIL import Image
from Histogramme import Histogramme
from colorsys import rgb_to_hsv as r2h


#J'ai pas fait une classe puisque ca m'avait pas l'air nécessaire

HARMONY_TEMPLATES = {
    "i": [(0, math.pi/6)],
    "I": [(0, math.pi/6), (math.pi, 7*math.pi/6)],
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

def assign_closest_edge(hue, template):
    closest_edge = None
    for (start, end) in template:
        distance = min(abs(hue - start), abs(hue - end))
        if closest_edge is None or distance < closest_edge[1]:
            closest_edge = ((start, end), distance)
    return closest_edge[0]

def get_hsv(pixel):
    r, g, b = [c/255.0 for c in pixel]
    return r2h(r, g, b)

def get_hue(pixel):
    return get_hsv(pixel)[0]

def get_saturation(pixel):
    return get_hsv(pixel)[1]

#F(X,(m,alpha)) = ||H(X) - E_tm(alpha)|| * S(X) in the paper
def harmony_by_template(image , template):
    sum_harmony_value = 0
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            hue = get_hue(image.getpixel((i, j)))
            E = assign_closest_edge(hue, template)
            saturation = get_saturation(image.getpixel((i, j)))
            sum_harmony_value += abs(hue - E) * saturation
    return sum_harmony_value

#M(X , Tm) = (m , alpha0) avec alpha0 = argmin alpha F(X , (m,alpha)) in the paper
def best_angle_radians(image, template):
    pass

#B(X) = (m0 , alpha0) avec m0 = argmin m M(X , Tm) in the paper
def best_template(image):
    pass

#H'(X) = C(p) + (w/2) * (1 - Gaussian() in the paper

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

nameImage = "perso.ppm"
im = Image.open(nameImage)
