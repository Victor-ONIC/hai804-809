## Harmonization.py
# Color Harmonization based on Cohen-Or et al. 2006

import math
from PIL import Image
from utils.Histogramme import Histogramme
from colorsys import rgb_to_hsv as r2h
from utils.Modele import Modele

def to_degrees(alpha):
    return alpha * 180 / math.pi

def to_radians(alpha):
    return alpha * math.pi / 180

#returns the closest edge to the hue, the border and the center of the sector
def assign_closest_edge_index(hue, model: Modele):
    min_distance = float('inf')
    closest_edge_index = None
    
    for i in range(len(model.C)):
        distance = model.distance_secteur(hue, i)
        if distance == -1: 
            return i
        if distance < min_distance:
            min_distance = distance
            closest_edge_index = i

    return closest_edge_index

#FIXME : just does not really work
def assign_closest_edge_index_optimized(hue, model: Modele, image , i , j):
    pixels = get_neighboring_pixels(image, i, j)
    omega = get_omega(pixels, image)

    min_energy = float('inf')
    closest_edge_index = None

    for i in range(len(model.C)):  # Iterate over the number of template edges
        central_hue = model.C[i]
        w = model.w[i]

        E1 = sum(abs(h - central_hue) * w for h in omega)

        E2 = 0
        for p in range(len(pixels)):
            for q in range(len(pixels)):
                if p != q:
                    _, s_p, _ = r2h(*[c / 255.0 for c in image.getpixel((pixels[p][0], pixels[p][1]))])
                    _, s_q, _ = r2h(*[c / 255.0 for c in image.getpixel((pixels[q][0], pixels[q][1]))])

                    Smax_pq = max(s_p, s_q)
                    hue_diff = abs(omega[p] - omega[q])
                    if hue_diff > 0: 
                        E2 += Smax_pq * (1 / hue_diff)

        # Total energy
        E = E1 + E2

        # Find the minimum energy edge
        if E < min_energy:
            min_energy = E
            closest_edge_index = i

    return closest_edge_index



def get_neighboring_pixels(image, i, j):
    pixels = []
    for x in range(i-1, i+1):
        for y in range(j-1, j+1):
            pixels.append([x, y])
    
    return pixels

def get_omega(pixels, image):
    omega = []
    #fill omega with the hue of the pixels
    for p in pixels:
        r, g, b = image.getpixel((p[0], p[1]))
        h, _, _ = r2h(r / 255.0, g / 255.0, b / 255.0)
        omega.append(h)
    
    return omega


#F(X,(m,alpha)) = sum ||H(X) - E_tm(alpha)|| * S(X) in the paper
def harmony_by_template(image, model: Modele):
    sum_harmony_value = 0
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            r, g, b = image.getpixel((i, j))
            hue , saturation , _ = r2h(r / 255.0, g / 255.0, b / 255.0)
            closest_edge_index = assign_closest_edge_index(hue, model)
            E = model.bord(closest_edge_index)
            sum_harmony_value += abs(hue - E[0]) * saturation  # Use closest edge

    return sum_harmony_value

#M(X , Tm) = (m , alpha0) avec alpha0 = argmin alpha F(X , (m,alpha)) in the paper
def best_angle_radians(image, template):
    pass

#B(X) = (m0 , alpha0) avec m0 = argmin m M(X , Tm) in the paper
def best_template(image):
    pass

#d = H - C (idk what c is yet but tkt)
#w = the width of a zone

def gaussian(x):
    return math.exp(-x**2)

#H'(X) = C(p) + (w/2) * (1 - Gaussian_w/2(||H(p)-C(p))) in the paper
def harmonize(image, imageOut, template: Modele, alpha):
    template.radRotate(alpha)
    
    image = image.convert("RGB")
    width, height = image.size
    pixels = image.load()
    pixels_out = imageOut.load()

    for i in range(width):
        for j in range(height):
            r, g, b = pixels[i, j]
            h, s, v = r2h(r / 255.0, g / 255.0, b / 255.0)
            closest_edge_index = assign_closest_edge_index(h, template)
            central_hue = template.C[closest_edge_index]
            width = template.w[closest_edge_index]

            d = Modele.distance_congru(h, central_hue)
            new_h = central_hue + (width / 2) * (1 - math.exp(- (d ** 2) / (width ** 2 / 2)))
            new_h = Modele.congru(new_h)

            new_r, new_g, new_b = [int(c * 255) for c in r2h(new_h, s, v)]
            pixels_out[i, j] = (new_r, new_g, new_b)

    return imageOut

#FIXME : optimized edge detection is fraudulent too
def harmonize_opti(image , imageOut , template : Modele , alpha):
        # Step 1: Rotate the template
    template.radRotate(alpha)
    
    # Step 2: Convert image to HSV
    image = image.convert("RGB")
    width, height = image.size
    pixels = image.load()
    pixels_out = imageOut.load()

    # Step 3: Apply harmonization
    for i in range(width):
        for j in range(height):
            r, g, b = pixels[i, j]
            h, s, v = r2h(r / 255.0, g / 255.0, b / 255.0)
            closest_edge_index = assign_closest_edge_index_optimized(h , template , image , i , j)
            central_hue = template.C[closest_edge_index]
            width = template.w[closest_edge_index]

            # Apply the formula for color adjustment
            d = Modele.distance_congru(h, central_hue)
            new_h = central_hue + (width / 2) * (1 - math.exp(- (d ** 2) / (width ** 2 / 2)))
            new_h = Modele.congru(new_h)  # Ensure hue stays in [0,1]

            # Convert back to RGB
            new_r, new_g, new_b = [int(c * 255) for c in r2h(new_h, s, v)]
            pixels_out[i, j] = (new_r, new_g, new_b)

    return imageOut
#--------------------------------------------------tests-----------------------------------------------------------
nameImage = "./src/images/perso.ppm"
im = Image.open(nameImage)
imOut = Image.new(im.mode, im.size)

harmonize(im, imOut , Modele("I"), 0)

imOut.save("./src/images/harmonized.ppm")
imOut.close()

imOut2 = Image.new(im.mode, im.size)

harmonize_opti(im, imOut2 , Modele("I"), 0)

imOut2.save("./src/images/harmonized_opti.ppm")
imOut2.close()
im.close()
