## Harmonization.py
# Color Harmonization based on Cohen-Or et al. 2006

import math
from PIL import Image
from utils.Histogramme import Histogramme
from colorsys import rgb_to_hsv as r2h
from colorsys import hsv_to_rgb as h2r
from utils.Modele import Modele
import time
import numpy as np

############################################################## Utils ################################################################################""

def to_degrees(alpha):
    '''
        Radians to degrees
    '''
    return alpha * 180 / math.pi


def to_radians(alpha):
    '''
        Degrees to radians
    '''
    return alpha * math.pi / 180

############################################################## Closest border n dem ###################################################################

def assign_closest_sector_index(hue, model: Modele):
    ''' 
        Return the center of the closest edge, us just calculating distance.
    '''
    
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


def assign_closest_edge(hue, model: Modele):
    '''
        Returns the closest border based on the closest edge
    '''

    sector_index = assign_closest_sector_index(hue, model)
    if sector_index is None:
        return hue 

    bord = model.bord(sector_index)

    dist1 = Modele.distance_congru(hue , bord[0])
    dist2 = Modele.distance_congru(hue , bord[1])
    if(dist1 < dist2):
        return bord[1]
    else : 
        return bord[0]

 
def assign_closest_edge_opti(hue, h_array, s_array, v_array, model: Modele, i, j):
    sector_index = assign_closest_sector_index_optimized(model, h_array, s_array, v_array, i, j)
    if sector_index is None:
        return hue

    bord = model.bord(sector_index)
    dist1 = Modele.distance_congru(hue, bord[0])
    dist2 = Modele.distance_congru(hue, bord[1])
    return bord[1] if dist1 < dist2 else bord[0]

def assign_closest_sector_index_optimized(model: Modele, h_array , s_array , v_array , i, j):
    pixels = get_neighboring_pixels(h_array, i, j) 
    omega = get_omega(pixels, h_array , s_array , v_array)

    min_energy = float('inf')
    closest_sector_index = None

    for idx in range(len(model.C)):  
        central_hue = model.C[idx]
        E1 = 0
        for hn, sn , _ in omega:
            hvp = project_hue(central_hue , model.w[idx]/2 , hn)
            E1 += Modele.distance_congru(hn, hvp) * sn
 
        if E1 < min_energy:  
            min_energy = E1
            closest_sector_index = idx

    return closest_sector_index



############################################################## Neigboring ###############################################################################

def get_neighboring_pixels(h_array, i, j):

    height, width = h_array.shape
    neighbors = []

    

    if j + 1 < width:
        neighbors.append((i, j + 1))
    if j - 1 >= 0:
        neighbors.append((i, j - 1))
    if i + 1 < height:
        neighbors.append((i + 1, j))
    if i - 1 >= 0:
        neighbors.append((i - 1, j))


    return neighbors


def get_omega(pixels, h_array , s_array , v_array):
    omega = []
    for (x, y) in pixels:
        h = h_array[x][y]
        s = s_array[x][y]
        v = v_array[x][y]
        omega.append((h, s, v))
    return omega

############################################################# Harmony utils ##############################################################################

def preprocess_hsv(image):
    width, height = image.size
    h_array = np.zeros((width, height))
    s_array = np.zeros((width, height))
    v_array = np.zeros((width, height))

    for i in range(width):
        for j in range(height):
            r, g, b = image.getpixel((i, j))
            h, s, v = r2h(r / 255.0, g / 255.0, b / 255.0)
            h_array[i, j] = h
            s_array[i, j] = s
            v_array[i, j] = v

    return h_array, s_array, v_array



def harmony_by_template(h_array, s_array, v_array, model: Modele):
    '''
        Calculate the harmony value of the image. This is a sort of arbitrary value I guess, but it's in the paper.
    '''
    
    height, width = h_array.shape
    sum_harmony_value = 0

    for i in range(height):
        for j in range(width):
            hue = h_array[i, j]
            saturation = s_array[i, j]
            closest_edge = assign_closest_edge_opti(hue, h_array, s_array, v_array, model, i, j)
            sum_harmony_value += Modele.distance_congru(hue, closest_edge) * saturation

    return sum_harmony_value


def best_angle_radians(h_array, s_array, v_array, template):
    '''
        Use binary search to find the best angle for the given template.
    '''
    
    def evaluate_harmony(alpha):
        alpha = Modele.radCongru(alpha)  # Ensure alpha is a valid angle
        template.radRotate(alpha)
        harmony_value = harmony_by_template(h_array, s_array, v_array, template)
        template.C = original_C[:]  # Reset template to original state
        return harmony_value

    original_C = template.C[:]
    low = 0
    high = 2 * math.pi
    best_alpha = 0
    best_harmony_value = -float('inf')
    tolerance = to_radians(10)  

    while high - low > tolerance:
        mid1 = low + (high - low) / 3
        mid2 = high - (high - low) / 3

        harmony1 = evaluate_harmony(mid1)
        harmony2 = evaluate_harmony(mid2)

        if harmony1 > harmony2:
            high = mid2
            if harmony1 > best_harmony_value:
                best_harmony_value = harmony1
                best_alpha = mid1
        else:
            low = mid1
            if harmony2 > best_harmony_value:
                best_harmony_value = harmony2
                best_alpha = mid2

    return Modele.radCongru(best_alpha)  # Ensure the final angle is valid


def best_template(image):
    '''
        Iterate over each template and angle for each template to get the highest harmony level.
    '''

    best_harmony_value = 0.0
    best_template = None
    best_alpha = None
    for template_name in Modele.get_liste_modeles():
        template = Modele(template_name)
        alpha = best_angle_radians(image, template)
        harmony_value = harmony_by_template(image, template)
        
        # Ensure harmony_value is a valid number
        if not isinstance(harmony_value, (int, float)) or math.isnan(harmony_value) or math.isinf(harmony_value):
            raise ValueError(f"Invalid harmony value: {harmony_value}")
        
        if harmony_value > best_harmony_value:
            best_harmony_value = harmony_value
            best_template = template
            best_alpha = alpha


    if best_template is None or best_alpha is None:
        raise ValueError("No valid template or angle found for the given image.")

    return best_template, best_alpha


def project_hue(center , variance , h):
    '''
        Project the given hue according to the projection formula in da paper
    '''

    d = Modele.distance_congru(h, center)
    new_h = center + (variance) * (1 - math.exp(-1/2 * (( (d) ** 2) / (variance ** 2))) / math.sqrt(variance*2*math.pi))
    new_h = Modele.congru(new_h) 
    return new_h


############################################################# Different harmonization methods ########################################################

def harmonize(image, imageOut, template: Modele, alpha):
    ''' 
        Harmonize a given image with a manually set template and angle.
        This uses the naive one that just sends pixels with no regard to neighboring pixels
    '''

    #Rotate the template
    template.radRotate(alpha)
    
    #Convert to HSV
    image = image.convert("RGB")
    width, height = image.size
    pixels = image.load()
    pixels_out = imageOut.load()

    #Harmonization
    for i in range(width):
        for j in range(height):
            r, g, b = pixels[i, j]
            h, s, v = r2h(r / 255.0, g / 255.0, b / 255.0)
            closest_sector_index = assign_closest_sector_index(h, template)
            central_hue = template.C[closest_sector_index]
            sector_width = template.w[closest_sector_index]

            #If already in the template, keep it
            if template.distance_secteur(h, closest_sector_index) == -1:
                new_h = h
            else:
                new_h = project_hue(central_hue , sector_width/2 , h)

            new_r, new_g, new_b = [int(c * 255) for c in h2r(new_h, s, v)] #Convert back to RGB
            pixels_out[i, j] = (new_r, new_g, new_b)

    return imageOut


def harmonize_opti(image , imageOut , template : Modele , alpha):
    ''' 
        Harmonize a given image with a manually set template and angle.
        This uses the optimized algorithm that fixes neighboring pb
    '''

    #Rotate the template
    template.radRotate(alpha)
    
    #Convert to HSV
    image = image.convert("RGB")
    width, height = image.size
    pixels_out = imageOut.load()

    h_array , s_array , v_array = preprocess_hsv(image)

    #Harmonization
    for i in range(width):
        for j in range(height):

            h = h_array[i, j]
            s = s_array[i, j]
            v = v_array[i, j]
            
            closest_edge_index = assign_closest_sector_index_optimized(template, h_array , s_array , v_array , i , j)
            central_hue = template.C[closest_edge_index]
            sector_width = template.w[closest_edge_index]

            #If already in the template, keep it
            if template.distance_secteur(h, closest_edge_index) == -1:
                new_h = h
                
            else:
                new_h = project_hue(central_hue , sector_width/2 , h)
            
            new_r, new_g, new_b = [int(c * 255) for c in h2r(new_h, s, v)] #Convert back to RGB
            pixels_out[i, j] = (new_r, new_g, new_b)

    return imageOut

def harmonize_opti_from_arrays(h_array, s_array, v_array, imageOut, template: Modele, alpha):
    template.radRotate(alpha)
    pixels_out = imageOut.load()
    height, width = h_array.shape

    for i in range(height):
        for j in range(width):
            h = h_array[i, j]
            s = s_array[i, j]
            v = v_array[i, j]

            closest_index = assign_closest_sector_index_optimized(template, h_array, s_array, v_array, i, j)
            central_hue = template.C[closest_index]
            sector_width = template.w[closest_index]

            if template.distance_secteur(h, closest_index) == -1:
                new_h = h
            else:
                new_h = project_hue(central_hue, sector_width / 2, h)

            r, g, b = [int(c * 255) for c in h2r(new_h, s, v)]
            pixels_out[i,j] = (r, g, b)

    return imageOut


def harmonize_auto_angle(image, imageOut, template: Modele):
    image = image.convert("RGB")
    h_array, s_array, v_array = preprocess_hsv(image)

    alpha = best_angle_radians(h_array, s_array, v_array, template)
    print("Best angle:", to_degrees(alpha))

    return harmonize_opti_from_arrays(h_array, s_array, v_array, imageOut, template, alpha)

def harmonize_auto(image, imageOut):
    cpt = 0
    image = image.convert("RGB")
    
    h_array, s_array, v_array = preprocess_hsv(image)

    best_tmpl = None
    best_alpha = None
    best_val = -float('inf')

    #for template_name in Modele.get_liste_modeles():
    for template_name in "T":
        template = Modele(template_name)
        alpha = best_angle_radians(h_array, s_array, v_array, template)
        harmony_value = harmony_by_template(h_array, s_array, v_array, template)
        if harmony_value > best_val:
            best_val = harmony_value
            best_tmpl = template
            best_alpha = alpha
        cpt+=1
        print(f"{cpt}th template done : {to_degrees(best_alpha)}, harmony = {harmony_value} , template = {template_name}" )

    print("Best template:", best_tmpl.type)
    print("Best angle:", to_degrees(best_alpha))

    return harmonize_opti_from_arrays(h_array, s_array, v_array, imageOut, best_tmpl, best_alpha)


############################################################ Tests ########################################################################################

#Opening the image
nameImage = "./src/images/peacock.jpg"
im = Image.open(nameImage)
imOut = Image.new(im.mode, im.size)

template = Modele("I")
alpha = to_radians(200)

# print("Harmonization opti")
# # Measure time for harmonize_opti
# start_time = time.time()
# imOut2 = Image.new(im.mode, im.size)
# harmonize_opti(im, imOut2, template, to_radians(45))
# imOut2.save("./src/images/peacockReg.jpg")
# print("Time for harmonize_opti:", time.time() - start_time, "seconds")

# # Measure time for harmonize_auto_angle
# start_time = time.time()
# imOut4 = Image.new(im.mode, im.size)
# harmonize_auto_angle(im, imOut4, template)
# imOut4.save("./src/images/30on30AutoAngle.jpg")
# print("Time for harmonize_auto_angle:", time.time() - start_time, "seconds")

# Measure time for harmonize_auto
print("Starting to save auto")
start_time = time.time()
imOut3 = Image.new(im.mode, im.size)
#harmonize_auto(im, imOut3)
harmonize_auto(im, imOut3)
imOut3.save("./src/images/peacock_fullauto.jpg")
print("Time for harmonize_auto:", time.time() - start_time, "seconds")

