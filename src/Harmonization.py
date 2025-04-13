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

#idk this looks op ash 
#https://numba.pydata.org/numba-doc/dev/user/5minguide.html
from numba import jit,njit,prange

#profiling to see whats da bottleneck
import cProfile
import re
############################################################## Utils ################################################################################""

@njit
def to_degrees(alpha):
    '''
        Radians to degrees
    '''
    return alpha * 180 / math.pi

@njit
def to_radians(alpha):
    '''
        Degrees to radians
    '''
    return alpha * math.pi / 180

############################################################## Closest border n dem ###################################################################

# def assign_closest_sector_index(hue, model: Modele):
#     ''' 
#         Return the center of the closest edge, us just calculating distance.
#     '''
    
#     min_distance = float('inf')
#     closest_edge_index = None
    
#     for i in range(len(model.C)):
#         distance = model.distance_secteur(hue, i)
#         if distance == -1: 
#             return i
#         if distance < min_distance:
#             min_distance = distance
#             closest_edge_index = i

#     return closest_edge_index


# def assign_closest_edge(hue, model: Modele):
#     '''
#         Returns the closest border based on the closest edge
#     '''

#     sector_index = assign_closest_sector_index(hue, model)
#     if sector_index is None:
#         return hue 

#     bord = model.bord(sector_index)

#     dist1 = Modele.distance_congru(hue , bord[0])
#     dist2 = Modele.distance_congru(hue , bord[1])
#     if(dist1 < dist2):
#         return bord[1]
#     else : 
#         return bord[0]

 
# def assign_closest_edge_opti(hue, h_array, s_array, v_array, model: Modele, i, j):
#     sector_index = assign_closest_sector_index_optimized(model, h_array, s_array, v_array, i, j)
#     if sector_index is None:
#         return hue

#     bord = model.bord(sector_index)
#     dist1 = Modele.distance_congru(hue, bord[0])
#     dist2 = Modele.distance_congru(hue, bord[1])
#     return bord[1] if dist1 < dist2 else bord[0]

# def assign_closest_sector_index_optimized(model: Modele, h_array , s_array , v_array , i, j):
#     pixels = get_neighboring_pixels(h_array, i, j) 
#     omega = get_omega(pixels, h_array , s_array , v_array)

#     min_energy = float('inf')
#     closest_sector_index = None

#     for idx in range(len(model.C)):  
#         central_hue = model.C[idx]
#         E1 = 0
#         for hn, sn , _ in omega:
#             hvp = project_hue(central_hue , model.w[idx]/2 , hn)
#             E1 += Modele.distance_congru(hn, hvp) * sn
 
#         if E1 < min_energy:  
#             min_energy = E1
#             closest_sector_index = idx

#     return closest_sector_index

############################################################## Testing out Numba jit or njit idk ########################################################


#If i understood correctly, jit (or njit idk the difference yet) is a decorator that ignores python interpreter, so we gotta remove calls to classes n stuff and just precompute that
#thats wht the parameters changed a luh bit
#@jit(nopython=True)
@njit
def assign_closest_edge_opti_jit(hue, h_array, s_array, v_array, model_w, model_c, model_bords, i, j):
    sector_index = assign_closest_sector_index_optimized_jit(model_w, model_c, h_array, s_array, v_array, i, j)
    if sector_index == -1: 
        return hue

    bord = model_bords[sector_index]
    dist1 = distance_congru_jit(hue, bord[0])
    dist2 = distance_congru_jit(hue, bord[1])
    return bord[1] if dist1 < dist2 else bord[0]

@njit
def assign_closest_sector_index_optimized_jit(model_w, model_c, h_array, s_array, v_array, i, j):
    # Get neighbors and count
    neighbors, neighbor_count = get_neighboring_pixels_jit(h_array, i, j) 
    omega = get_omega_jit(neighbors, neighbor_count, h_array, s_array, v_array)

    min_energy = np.inf
    closest_sector_index = -1

    for idx in range(len(model_c)):  
        central_hue = model_c[idx]
        E1 = 0.0
        for n in range(neighbor_count):
            hn = omega[n, 0]  # hue
            sn = omega[n, 1]  # saturation
            hvp = project_hue_jit(central_hue, model_w[idx]/2, hn)
            E1 += distance_congru_jit(hn, hvp) * sn
 
        if E1 < min_energy:  
            min_energy = E1
            closest_sector_index = idx

    return closest_sector_index

@njit
def get_neighboring_pixels_jit(h_array, i, j):
    height, width = h_array.shape
    # Pre-allocate fixed size array (maximum 4 neighbors)
    neighbors = np.zeros((4, 2), dtype=np.int32)
    count = 0
    
    if j + 1 < width:
        neighbors[count, 0] = i
        neighbors[count, 1] = j + 1
        count += 1
    if j - 1 >= 0:
        neighbors[count, 0] = i
        neighbors[count, 1] = j - 1
        count += 1
    if i + 1 < height:
        neighbors[count, 0] = i + 1
        neighbors[count, 1] = j
        count += 1
    if i - 1 >= 0:
        neighbors[count, 0] = i - 1
        neighbors[count, 1] = j
        count += 1

    return neighbors[:count], count  # Return actual neighbors and count

@njit
def get_omega_jit(neighbors, neighbor_count, h_array, s_array, v_array):
    # Pre-allocate omega with maximum size
    omega = np.zeros((neighbor_count, 3), dtype=np.float32)
    
    for idx in range(neighbor_count):
        x = int(neighbors[idx, 0])
        y = int(neighbors[idx, 1])
        omega[idx, 0] = h_array[x, y]  # hue
        omega[idx, 1] = s_array[x, y]  # saturation
        omega[idx, 2] = v_array[x, y]  # value
        
    return omega

@njit
def project_hue_jit(center , variance , h):
    '''
        Project the given hue according to the projection formula in da paper
    '''

    d = distance_congru_jit(h, center)
    new_h = center + (variance) * (1 - math.exp(-1/2 * (( (d) ** 2) / (variance ** 2))) / math.sqrt(variance*2*math.pi))
    new_h = congru_jit(new_h) 
    return new_h

@njit
def distance_congru_jit(a, b) :     # Retourne la distance congru entre deux angles
    d = abs(a-b)                # ex : entre 0 et 0.9, la distance et 0.1 car 0 est congru à 1
    return min(d, 1-d)    

@njit
def congru_jit(n) :             # Permet de faire la congruence
    return n-int(n)         # si on obtient un angle supérieur à 1 par exempl

@njit
def rad_congru_jit(n) : 
    res = n/2/math.pi     
    res -= int(res)
    return 2*math.pi*res


############################################################## Vectorize/Preprocess edges maybe #################################################

## GOAL : Per pixel calculations are the heaviest, so maybe try to avoid calculating them a bajillion times

#NOT BAD TBH
def harmony_by_template_faster(h_array, s_array, v_array, model: Modele):
    '''
        Calculate the harmony value of the image. This is a sort of arbitrary value I guess, but it's in the paper.
    '''
    
    height, width = h_array.shape
    sum_harmony_value = 0
    template_c = model.C
    template_w = model.w
    template_edges = np.array([template.bord(i) for i in range(len(template_c))])    

    for i in range(height):
        for j in range(width):
            hue = h_array[i, j]
            saturation = s_array[i, j]         
            closest_edge = assign_closest_edge_opti_jit(hue, h_array, s_array, v_array, template_w, template_c, template_edges, i, j)
            sum_harmony_value += distance_congru_jit(hue, closest_edge) * saturation

    return sum_harmony_value

@njit(parallel=True)
def harmony_by_template_fastest(h_array, s_array, v_array, template_w, template_c, template_edges):
    height, width = h_array.shape
    sum_harmony_value = 0.0
    
    #prange i guess allows for parallel
    for i in prange(height):
        for j in range(width):
            hue = h_array[i, j]
            saturation = s_array[i, j]         
            closest_edge = assign_closest_edge_opti_jit(hue, h_array, s_array, v_array, template_w, template_c, template_edges, i, j)
            sum_harmony_value += distance_congru_jit(hue, closest_edge) * saturation
    
    return sum_harmony_value



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

def compute_omega_array(h_array, s_array, v_array):
    height, width = h_array.shape
    omega_array = np.empty((height, width), dtype=object)

    for i in range(height):
        for j in range(width):
            neighbors = get_neighboring_pixels(h_array, i, j)
            omega = []
            for x, y in neighbors:
                h = h_array[x][y]
                s = s_array[x][y]
                v = v_array[x][y]
                omega.append((h, s, v))
            omega_array[i, j] = omega

    return omega_array


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
            template_c = model.C
            template_w = model.w
            template_edges = np.array([template.bord(i) for i in range(len(template_c))])             
            closest_edge = assign_closest_edge_opti_jit(hue, h_array, s_array, v_array, template_w, template_c, template_edges, i, j)
            sum_harmony_value += distance_congru_jit(hue, closest_edge) * saturation

    return sum_harmony_value


#https://realpython.com/inner-functions-what-are-they-good-for/
# ^^ that link for understanding why i def in a def
def best_angle_radians(h_array, s_array, v_array, template:Modele):
    '''
        Use binary search to find the best angle for the given template.
    ''' 

    def evaluate_harmony(alpha):
        alpha = rad_congru_jit(alpha) 
        template.radRotate(alpha)
        template_w = template.w
        template_c = template.C
        template_edges = np.array([template.bord(i) for i in range(len(template_c))])  
        harmony_value = harmony_by_template_fastest(h_array, s_array, v_array, template_w, template_c, template_edges)
        #print(f"Sum harmony value : {harmony_value}")
        template.C = original_C[:] 
        return harmony_value

    original_C = template.C[:]
    low = 0
    high = 2 * math.pi
    best_alpha = 0
    best_harmony_value = -float('inf')
    tolerance = to_radians(1)  

    while high - low > tolerance:
        mid1 = low + (high - low) / 3
        mid2 = high - (high - low) / 3
        #print(f"Alpha 1:{mid1}")
        harmony1 = evaluate_harmony(mid1)
        #print(f"Alpha 1:{mid2}")
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

    return rad_congru_jit(best_alpha)  # Ensure the final angle is valid

# def project_hue(center , variance , h):
#     '''
#         Project the given hue according to the projection formula in da paper
#     '''

#     d = Modele.distance_congru(h, center)
#     new_h = center + (variance) * (1 - math.exp(-1/2 * (( (d) ** 2) / (variance ** 2))) / math.sqrt(variance*2*math.pi))
#     new_h = Modele.congru(new_h) 
#     return new_h


############################################################# Different harmonization methods ########################################################

# def harmonize(image, imageOut, template: Modele, alpha):
#     ''' 
#         Harmonize a given image with a manually set template and angle.
#         This uses the naive one that just sends pixels with no regard to neighboring pixels
#     '''

#     #Rotate the template
#     template.radRotate(alpha)
    
#     #Convert to HSV
#     image = image.convert("RGB")
#     width, height = image.size
#     pixels = image.load()
#     pixels_out = imageOut.load()

#     #Harmonization
#     for i in range(width):
#         for j in range(height):
#             r, g, b = pixels[i, j]
#             h, s, v = r2h(r / 255.0, g / 255.0, b / 255.0)
#             closest_sector_index = assign_closest_sector_index(h, template)
#             central_hue = template.C[closest_sector_index]
#             sector_width = template.w[closest_sector_index]

#             #If already in the template, keep it
#             if template.distance_secteur(h, closest_sector_index) == -1:
#                 new_h = h
#             else:
#                 new_h = project_hue(central_hue , sector_width/2 , h)

#             new_r, new_g, new_b = [int(c * 255) for c in h2r(new_h, s, v)] #Convert back to RGB
#             pixels_out[i, j] = (new_r, new_g, new_b)

#     return imageOut


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

            template_c = template.C
            template_w = template.w
            closest_edge_index = assign_closest_sector_index_optimized_jit(template_w, template_c, h_array , s_array , v_array , i , j)
            central_hue = template_c[closest_edge_index]
            sector_width = template_w[closest_edge_index]

            #If already in the template, keep it
            if template.distance_secteur(h, closest_edge_index) == -1:
                new_h = h
                
            else:
                new_h = project_hue_jit(central_hue , sector_width/2 , h)
            
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

            template_c = template.C
            template_w = template.w
            closest_index = assign_closest_sector_index_optimized_jit(template_w, template_c, h_array , s_array , v_array , i , j)            
            central_hue = template_c[closest_index]
            sector_width = template_w[closest_index]

            if template.distance_secteur(h, closest_index) == -1:
                new_h = h
            else:
                new_h = project_hue_jit(central_hue, sector_width / 2, h)

            r, g, b = [int(c * 255) for c in h2r(new_h, s, v)]
            pixels_out[i,j] = (r, g, b)

    return imageOut


def harmonize_auto_angle(image, imageOut, template: Modele):
    image = image.convert("RGB")
    h_array, s_array, v_array = preprocess_hsv(image)

    template_c = template.C
    template_w = template.w
    template_edges = np.array([template.bord(i) for i in range(len(template_c))])  
        
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

    for template_name in Modele.get_liste_modeles():
    #for template_name in "V":
        template = Modele(template_name)
        template_c = template.C
        template_w = template.w
        template_edges = np.array([template.bord(i) for i in range(len(template_c))])  
        
        alpha = best_angle_radians(h_array, s_array, v_array, template)
        harmony_value = harmony_by_template_fastest(h_array, s_array, v_array, template_w, template_c, template_edges)
        if harmony_value > best_val:
            best_val = harmony_value
            best_tmpl = template
            best_alpha = alpha
        cpt+=1
        #print(f"{cpt}th template done : {to_degrees(best_alpha)}, harmony = {harmony_value} , template = {template_name}" )

    print("Best template:", best_tmpl.type)
    print("Best angle:", to_degrees(best_alpha))

    return harmonize_opti_from_arrays(h_array, s_array, v_array, imageOut, best_tmpl, best_alpha)

def color_harmonisation(imIn, template_name):

    imOut = Image.new(imIn.mode, imIn.size)

    if template_name == "auto":
        harmonize_auto(imIn, imOut)
    else :
        template = Modele(template_name)
        harmonize_auto_angle(imIn, imOut, template)
        
    #imOut.save(filename)
    return imOut

############################################################ Tests ########################################################################################

#Opening the image
nameImage = "./src/images/peacock.jpg"
im = Image.open(nameImage)
imOut = Image.new(im.mode, im.size)

template = Modele("T")
alpha = to_radians(200)

print("Harmonization opti")
# Measure time for harmonize_opti
# start_time = time.time()
# imOut2 = Image.new(im.mode, im.size)
# harmonize_opti(im, imOut2, template, to_radians(240))
# imOut2.save("./src/images/peacockReg.jpg")
# print("Time for harmonize_opti:", time.time() - start_time, "seconds")

# # Measure time for harmonize_auto_angle
start_time = time.time()
imOut4 = Image.new(im.mode, im.size)
harmonize_auto_angle(im, imOut4, template)
imOut4.save("./src/images/30on30AutoAngle.jpg")
print("Time for harmonize_auto_angle:", time.time() - start_time, "seconds")


#Profile the harmonize_auto_angle function for performance analysis
# imOut4 = Image.new(im.mode, im.size)
# print("Starting to profile harmonize auto angle")
# cProfile.run('harmonize_auto_angle(im, imOut4, template)')
# imOut4.save("./src/images/30on30AutoAngle.jpg")
# print("Done")

# Measure time for harmonize_auto
# print("Starting to save auto")
# start_time = time.time()
# imOut3 = Image.new(im.mode, im.size)
# harmonize_auto(im, imOut3)
# imOut3.save("./src/images/peacock_fullauto.jpg")
# print("Time for harmonize_auto:", time.time() - start_time, "seconds")

