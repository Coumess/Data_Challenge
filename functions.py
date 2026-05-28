# ==========================================================================================
# Importatations
# ==========================================================================================
import json
import numpy as np
from scipy.ndimage import convolve1d    

# ==========================================================================================
# Correction Json Functions
# ==========================================================================================

def load_json(file_path):
    """
    Loads a JSON file and returns its content.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        list/dict: The parsed JSON data.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    return data

# =========================================================================================

def get_defect_coordinates(json_data, image_number):
    """
    Parses JSON data to extract defect coordinates for a specific image number.

    Args:
        json_data (list): The list of dictionaries loaded from the JSON file.
        image_number (str or int): The specific image number to look for in the JSON.

    Returns:
        dict: A dictionary where keys are x-coordinates (columns) and values 
              are dictionaries containing lists of 'start' and 'stop' y-coordinates.
    """
    defect_dict = {}
    
    for item in json_data:
        value = item.get('signal', {}).get(str(image_number))
        
        if value is not None:
            for x_coord in item.get('x_coord', []):
                
                defect_dict[x_coord] = {
                    'ycords': (list(zip(item.get('y_start', []),item.get('y_stop', [])))),
                    'type': item.get('name', [])
                }

    return defect_dict

# ===========================================================================
# Detect Methods Functions :
# ===========================================================================
def detecter_et_mesurer_defauts_complet(image, hauteur_bande=50, train_cells=4, guard_cells=2, multiplicateur_rupture=3.5):
    """
    Pipeline complet de détection et mesure des colonnes défectueuses (entières et fragmentées).
    
    1. Découpe l'image en bandes et applique un filtre CA-CFAR pour trouver des "graines" de défauts.
    2. Prolonge ces graines vers le haut et le bas (Region Growing) jusqu'à une rupture d'intensité.
    3. Fusionne les segments qui se chevauchent et formate le résultat final.
    
    Args:
        image (numpy.ndarray): L'image 16-bits en entrée.
        hauteur_bande (int): Taille de la bande horizontale pour le CFAR (défaut: 50).
        train_cells (int): Nombre de cellules d'entraînement de chaque côté (défaut: 4).
        guard_cells (int): Nombre de cellules de garde de chaque côté (défaut: 2).
        multiplicateur_rupture (float): Tolérance pour le seuil de rupture lors de la croissance (défaut: 3.5).
        
    Returns:
        dict: Dictionnaire formaté {x: [(y1_start, y1_end), (y2_start, y2_end), ...]} 

    """
    height, width = image.shape

    # =========================================================
    # ETAPE 1 : DETECTION DES GRAINES (CFAR par bandes)
    # =========================================================
    segments_initiaux = []
    
    # --- Paramètres du CFAR ---
    num_train_side = train_cells 
    num_guard_side = guard_cells  
    taille_fenetre = (num_train_side * 2) + (num_guard_side * 2) + 1 
    noyau = np.zeros(taille_fenetre)
    noyau[:num_train_side] = 1.0  
    noyau[-num_train_side:] = 1.0 
    noyau = noyau / (num_train_side * 2)

    # On parcourt l'image de haut en bas, en sautant de 'hauteur_bande' en 'hauteur_bande'
    for y_start in range(0, height, hauteur_bande):
        y_end = min(y_start + hauteur_bande, height) # min() pour ne pas déborder à la fin
        
        # On extrait la sous-image (la bande horizontale)
        bande = image[y_start:y_end, :]
        
        # On calcule la projection médiane UNIQUEMENT sur cette bande
        projection_bande = np.median(bande, axis=0)
        
        # On applique le CFAR
        bruit_de_fond_local = convolve1d(projection_bande, noyau, mode='reflect')
        
        # Tolérance locale pour cette bande spécifique
        ecart_type_bande = np.std(projection_bande)
        tolerance = 3 * ecart_type_bande  
        
        seuil_haut = bruit_de_fond_local + tolerance
        seuil_bas = bruit_de_fond_local - tolerance
        
        # Détection pour cette bande
        colonnes_detectees = np.where(
            (projection_bande > seuil_haut) | 
            (projection_bande < seuil_bas)
        )[0]
        
        # On enregistre les résultats sous forme de segments initiaux (graines)
        for x in colonnes_detectees:
            segments_initiaux.append([(x, y_start), (x, y_end)])


    # =========================================================
    # ETAPE 2 : CROISSANCE DE REGION (Region Growing)
    # =========================================================
    segments_etendus = []

    for segment in segments_initiaux:
        # On force la conversion en entier natif python pour la suite
        x = int(segment[0][0])
        y_start = int(segment[0][1])
        y_end = int(segment[1][1])
        
        colonne = image[:, x].astype(np.float32) # pour chaque x, on prend tous les px de la col
        
        # 1. Calculer ce qu'est un "saut normal" / et seuil sur cette colonne
        sauts_verticaux = np.abs(np.diff(colonne))  # On regarde la dérivée absolue (la différence entre chaque pixel et le suivant (de la col))

        bruit_normal = np.median(sauts_verticaux) # on calc les sauts normaux (sur le défaut, ou sur le vrai paysage de l'img) en prennant médiane de tous les sauts 
        ecart_sauts = np.std(sauts_verticaux)
        
        seuil_rupture = bruit_normal + (multiplicateur_rupture * ecart_sauts) # calc marche qu'on considère trop grande (fin du défaut)

        # 2. Prolonger vers le HAUT (on remonte la colonne)
        while y_start > 0: # boucle tant qu'on a pas atteint le haut de l'image
            # Quelle est la taille de la marche pour monter sur le pixel du dessus ?
            saut_haut = np.abs(colonne[y_start] - colonne[y_start - 1]) # diff entre px et celui d'avnat 
            
            if saut_haut < seuil_rupture:
                # Intensité similaire : on est toujours dans le défaut
                y_start -= 1 # on continue de rémonter 
            else:
                # BOUM ! Rupture forte d'intensité, on a trouvé le bord supérieur.
                break

        # 3. Prolonger vers le BAS (on descend la colonne)
        while y_end < (height - 1): # boucle tant qu'on a pas atteint le bas de l'image
            # Quelle est la taille de la marche pour descendre sur le pixel du dessous ?
            saut_bas = np.abs(colonne[y_end] - colonne[y_end + 1]) # diff entre px et celui d'après 
            
            if saut_bas < seuil_rupture:
                y_end += 1 # on continue de descendre 
            else:
                break # sinon saut trop gros, on a trouvé bord inf
                
        # On stocke les coordonnées étendues sous forme de liste simple pour la fusion
        segments_etendus.append([x, y_start, y_end])


    # =========================================================
    # ETAPE 3 : FUSION ET FORMATAGE DU DICTIONNAIRE FINAL
    # =========================================================
    if not segments_etendus: 
        return {}
    
    # Grouper par colonne (x)
    dict_cols = {}
    for seg in segments_etendus:
        x, y_s, y_e = seg[0], seg[1], seg[2]
        if x not in dict_cols: 
            dict_cols[x] = []
        dict_cols[x].append([y_s, y_e])
        
    segments_fusion_dict = {}
    
    for x, intervalles in dict_cols.items():
        intervalles.sort(key=lambda v: v[0]) # Trier de haut en bas
        fusion = [intervalles[0]]
        
        for courant in intervalles[1:]:
            dernier = fusion[-1]
            # Si les segments se chevauchent ou se touchent
            if courant[0] <= dernier[1] + 1:
                dernier[1] = max(dernier[1], courant[1])
            else:
                fusion.append(courant)
                
        # On remplit le dictionnaire avec une liste de tuples (y_start, y_end)
        segments_fusion_dict[x] = [(y_s, y_e) for y_s, y_e in fusion]
            
    return segments_fusion_dict

# ================================================================================================

