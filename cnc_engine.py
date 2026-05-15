import cv2
import numpy as np
from shapely.geometry import Polygon

def extract_contours(image_bytes, threshold1=100, threshold2=200, scale=1.0, use_canny=True, simplification=0.1, tool_diameter=0.0):
    """
    Extrait les contours d'une image PNG, applique une simplification et une compensation d'outil.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Ajout d'une bordure blanche pour fermer les contours qui touchent le bord de l'image
    border_size = 5
    gray = cv2.copyMakeBorder(gray, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=255)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if use_canny:
        edges = cv2.Canny(blurred, threshold1, threshold2)
    else:
        _, edges = cv2.threshold(blurred, threshold1, 255, cv2.THRESH_BINARY_INV)

    contours_raw, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = img.shape[:2]
    scaled_contours = []
    
    if hierarchy is not None:
        for i, cnt in enumerate(contours_raw):
            # Simplification du contour (Ramer-Douglas-Peucker)
            epsilon = simplification * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            if len(approx) < 2: continue
            
            # Gérer la hiérarchie pour identifier les trous
            is_top_level = (hierarchy[0][i][3] == -1)
            
            # Garder uniquement les composants de premier niveau en Canny (évite les doubles lignes)
            if use_canny and not is_top_level:
                continue
                
            scaled_cnt = approx.astype(np.float32).copy()
            
            # Soustraire la bordure avant la mise à l'échelle
            scaled_cnt[:, 0, 0] -= border_size
            scaled_cnt[:, 0, 1] -= border_size
            
            # Mise à l'échelle
            scaled_cnt[:, 0, 0] *= scale # X
            scaled_cnt[:, 0, 1] *= scale # Y
            
            # Inversion axe Y (CNC)
            scaled_cnt[:, 0, 1] = (h * scale) - scaled_cnt[:, 0, 1]
            
            # --- Nouvelle méthode d'offset vectorielle avec Shapely ---
            if tool_diameter > 0 and len(scaled_cnt) >= 3:
                radius = tool_diameter / 2.0
                pts = scaled_cnt.reshape(-1, 2)
                
                # Créer le polygone fermé
                if np.linalg.norm(pts[0] - pts[-1]) < 1e-3 or len(pts) > 2:
                    poly = Polygon(pts)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                        
                    # Aggrandir l'extérieur, rétrécir l'intérieur (trous)
                    offset_dist = radius if is_top_level else -radius
                    
                    try:
                        buffered = poly.buffer(offset_dist, resolution=8, join_style=1) # ROUND
                    except:
                        buffered = poly
                        
                    if buffered.is_empty:
                        continue
                        
                    geoms = [buffered] if buffered.geom_type == 'Polygon' else buffered.geoms
                    for geom in geoms:
                        if geom.exterior is not None:
                            ext_coords = np.array(geom.exterior.coords)
                            if len(ext_coords) > 0:
                                ext_cnt = ext_coords.reshape(-1, 1, 2).astype(np.float32)
                                # Re-simplifier après l'offset (Shapely génère beaucoup de points sur les arrondis)
                                eps_mm = simplification * cv2.arcLength(ext_cnt, True)
                                ext_cnt = cv2.approxPolyDP(ext_cnt, eps_mm, True)
                                if len(ext_cnt) >= 2:
                                    scaled_contours.append(ext_cnt)
                else:
                    scaled_contours.append(scaled_cnt)
            else:
                scaled_contours.append(scaled_cnt)
                
    return scaled_contours, img

def get_arc_params(p1, p2, p3):
    """
    Calcule le centre et le rayon du cercle passant par 3 points.
    Retourne (center_x, center_y, radius, is_clockwise)
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    # Méthode des médiatrices
    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(D) < 1e-6:
        return None # Points alignés
    
    center_x = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / D
    center_y = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / D
    
    radius = np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)
    
    # Déterminer le sens (produit vectoriel)
    # Vecteur AB et BC
    v1 = (x2 - x1, y2 - y1)
    v2 = (x3 - x2, y3 - y2)
    z = v1[0] * v2[1] - v1[1] * v2[0]
    is_clockwise = z < 0
    
    return (center_x, center_y, radius, is_clockwise)

def generate_gcode(contours, z_safe, z_depth, z_pass, feed_rate, feed_rate_z, use_arcs=False):
    """
    Génère le G-code avec support optionnel des arcs G2/G3 et compression.
    """
    gcode = []
    gcode.append("; G-code genere par PNG-to-Gcode Expert (Optimise)")
    gcode.append("G21 ; mm")
    gcode.append("G90 ; Abs")
    gcode.append("G94 ; mm/min")
    gcode.append("M3 S1000")
    gcode.append(f"G0 Z{z_safe:.3f}")
    
    num_passes = int(np.ceil(z_depth / z_pass)) if z_pass > 0 else 1
    
    current_f = None
    
    def get_f_str(f_val):
        nonlocal current_f
        if current_f != f_val:
            current_f = f_val
            return f" F{f_val:.0f}"
        return ""

    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        if len(pts) < 2: continue
        
        # Aller au point de départ
        gcode.append(f"G0 X{pts[0][0]:.3f} Y{pts[0][1]:.3f}")
        
        for p in range(1, num_passes + 1):
            current_z = -min(p * z_pass, z_depth)
            f_str = get_f_str(feed_rate_z)
            gcode.append(f"G1 Z{current_z:.3f}{f_str}")
            
            i = 0
            while i < len(pts):
                p1 = pts[i]
                # Point suivant (bouclage si dernier point pour fermer le contour)
                idx2 = (i + 1) % len(pts)
                idx3 = (i + 2) % len(pts)
                
                p2 = pts[idx2]
                p3 = pts[idx3]
                
                arc = None
                if use_arcs and len(pts) > 2:
                    arc = get_arc_params(p1, p2, p3)
                
                if arc:
                    cx, cy, r, cw = arc
                    # On ne fait un arc que si le rayon est raisonnable (pas trop grand/infini)
                    if 0.1 < r < 1000:
                        cmd = "G2" if cw else "G3"
                        # I, J sont relatifs au point de départ (p1)
                        I = cx - p1[0]
                        J = cy - p1[1]
                        f_str = get_f_str(feed_rate)
                        gcode.append(f"{cmd} X{p3[0]:.3f} Y{p3[1]:.3f} I{I:.3f} J{J:.3f}{f_str}")
                        i += 2 # On a sauté p2
                        if i >= len(pts): break
                        continue
                
                # Sinon simple ligne G1
                f_str = get_f_str(feed_rate)
                gcode.append(f"G1 X{p2[0]:.3f} Y{p2[1]:.3f}{f_str}")
                i += 1
            
            # Retour au point de depart pour fermer (si pas déjà fait par le dernier arc/ligne)
            f_str = get_f_str(feed_rate)
            gcode.append(f"G1 X{pts[0][0]:.3f} Y{pts[0][1]:.3f}{f_str}")
            
        gcode.append(f"G0 Z{z_safe:.3f}")
        
    gcode.append("M5")
    gcode.append("G0 X0 Y0")
    gcode.append("M30")
    
    return "\n".join(gcode)
