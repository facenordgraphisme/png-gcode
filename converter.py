import cv2
import numpy as np

def extract_contours(image_bytes, threshold1=100, threshold2=200, scale=1.0, use_canny=True):
    """
    Extrait les contours d'une image PNG et les met à l'échelle.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if use_canny:
        edges = cv2.Canny(blurred, threshold1, threshold2)
    else:
        # Threshold simple pour les images contrastées (logo noir sur blanc)
        _, edges = cv2.threshold(blurred, threshold1, 255, cv2.THRESH_BINARY_INV)
        
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Mise à l'échelle et inversion de l'axe Y pour correspondre au repère CNC
    # (L'origine (0,0) est souvent en bas à gauche sur une CNC)
    h, w = img.shape[:2]
    scaled_contours = []
    for cnt in contours:
        if len(cnt) < 2: continue
        scaled_cnt = cnt.astype(np.float32).copy()
        # Scale
        scaled_cnt[:, 0, 0] *= scale # X
        scaled_cnt[:, 0, 1] *= scale # Y
        
        # Invert Y: Y_cnc = (Height_px * scale) - Y_px * scale
        scaled_cnt[:, 0, 1] = (h * scale) - scaled_cnt[:, 0, 1]
        
        scaled_contours.append(scaled_cnt)
        
    return scaled_contours, img

def generate_gcode(contours, z_safe, z_depth, z_pass, feed_rate):
    """
    Génère le G-code avec gestion multi-passes.
    z_depth et z_pass sont positifs dans l'UI mais convertis en négatif pour le travail.
    """
    gcode = []
    
    # En-tête
    gcode.append("; G-code genere par PNG-to-Gcode")
    gcode.append("G21 ; Unites en mm")
    gcode.append("G90 ; Positionnement absolu")
    gcode.append("G94 ; Avance en mm/min")
    gcode.append("M3 S1000 ; Broche ON")
    gcode.append(f"G0 Z{z_safe:.3f} ; Hauteur de securite")
    
    # Calcul du nombre de passes
    num_passes = int(np.ceil(z_depth / z_pass)) if z_pass > 0 else 1
    
    for cnt in contours:
        start_point = cnt[0][0]
        
        # Deplacement rapide au depart du contour
        gcode.append(f"G0 X{start_point[0]:.3f} Y{start_point[1]:.3f}")
        
        for p in range(1, num_passes + 1):
            current_z = -min(p * z_pass, z_depth)
            
            # Plongee en Z (moitie de la vitesse d'avance)
            gcode.append(f"G1 Z{current_z:.3f} F{feed_rate/2:.0f}")
            
            # Suivi du contour
            for pt in cnt:
                x, y = pt[0]
                gcode.append(f"G1 X{x:.3f} Y{y:.3f} F{feed_rate:.0f}")
                
            # Retour au point de depart pour fermer le contour
            gcode.append(f"G1 X{start_point[0]:.3f} Y{start_point[1]:.3f} F{feed_rate:.0f}")
            
        # Remontee en Z securise apres avoir fini toutes les passes du contour
        gcode.append(f"G0 Z{z_safe:.3f}")
        
    # Pied de page
    gcode.append("M5 ; Broche OFF")
    gcode.append("G0 X0 Y0 ; Retour origine")
    gcode.append("M30 ; Fin de programme")
    
    return "\n".join(gcode)
