import streamlit as st
import cv2
import numpy as np
from converter import extract_contours, generate_gcode
import matplotlib.pyplot as plt
from PIL import Image
import io

st.set_page_config(
    page_title="CNC PNG to G-code",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more premium look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
    }
    .stDownloadButton>button {
        width: 100%;
        background-color: #1565c0;
        color: white;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e3b4e,#2e3b4e);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛠️ PNG to G-code pour les collègues")
st.markdown("Convertissez vos images en fichiers G-code (.nc) pour votre machine CNC DIY.")

# Sidebar parameters
st.sidebar.header("⚙️ Paramètres de Coupe")
z_safe = st.sidebar.number_input("Hauteur de sécurité ($Z_{safe}$) [mm]", value=5.0, step=0.5, help="Hauteur pour les déplacements rapides G0")
z_depth = st.sidebar.number_input("Profondeur totale ($Z_{depth}$) [mm]", value=6.0, min_value=0.1, step=0.5, help="Profondeur finale de la coupe")
z_pass = st.sidebar.number_input("Profondeur par passe [mm]", value=2.0, min_value=0.1, step=0.5, help="Profondeur maximale par passage de l'outil")

col_f1, col_f2 = st.sidebar.columns(2)
feed_rate = col_f1.number_input("Avance XY [mm/min]", value=1000, min_value=1, step=100)
feed_rate_z = col_f2.number_input("Avance Z [mm/min]", value=300, min_value=1, step=50)

scale = st.sidebar.number_input("Échelle (Pixel vers mm)", value=0.1, format="%.4f", help="Facteur de multiplication pour passer des pixels aux mm")

st.sidebar.markdown("---")
st.sidebar.header("🛠️ Outil & Précision")
tool_diameter = st.sidebar.number_input("Diamètre de la fraise [mm]", value=0.0, min_value=0.0, step=0.1, help="0 pour aucune compensation. Décale le tracé vers l'extérieur.")
simplification = st.sidebar.slider("Simplification des courbes", 0.0, 0.05, 0.005, format="%.4f", help="Plus la valeur est haute, moins il y a de petits segments (évite les saccades).")

st.sidebar.markdown("---")
st.sidebar.header("👁️ Détection des contours")
use_canny = st.sidebar.toggle("Utiliser Canny (Contours)", value=True)
if use_canny:
    thresh1 = st.sidebar.slider("Canny Seuil Bas", 0, 500, 100)
    thresh2 = st.sidebar.slider("Canny Seuil Haut", 0, 500, 200)
else:
    thresh1 = st.sidebar.slider("Seuil Binaire (Threshold)", 0, 255, 127)
    thresh2 = 255 # Non utilisé pour threshold

uploaded_file = st.file_uploader("📂 Charger une image PNG ou JPG", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🖼️ Image Originale")
        st.image(image_bytes, use_container_width=True)
        
    # Extraction des contours
    with st.spinner("Analyse de l'image..."):
        contours, original_img = extract_contours(
            image_bytes, 
            threshold1=thresh1, 
            threshold2=thresh2, 
            scale=scale, 
            use_canny=use_canny,
            simplification=simplification,
            tool_diameter=tool_diameter
        )
    
    if contours is not None and len(contours) > 0:
        with col2:
            st.subheader("📍 Aperçu du parcours CNC")
            fig, ax = plt.subplots(figsize=(8, 8))
            for cnt in contours:
                pts = cnt.reshape(-1, 2)
                # Fermer le contour pour l'aperçu
                pts = np.vstack([pts, pts[0]])
                ax.plot(pts[:, 0], pts[:, 1], 'b-', linewidth=1)
            
            ax.set_aspect('equal')
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_title(f"{len(contours)} contours détectés")
            st.pyplot(fig)
            
        st.markdown("---")
        
        if st.button("🚀 GÉNÉRER LE G-CODE"):
            with st.spinner("Génération des instructions G-code..."):
                gcode_text = generate_gcode(contours, z_safe, z_depth, z_pass, feed_rate, feed_rate_z)
                
                st.success(f"✅ G-code généré ! Profondeur totale : {z_depth}mm en {int(np.ceil(z_depth/z_pass))} passes.")
                
                # Visualisation du G-code
                with st.expander("📝 Aperçu des premières lignes du fichier"):
                    lines = gcode_text.split('\n')
                    st.code('\n'.join(lines[:50]) + ("\n..." if len(lines) > 50 else ""), language="gcode")
                
                # Download button
                st.download_button(
                    label="📥 TÉLÉCHARGER LE FICHIER G-CODE (.nc)",
                    data=gcode_text,
                    file_name=f"{uploaded_file.name.split('.')[0]}_cnc.nc",
                    mime="text/plain"
                )
    else:
        st.error("❌ Aucun contour n'a pu être extrait. Essayez d'ajuster les seuils de détection.")
else:
    st.info("💡 Commencez par charger une image pour voir l'aperçu et générer le code.")
