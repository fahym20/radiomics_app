import csv
import os
import time
import json
import shutil
import six
# import copy
# import difflib
# import dicom2nifti
import streamlit as st
# import yaml
# from pathlib import Path
# import pydicom as pdc
# import pandas as pd
# from collections import OrderedDict
# from SimpleITK import ImageFileReader, ImageSeriesReader
# from matplotlib.backends.backend_pdf import PdfPages
# import matplotlib.pyplot as plt
# import numpy as np
# import nibabel as nb
# from radiomics import featureextractor, getFeatureClasses

# from anonymizator_functions import *
# from niftiConvertor_functions import *


st.set_page_config(
    page_title="Radiomics App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="auto"
)

st.markdown(
    """
    <style>
    :root {
        --primary-color: blue;
    }
    body {
        color: var(--primary-color);
    }
    </style>
    """,
    unsafe_allow_html=True
)

    
def st_file_selector(clef, path='.', label='FILE'):
    # get base path (directory)
    base_path = '.' if path is None or path is '' else path
    base_path = base_path if os.path.isdir(
        base_path) else os.path.dirname(base_path)
    base_path = '.' if base_path is None or base_path is '' else base_path
    # list files in base path directory
    files = os.listdir(base_path)
    selected_file = st.selectbox(label=label, options=files, key=clef, help='TODO')
    selected_path = os.path.normpath(os.path.join(base_path, selected_file))
    return selected_path


def lister_dossiers_dossier(chemin_du_dossier):
    try:
        elements = os.listdir(chemin_du_dossier)
        dossiers = [element for element in elements if os.path.isdir(os.path.join(chemin_du_dossier, element))]
        for dossier in dossiers:
            return dossier

    except FileNotFoundError:
        print("Le dossier spécifié n'existe pas.")


def lister_noms_fichiers_dossier(chemin_du_dossier):
    try:
        elements = os.listdir(chemin_du_dossier)
        fichiers = [element for element in elements if os.path.isfile(os.path.join(chemin_du_dossier, element))]
        return fichiers
    except FileNotFoundError:
        print("Le dossier spécifié n'existe pas.")
        return []


def main():
    st.title("**Radiomics App**")

    page = st.sidebar.radio("Sélectionnez une page", ("Anonymizator", "NIfTI Convertor", "PyRadiomics Extractor"))

    if page == "Anonymizator":
        page1()
    elif page == "NIfTI Convertor":
        page2()
    elif page == "PyRadiomics Extractor":
        page3()

# Anonymizator

def page1():
    st.title("Anonymizator")
    st.header("*Interface graphique du logiciel d'anonymisation des DICOM*")
    st.subheader("1) Réglages et options")

    with open('champs_a_anonymiser.txt') as f:
        lines = f.readlines()
    active_lines = [x for x in lines if x.startswith('#') == False and x.startswith("\n") == False]
    tag_ano = []
    for j, i in enumerate(lines):
        if i.startswith('(') :
            if len(i) > 32 and i[31] == ')':            
                tag_ano.append(i[36:])
            else:
                tag_ano.append(i[20:])
        elif i.startswith('# ('):
            tag_ano.append(i[22:])

    new_tag = st.multiselect('Champs à ne pas anonymiser :', tag_ano, help='TODO')           #les champs à ne pas anonymiser

    prefix = st.text_input("(Optionnel) Entrer un préfixe d'anonymisation (par ex. NOM_DU_PROJET_):", help='TODO')

    arborescence = st.checkbox("Créer une arborescence par série", help='TODO')

    reanonymisation = st.checkbox("Réanonymiser ce qui a déjà été anonymisé", help='TODO')

#    if ' Study Date\n' in new_tag :
    conserver_annee = st.checkbox("Conserver l'année", key='ok', help='TODO')
 #   else :
  #      conserver_annee = st.checkbox("Conserver l'année (avec le champ 'Study date')", disabled=True, key='non', help='TODO')


    for i, j in enumerate(active_lines):
        for l in new_tag :
            if l in j:
                active_lines[i] = '#' + j

    with open('new_champs_ano.txt', 'w') as fichier:
        for i in active_lines:
            fichier.write(i)

    st.subheader("2) Choisir le dossier des DICOM à anonymiser")

    input_folder_path = st_file_selector(label='Entrez le chemin du dossier', clef='input')
    if input_folder_path:
        if os.path.isdir(input_folder_path):
            files = os.listdir(input_folder_path)
            st.write(f'Le dossier contient {len(files)} fichiers DICOM')
        else:
            st.write(input_folder_path, 'n\'est pas un dossier.')

    st.subheader("3) Choisir le dossier de sortie des DICOM anonymisés")

    output_folder_path = st_file_selector(label='Entrez le chemin du dossier de sortie', clef='output')
    if output_folder_path:
        if os.path.isdir(output_folder_path):
            files = os.listdir(output_folder_path)
            st.write(f'Le dossier contient déjà {len(files)} fichiers')
        else:
            st.write(input_folder_path, 'n\'est pas un dossier.')

    st.subheader("4) Exporter un résumé des séries DICOM en CSV")

    if st.button("Exporter", help='Crée un fichier \'liste_dicom.csv\' modifiable contenant une table d\'anonymisation basée sur le préfixe donné à l\'étape 1'):
        export_list_anonymisation(prefix, input_folder_path, output_folder_path)
    
    file_path = os.path.join(output_folder_path, 'liste_dicom.csv')
    file_path_backup = os.path.join(output_folder_path, 'liste_dicom_backup.csv')
    if os.path.exists(file_path):
        index = []
        df = pd.read_csv(file_path, delimiter=';')
        edited_df = st.data_editor(df, num_rows='dynamic')
        filter = st.multiselect('Filtrer par :', edited_df.columns, help='Choisir le nom des colonnes à filtrer')
        for categories in filter :
            column_filter = st.multiselect(f'Filtrage de {categories} :', edited_df[filter][categories].unique(), help="Sélectionner les valeurs à garder")
            for i in column_filter :
                for j in range (len(edited_df)):
                    if edited_df[categories][j] == i:
                        index.append(j)
        edited_df.drop(index, axis=0, inplace=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Backup", help="Enregistre le tableur avant modificiation dans 'liste_dicom_backup.csv'."):
                df.to_csv(file_path_backup, index=False, sep=';')
                st.success("Version originale enregistée avec succès dans liste_dicom_backup.csv")
        with col2:
            if st.button("Enregistrer les modifications", help="Le nouveau tableur sera enregistré dans 'liste_dicom.csv'."):
                df = edited_df
                edited_df.to_csv(file_path, index=False, sep=';')
                st.success("Modifications enregistrées avec succès dans 'liste_dicom.csv'.")
                st.write("Données mises à jour")

    st.subheader("5) Cliquer sur Anonymiser pour commencer l'anonymisation")

    if st.button("Anonymiser", help="Appuyer pour démarrer l'anonymisation avec les paramètres définis plus tôt"):
        deidentification(input_folder_path, output_folder_path, arborescence, reanonymisation, conserver_annee)

# Page 2
def page2():
    st.title("NIfTI Convertor")
    st.header("*Interface graphique du logiciel de conversion des images DICOM et segmentations OFF en Nifti*")
    st.subheader("1) Réglages et options")
    reconv_img = st.checkbox("Convertir à nouveau les images DICOM déjà converties en NIFTI", help='TODO')

    reconv_seg = st.checkbox("Convertir à nouveau les segmentations OFF déjà converties en NIFTI", help='TODO')

    st.subheader("2) Entrer les adresses suivantes :")


    dbjson_file = st.file_uploader("Sélectionner un fichier db.json", type=["json"], help='TODO')

    if dbjson_file is not None:
        # Charger le contenu du fichier JSON
        content = dbjson_file.read()
        try:
            data = json.loads(content)
            st.success("Le fichier JSON a été chargé avec succès !")
                        
        except json.JSONDecodeError:
            st.error("Erreur lors du chargement du fichier JSON.")


    st.markdown("**Dossier parent des images DICOM**")

    input_folder_dicom = st_file_selector(clef='input_folder_dicom', label = 'Entrez le chemin du dossier')
    if input_folder_dicom :
        if os.path.isdir(input_folder_dicom):
            files = os.listdir(input_folder_dicom)
            st.write(f'Le dossier contient {len(files)} fichiers DICOM')
        else:
            st.write("Le dossier n'existe pas")

    st.markdown("**Dossier parent des segmentations OFF**")

    input_folder_segm = st_file_selector(clef = 'input_folder_segm', label = 'Entrez le chemin du dossier')
    if input_folder_segm:
        if os.path.isdir(input_folder_segm):
            files = os.listdir(input_folder_segm)
            st.write(f'Le dossier contient {len(files)} segmentations OFF')
        else:
            st.write("Le dossier n'existe pas")

    st.markdown("**Dossier de sortie des NIFTI**")

    output_folder_nifiti = st_file_selector(clef = 'output_folder_nifti', label = 'Entrez le chemin du dossier')
    if output_folder_nifiti:
        if os.path.isdir(output_folder_nifiti):
            files = os.listdir(output_folder_nifiti)
            st.write(f'Le dossier contient {len(files)} fichiers')
        else:
            st.write("Le dossier n'existe pas")


    st.subheader('3) Exporter un résumé des segmentations en CSV')
    if st.button('Exporter', help='Crée un fichier \"liste_segmentations.csv\" modifiable contenant un résumé des segmentations et fichiers utilisés pour la conversion'):
        export_list_segmentations(input_folder_dicom, dbjson_file, data, output_folder_nifiti)

    file_path = os.path.join(output_folder_nifiti, 'liste_segmentations.csv')
    file_path_backup = os.path.join(output_folder_nifiti, 'liste_segmentations_backup.csv')

    col1, col2 = st.columns(2)
    if os.path.exists(file_path):
        index = []
        df = pd.read_csv(file_path, delimiter=';')
        edited_df = st.data_editor(df, num_rows='dynamic')
        filter = st.multiselect('Filtrer par :', edited_df.columns, help='Choisir le nom des colonnes à filtrer')
        for categories in filter :
            column_filter = st.multiselect(f'Filtrage de {categories} :', edited_df[filter][categories].unique(), help="Sélectionner les valeurs à garder")
            for i in column_filter :
                for j in range (len(edited_df)):
                    if edited_df[categories][j] == i:
                        index.append(j)
        edited_df.drop(index, axis=0, inplace=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Backup", help="Enregistre le tableur avant modificiation dans 'liste_segmentations_backup.csv'."):
                df.to_csv(file_path_backup, index=False, sep=';')
                st.success("Version originale enregistée avec succès dans liste_segmentations_backup.csv")
        with col2:
            if st.button("Enregistrer les modifications", help="Le nouveau tableur sera enregistré dans 'liste_segmentations.csv'."):
                df = edited_df
                edited_df.to_csv(file_path, index=False, sep=';')
                st.success("Modifications enregistrées avec succès dans 'liste_segmentations.csv'.")
                st.write("Données mises à jour")



    st.subheader('4) Convertir les images DICOM en NIFTI')
    if st.button('Convertir', key='dicom', help='TODO'):
        imgDicomToNiftiConversion(input_folder_dicom, output_folder_nifiti, reconv_img)

    st.subheader('5) Convertir les segmentations OFF en NIFTI')
    if st.button('Convertir', key = 'off', help='TODO'):
        segOffToNiftiConversion(input_folder_segm, output_folder_nifiti, reconv_seg)

    st.subheader('6) Affichage des fichiers NIfTI')
    output = lister_dossiers_dossier(output_folder_nifiti)
    nifti_files = []
    if output is not None:
        nifti_files = lister_noms_fichiers_dossier(os.path.join(output_folder_nifiti, output))
    if nifti_files != [] :
        seqtime = nifti_files[0]
        if "_i" in seqtime :
            seqtime = seqtime.split("_i")[0]
        elif "_t" in seqtime:
            seqtime = seqtime.split("_t")[0]
    if st.button('Enregistrer les images en pdf', key = 'pdf', help='TODO'):
        store_pdf_segmentations_WM(seqtime, nifti_files)
        st.success('Le pdf a bien été enregistré')

# Page 3
def page3():
    st.title("PyRadiomics Extractor")
    st.header("*Interface graphique du logiciel d'extraction des paramètres de radiomique avec Pyradiomics*")
    st.subheader("1) Réglages et options")
    reextraction = st.checkbox("Réextraire ce qui a déjà été extrait")

    st.subheader("2) Choisir le dossier parent des images et masques Nifti")

    input_nifti = st_file_selector(clef='input_nifti', label = 'Entrez le chemin du dossier')
    if input_nifti :
        if os.path.isdir(input_nifti):
            files = os.listdir(input_nifti)
            st.write(f'Le dossier contient {len(files)} fichiers')
        else:
            st.write("Le dossier n'existe pas")

    st.subheader("3) Choisir le dossier de sortie des paramètres extraits")

    output_folder = st_file_selector(clef = 'output_folder', label = 'Entrez le chemin du dossier')
    if output_folder:
        if os.path.isdir(output_folder):
            files = os.listdir(output_folder)
            st.write(f'Le dossier contient {len(files)} fichiers')
        else:
            st.write("Le dossier n'existe pas")


    st.subheader("4) Choisir le fichier de prétraitement YAML")

    # Sélection du fichier YAML
    default = st.button('Choisir les paramètres par défaut')        
    yaml_file = st.file_uploader("Sélectionnez un fichier YAML", type=["yaml", "yml"])
    if default :
        with open('params_par_defaut.yaml', 'r') as f:
            content_yaml = f.read()
        try:
            st.success("Le fichier 'params_par_defaut.yaml' a été chargé avec succès !")            
        except :
            st.error("Erreur lors du chargement du fichier yaml.")

    if yaml_file is not None:
        # Charger le contenu du fichier yaml
        content_yaml = yaml_file.read()
        try:
            st.success("Le fichier YAML a été chargé avec succès !")
                        
        except :
            st.error("Erreur lors du chargement du fichier yaml.")

    if input_nifti is not None :
        Extractor = featureextractor.RadiomicsFeatureExtractor()

    st.subheader('5) Personnaliser le prétraitement yaml')


    

    with st.expander("Paramétrer"):
        ExtractorParams = getParamsFromFile(Extractor)
        ExtractorTemp = copy.deepcopy(Extractor)
        CustomImageTypes = ExtractorTemp.enabledImagetypes

        st.title("Paramétrage de l'extraction")
        
        # Réglage 1 : Filtres

        st.subheader("Filtres")

        ## Original
        original = st.checkbox('Original', value=True)
        original_val = "original_on" if original else "original_off"
        changeImageTypes(original_val, CustomImageTypes, None, None, None, None, None)



        ## Wavelet 
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.write('')
            st.write('')
            st.write('')
            wavelet = st.checkbox('Wavelet')
        if wavelet:
            with col2:
                option = ["coif", "dmey", "haar", "sym", "db", "bior", "rbio"]
                selected_options = st.selectbox('Type :', option)
            with col3:
                valeur = st.text_input("Valeur :", value="1")
            with col4:
                start = st.text_input("Start :", value="0")
            with col5:
                level = st.text_input("Level :", value="1")
        else :
            with col2:
                option = ["coif", "dmey", "haar", "sym", "db", "bior", "rbio"]
                selected_options = st.selectbox('Type :', option, disabled = True)
            with col3:
                valeur = st.text_input("Valeur :", value="1", disabled = True)
            with col4:
                start = st.text_input("Start :", value="0", disabled = True)
            with col5:
                level = st.text_input("Level :", value="1", disabled = True)

        wavelet_val = "wavelet_on" if wavelet else "wavelet_off"
        changeImageTypes(wavelet_val, CustomImageTypes, None, level, start, valeur, selected_options)

        ## LoG
        with col1 :
            st.write('')
            st.write('')
            log = st.checkbox('LoG')
        if log:
            with col2:
                sigma = st.text_input("Sigma :", value="1.0")
        else :
            with col2:
                sigma = st.text_input("Sigma :", value ="1.0", disabled = True)
        log_val = "log_on" if log else "log_off"
        changeImageTypes(log_val, CustomImageTypes, sigma, level, start, valeur, selected_options)
    
        # Réglage 2 : Paramètres

        st.subheader('Paramètres')

        col1_bis, col2, col3, col4, col5 = st.columns(5)


        with col1_bis:
            shape_3D = st.checkbox('Shape3D', value=True)
            first_order = st.checkbox('Firstorder', value=True)

        with col2:
            shape_2D = st.checkbox('Shape2D')

        col1_bis, col2, col3, col4, col5 = st.columns(5)


        with col1_bis:
            GLCM = st.checkbox('GLCM', value=True)

        with col2:
            GLRLM = st.checkbox('GLRLM', value=True)

        with col3:
            GLSZM = st.checkbox('GLSZM', value=True)

        with col4:
            GLDM = st.checkbox('GLDM', value=True)   

        with col5:
            NGTDM = st.checkbox('NGTDM', value=True)  

        ftShape3D = "shape" if shape_3D else "disabled"
        ftShape2D = "shape2D" if shape_2D else "disabled"
        ftFirstOrder = "firstorder" if first_order else "disabled"
        ftGLCM = "glcm" if GLCM else "disabled"
        ftGLRLM = "glrlm" if GLRLM else "disabled"
        ftGLSZM = "glszm" if GLSZM else "disabled"
        ftGLDM = "gldm" if GLDM else "disabled"
        ftNGTDM = "ngtdm" if NGTDM else "disabled"

        changeExtractorFtClasses(ExtractorTemp, ftShape3D)
        changeExtractorFtClasses(ExtractorTemp, ftShape2D)
        changeExtractorFtClasses(ExtractorTemp, ftFirstOrder)
        changeExtractorFtClasses(ExtractorTemp, ftGLCM)
        changeExtractorFtClasses(ExtractorTemp, ftGLRLM)
        changeExtractorFtClasses(ExtractorTemp, ftGLSZM)
        changeExtractorFtClasses(ExtractorTemp, ftGLDM)
        changeExtractorFtClasses(ExtractorTemp, ftNGTDM)
           
        # Réglage 3 : Discrétisation des niveaux de gris
        st.subheader('Discrétisation des niveaux de gris')
        discretisation = st.radio('Type de discrétisation :', ["Fixed Bin Size", "Fixed Bin Number"], index=["Fixed Bin Size", "Fixed Bin Number"].index("Fixed Bin Size"))
        if discretisation == "Fixed Bin Size":
            bin_size = st.text_input('Largeur des niveaux de gris :', value=25)
            bin_number = None
        elif discretisation == 'Fixed Bin Number':
            bin_number = st.text_input('Nombre de niveaux de gris', value=25)        
            bin_size = None

        switchLabelDNG(discretisation, ExtractorTemp, ExtractorParams, bin_size, bin_number)

        # Réglage 4 : Normalisation des intensités
        st.subheader('Normalisation des intensités')

        standardisation = st.checkbox('Standardisation des intensités (z-score)', value=True)
        if standardisation:
            outlier = st.checkbox('Retrait des outliers')
            value_outlier = st.text_input('', value='3')
            switchNormalization(ExtractorTemp)
            if outlier:
                switchOutliers(value_outlier, ExtractorTemp)

        else :
            st.checkbox('Retrait des outliers', disabled=True)
            st.text_input('', value='3', disabled=True)
        # Réglage 5 : Interpolation des images
        st.subheader('Interpolation des images')

        interpolation = st.checkbox('Interpolation', value=True)
        interpol_option = ["sitkNearestNeighbor", "sitkLinear", "sitkBSpline", "sitkGaussian", "sitkLabelGaussian", "sitkHammingWindowedSinc", "sitkCosineWindowedSinc", "sitkWelchWindowedSinc", "sitkLanczosWindowedSinc", "sitkBlackmanWindowedSinc"]
        if interpolation :
            switchInterpolation(ExtractorTemp)
            selected_interpol_options = st.selectbox("", interpol_option)
            values_interpol = st.text_input('Valeur en [x, y, z] :', value='[1, 1, 1]')
        else :
            selected_interpol_options = st.selectbox("", interpol_option, disabled = True)
            values_interpol = st.text_input('Valeur en [x, y, z] :', value='[1, 1, 1]', disabled = True)

        validation = st.button('Valider')
        if validation:
            validate_preprocessing(CustomImageTypes, ExtractorTemp, ExtractorParams, log_val, sigma, wavelet_val, level, start, value_outlier, discretisation, bin_size, bin_number, valeur, selected_options)

# # # # # valeur en x y z 
# # # # # interpolation option
# # # # # param par defaut chargé dans le params_par_defaut.yaml ??


    st.subheader('6) Extraire les paramètres de radiomique')
    st.write('Crée un fichier \"radiomics_features.csv\" contenant les paramètres de radiomique')
    if st.button('Extraire', key = 'param'):
        extract_radiomics_features(input_nifti, content_yaml, reextraction, output_folder)
    file_path = os.path.join(output_folder, 'radiomics_features.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, delimiter=';')
        edited_df = st.data_editor(df, num_rows='dynamic')
        if st.button("Enregistrer les modifications"):
            df = edited_df
            edited_df.to_csv(file_path, index=False, sep=';')
            st.success("Modifications enregistrées avec succès dans 'radiomics_features.csv'.")
            st.write("Données mises à jour")



#### FONCTIONS NIFTI CONVERTOR ####


def trouver_longueur_dictionnaire(dictionnaire):
    if isinstance(dictionnaire, dict):
        if 'segmentations' in dictionnaire:
            return len(dictionnaire['segmentations'])
        else:
            for valeur in dictionnaire.values():
                resultat = trouver_longueur_dictionnaire(valeur)
                if resultat is not None:
                    return resultat
    elif isinstance(dictionnaire, list):
        for element in dictionnaire:
            resultat = trouver_longueur_dictionnaire(element)
            if resultat is not None:
                return resultat
    return None

# Exportation du fichier liste_segmentations.csv résumant les segmentations trouvées
def export_list_segmentations(dicom_input, dbjson_file, data, output):

    # Début du chronométrage
    startTime = time.time()

    # Création ou remise à zéro du fichier de log
    open('..\log_export.txt', 'w', encoding="utf-8").close()

    # Vérification : l'adresse du fichier db.json est-elle bien spécifié ?
    if len(dicom_input) == 0:
        st.write("Dossier DICOM non trouvé, vous devez sélectionner le dossier parent des fichiers DICOM (Étape 2)")
        return
    elif dbjson_file is None:
        st.write("Fichier db.json non trouvé, vous devez sélectionner un fichier db.json valide (Étape 2)")
        return

    # Initialisation de la barre de progression
    progress_bar = st.progress(0)
    ProgbarSet = 0
    # Nombre de segmentations :

    segNumber = trouver_longueur_dictionnaire(data)

    segNumber = 0.1 if segNumber == 0 else segNumber


    # Extraction des données du fichier db.json
    listSegmentations = {}
    for p, v in data.items():
        try:
            for annot in v["annotations"] :
                try:
                    numAno = p.replace(" ", "")
                    annotId = annot["id"] if len(annot["id"]) != 0 else "_"
                    for seg in annot["segmentations"]:
                        try:
                            maskName = seg["id"] if len(seg["id"]) != 0 else "_"
                            offFileName = seg["url"].split('/')[-1]
                            imRefId = seg["image_ref_id"]
                            imgList = [img["url"] for img in v["images"] if img["id"] == imRefId][0]
                            seriesDescription = [img["comments"] for img in v["images"] if img["id"] == imRefId][0]
                            seriesOutPut = "_".join(re.sub(r"[^a-zA-Z0-9]", " ", seriesDescription).upper().split())

                            listSegmentations[p, annotId, offFileName] = [numAno, annotId, offFileName, maskName, imRefId, seriesDescription, seriesOutPut, imgList]

                            ProgbarSet = 1 / segNumber
                            progress_bar.progress(ProgbarSet)
                            time.sleep(0.1)

                        except Exception as g:
                            with open('..\log_export.txt', 'a', encoding="utf-8") as log_export:
                                log_export.write(f"Failed to read {seg}: {g} \n")
                                st.write('Failed to read', seg, ':', g)
                            ProgbarSet = 1 / segNumber
                            progress_bar.progress(ProgbarSet)
                            time.sleep(0.1)
                            print(f"g : {ProgBarSet}")
                            continue

                except Exception as f:
                    with open('..\log_export.txt', 'a', encoding="utf-8") as log_export:
                        log_export.write(f"Failed to read {annot}: {f} \n")
                        st.write('Failed to read', annot, ':', f)
                    ProgbarSet += 1 / segNumber
                    progress_bar.progress(ProgbarSet)
                    time.sleep(0.1)
                    continue

        except Exception as e:
            with open('..\log_export.txt', 'a', encoding="utf-8") as log_export:
                log_export.write(f"Failed to read {p}: {e} \n")
                st.write('Failed to read', p, ':', e)
            ProgbarSet += 1 / segNumber
            progress_bar.progress(ProgbarSet)
            time.sleep(0.1)
            continue

    # Ajout de SeriesInstanceUID :
    listDcm = []
    for (dirPath, _, fileNames) in os.walk(dicom_input):
        listDcm += [os.path.join(dirPath, filename).replace("\\\\", "/").replace("\\", "/") for filename in fileNames if (".dcm" in filename) or (is_file_a_dicom(os.path.join(dirPath, filename)) == True)]
    
    for k, v in listSegmentations.items():
        try:
            img1 = v[-1][0]
            img1Path = [i for i in listDcm if img1 in i][0] if any(img1 in i for i in listDcm) else "NA"
            if (img1Path != "NA") and (os.path.exists(img1Path)):
                file_reader = ImageFileReader()
                file_reader.SetFileName(img1Path)
                file_reader.ReadImageInformation()
                series_UID = file_reader.GetMetaData('0020|000e')
                seriesInstanceUID_formatted = series_UID.replace(".", "")[-1:2:-2]
            else:
                seriesInstanceUID_formatted = ""
            v.append(f"s{seriesInstanceUID_formatted}")     
        except Exception as e:
            with open('..\log_export.txt', 'a', encoding="utf-8") as log_export:
                log_export.write(f"Failed to add SeriesInstanceUID for {k} \n")
                st.write("Failed to add SeriesInstanceUID for", k)
            ProgBarSet += 1 / segNumber
            progress_bar.progress(ProgbarSet)
            time.sleep(0.1)
            continue            

    file_path = os.path.join(output, 'liste_segmentations.csv')


    # Enregistrement du fichier csv contenant la liste des segmentations
    with open(file_path, mode='w', newline='', encoding="utf-8") as csv_file:
        col_names = ["AnonymizationID",
                    "AnnotID",
                    "OffFileName",
                    "MaskName",
                    "ImageRefId",
                    "SeriesDescription",
                    "SeriesOutput",
                    "ImgList",
                    "SeriesInstanceUID"
                    ]
        writer = csv.writer(csv_file, delimiter=';')

        writer.writerow(col_names)
        for data in listSegmentations.values():
            writer.writerow(data)

    # Fin d'exécution de la fonction
    progress_bar.progress(100)
    endTime = time.time()
    executionDuration = int(endTime - startTime)
    st.write(f'Opération terminée en {executionDuration} secondes.')




def store_pdf_segmentations_WM(seqtime, nifti_files):
    with PdfPages("Visualisation_" + seqtime + ".pdf") as pdf:
        for pt in nifti_files:
            try:
                nr = 4
                nc = 6
                data = nb.load(pt)
                img = data.get_fdata()
                fig, ax = plt.subplots(nrows=nr, ncols=nc, figsize=(20,20), sharex=True, sharey=True, gridspec_kw={'hspace': 0})
                fig.suptitle(pt, fontsize = 20, color='white')
                fig.patch.set_facecolor('#000000')  
                bbox = bbox_3D(img)
                k = nr*nc
                ls = np.int64(np.ceil(np.linspace(bbox[2], bbox[3], k)))
                for i in range(nr):
                    for j in range(nc):
                        kp = i*nc + j
                        ax[i,j].imshow(np.rot90(img[bbox[0]:bbox[1],ls[kp],bbox[4]:bbox[5]]), cmap="gray")
                plt.tight_layout(rect=[0, 0.03, 1, 0.99])
                pdf.savefig(fig)
                plt.close(fig)
                print(f"{pt} ... success")
            except:
                print(f"{pt} ... error")





def bbox_3D(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax

# Conversion des images Dicom en Nifti pour les segmentations présentes dans le fichier liste_segmentations.csv
def imgDicomToNiftiConversion(dicom_input, nifti_output, reconv_img):
    '''
    Fonction permettant la conversion des images DICOM en Nifti

    '''

    # Début du chronométrage
    startTime = time.time()

    # Vérification : le dossier de sortie est-il bien spécifié, différent du dossier d'entrée, et valide ?
    if len(dicom_input) == 0:
        st.write("Dossier DICOM non trouvé, vous devez sélectionner le dossier parent des fichiers DICOM (Étape 2)")
        return
    if len(nifti_output) == 0:
        st.write("Dossier de sortie non trouvé, vous devez sélectionner le dossier de sortie des fichiers Nifti (Étape 2)")
        return
    elif nifti_output == dicom_input :
        st.write("Dossier Nifti de sortie non valide, vous devez sélectionner un dossier de sortie différent du dossier d'entrée (Étape 2)")
        return
    elif not os.path.exists(nifti_output):
        try:
            os.mkdir(nifti_output)
        except:
            st.write("Dossier Nifti de sortie non valide, vous devez sélectionner un dossier de sortie valide (Étape 3)")
            return

    # Lecture du fichier liste_segmentations
    cle = defaultdict(list)

    try:
        with open("..\liste_segmentations.csv", 'r', encoding="utf-8") as data:
            for line in csv.DictReader(data, delimiter=';'):
                for k, v in line.items():
                    cle[k].append(v)
    except:
        st.write("Fichier liste_segmentations.csv non trouvé, vous devez d'abord exporter un fichier liste_segmentations.csv (Etape 3)")
        return
    
    # Initialisation de la barre de progression
    ProgBarSet = 0
    ProgBarStep = 1 / len(cle["AnonymizationID"])
    progress_bar = st.progress(ProgBarSet)

    # Création ou remise à zéro du fichier de log
    open('..\log_imgConversion.txt', 'w', encoding="utf-8").close()

    # Liste des fichiers Dicom
    listDcm = []
    for (dirPath, _, fileNames) in os.walk(dicom_input):
        listDcm += [os.path.join(dirPath, filename).replace("\\\\", "/").replace("\\", "/") for filename in fileNames if (".dcm" in filename) or (is_file_a_dicom(os.path.join(dirPath, filename)) == True)]

    dicomImgUnique = []

    for i, imgPathList in enumerate(cle["ImgList"]):
        # Vérification : si choix de ne pas reconvertir à l'étape 1,  l'image convertie existe-t-elle déjà ?
        outputDir = os.path.join(nifti_output, cle["AnonymizationID"][i])
        numAno = cle["AnonymizationID"][i]
        seriesOutput = cle["SeriesOutput"][i]
        seriesInstanceUID_formatted = cle["SeriesInstanceUID"][i][1:]
        outputFileName = f"{numAno}_{seriesOutput}_{seriesInstanceUID_formatted}_img.nii.gz"

        if (reconv_img == 0) and (os.path.exists(outputDir + "/" + outputFileName)):
            ProgBarSet += ProgBarStep
            progress_bar.progress(ProgBarSet)
            time.sleep(0.1)
            continue   

        # Vérification que l'image n'a pas déjà été convertie dans la boucle (car le fichier liste_segmentations.csv contient une ligne par segmentation)
        if imgPathList not in dicomImgUnique:
            dicomImgUnique.append(imgPathList)

            try:
                imgPathList = list(imgPathList.strip('][').replace("'", "").split(', '))[0]
                dicomImgPath = [i for i in listDcm if imgPathList in i][0]

                # Lecture des métadonnées d'un fichier permettant de trouver le tag Series Instance UID propre à chaque série
                dcmFolder = '/'.join(dicomImgPath.split('/')[:-1])
                file_reader = ImageFileReader()
                file_reader.SetFileName(os.path.join(dicomImgPath))
                file_reader.ReadImageInformation()
                series_UID = file_reader.GetMetaData('0020|000e')  # Tag "Series Instance UID"

                # Liste de tous les fichiers DICOM mis dans l'ordre selon SliceLocation
                sorted_file_names = ImageSeriesReader.GetGDCMSeriesFileNames(dcmFolder, series_UID)

                # # Création et renvoi d'une image 3D
                # img = ReadImage(sorted_file_names)

                # Sauvegarde de l'image en Nifti
                if not os.path.exists(outputDir):
                    os.makedirs(outputDir)

                if not os.path.exists(".\Temp"):
                    os.makedirs(".\Temp")

                for file in sorted_file_names:
                    filename = Path(file).name
                    shutil.copyfile(file, f".\Temp/{filename}")

                try:
                    dicom2nifti.dicom_series_to_nifti(".\Temp", f"{outputDir}/{outputFileName}", reorient_nifti=True)
                    # WriteImage(img, outputDir + "/" + outputFileName)
                except BaseException as be:
                    with open("..\log_imgConversion.txt", "a", encoding="utf-8") as floggB:
                        floggB.write(f"Writing error: {be} \n")
                        st.write('Writing error :', be)
                
                shutil.rmtree(".\Temp", ignore_errors = True)

                ProgBarSet += ProgBarStep
                progress_bar.progress(ProgBarSet)
                time.sleep(0.1)
            
            except Exception as e:
                maskName = cle["OffFileName"][i]
                with open("..\log_imgConversion.txt", "a", encoding="utf-8") as flogg:
                    flogg.write(f"{maskName} error: {e} \n")
                    st.write(maskName, "error:", e)
                ProgBarSet += ProgBarStep
                progress_bar.progress(ProgBarSet)
                time.sleep(0.1)
                continue

        else:
            ProgBarSet += ProgBarStep
            progress_bar.progress(ProgBarSet)
            time.sleep(0.1)

    # Opération terminée : affichage de l'information
    progress_bar.progress(100)
    endTime = time.time()
    executionDuration = int(endTime - startTime)
    st.write(f'Opération terminée en {executionDuration} secondes.')


# Conversion des segmentations OFF en Nifti pour les segmentations présentes dans le fichier liste_segmentations.csv
def segOffToNiftiConversion(off_input, nifti_output, reconv_seg):
    '''
    Fonction permettant la conversion des segmentations OFF en Nifti

    '''

    # Début du chronométrage
    startTime = time.time()

    # Vérification : le dossier de sortie est-il bien spécifié, différent du dossier d'entrée, et valide ?
    if len(off_input) == 0:
        st.write("Dossier des segmentations non trouvé, vous devez sélectionner le dossier parent des segmentations OFF (Étape 2)")
        return
    if len(nifti_output) == 0:
        st.write("Dossier de sortie non trouvé, vous devez sélectionner le dossier de sortie des fichiers Nifti (Étape 2)")
        return

    # Lecture du fichier liste_segmentations
    cle = defaultdict(list)

    try:
        with open("..\liste_segmentations.csv", 'r', encoding="utf-8") as data:
            for line in csv.DictReader(data, delimiter=';'):
                for k, v in line.items():
                    cle[k].append(v)
    except:
        st.write("Fichier liste_segmentations.csv non trouvé, vous devez d'abord exporter un fichier liste_segmentations.csv (Etape 3)")
        return

    # Initialisation de la barre de progression
    ProgBarSet = 0
    ProgBarStep = 1 / len(cle["AnonymizationID"])
    progress_bar = st.progress(ProgBarSet)

    # Création ou remise à zéro du fichier de log
    open('..\log_segConversion.txt', 'w', encoding="utf-8").close()

    # Liste des fichiers OFF
    listOff = []
    for (dirPath, _, fileNames) in os.walk(off_input):
        listOff += [os.path.join(dirPath, filename).replace("\\\\", "/").replace("\\", "/") for filename in fileNames if (".off" in filename)]

    # Pour chaque ligne du fichier liste_segmentations.csv :
    for i, offName in enumerate(cle["OffFileName"]):

        # Création de l'adresse de sortie de la segmentation convertie
        outputDir = os.path.join(nifti_output, cle["AnonymizationID"][i])
        offFileName = cle["MaskName"][i]
        numAno = cle["AnonymizationID"][i]
        seriesOutput = cle["SeriesOutput"][i]
        seriesInstanceUID_formatted = cle["SeriesInstanceUID"][i][1:]
        outputSegName = f"{numAno}_{seriesOutput}_{seriesInstanceUID_formatted}_{offFileName}_mask.nii.gz"
        segNiftiPath = outputDir + "/" + outputSegName

        # Vérification : la segmentation existe-t-elle déjà ? Si oui et ReconversionSeg == 0 : on passe
        if (reconv_seg == 0) and (os.path.exists(segNiftiPath)):
            ProgBarSet += ProgBarStep
            progress_bar.progress(ProgBarSet)
            time.sleep(0.1)
            continue

        # Recherche du fichier OFF correspondant sur l'ordinateur :
        offFilePath = [x for x in listOff if offName in x]
        if len(offFilePath) == 0:
            with open("..\log_segConversion.txt", "a", encoding="utf-8") as flogg:
                    flogg.write(f"{offName} : fichier non trouvé \n")
                    st.write(offName, ": fichier non trouvé")
            ProgBarSet += ProgBarStep
            progress_bar.progress(ProgBarSet)
            time.sleep(0.1)
            continue
        else:
            offFilePath = offFilePath[0]

        # Recherche de l'image Nifti correspondante sur l'ordinateur :
        outputFileName = f"{numAno}_{seriesOutput}_{seriesInstanceUID_formatted}_img.nii.gz"
        imgNiftiPath = outputDir + "/" + outputFileName
        if not os.path.exists(imgNiftiPath):
            with open("..\log_segConversion.txt", "a", encoding="utf-8") as flogg:
                    flogg.write(f"{imgNiftiPath} : fichier non trouvé \n")
                    st.write(imgNiftiPath, ": fichier non trouvé")
            ProgBarSet += ProgBarStep
            progress_bar.progress(ProgBarSet)
            time.sleep(0.1)
            continue

        try:
            if "coro" in offName.lower():
                off2nii_manual_coro(imgNiftiPath, offFilePath, segNiftiPath)
            elif "sag" in offName.lower():
                off2nii_manual_coro(imgNiftiPath, offFilePath, segNiftiPath)
            else:
                off2nii_manual(imgNiftiPath, offFilePath, segNiftiPath)

            ProgBarSet += ProgBarStep
            progress_bar.progress(ProgBarSet)
            time.sleep(0.1)
        
        except Exception as e:
            with open("..\log_segConversion.txt", "a", encoding="utf-8") as flogg:
                flogg.write(f"{offName} off2nii conversion error: {e} \n")
                st.write(offName, "off2nii conversion error: ", e)

            ProgBarSet += ProgBarStep
            progress_bar.progress(ProgBarSet)
            time.sleep(0.1)
            continue


    # Opération terminée : affichage de l'information et déblocage du bouton
    progress_bar.progress(100)
    endTime = time.time()
    executionDuration = int(endTime - startTime)
    st.write(f'Opération terminée en {executionDuration} secondes.')
    return


#### FONCTIONS PYRADIOMICS EXTRACTOR ####


def getParamsFromFile(Extractor):
    ExtractorParams = {}
    ExtractorParams["imageTypes"] = [it.lower() for it, ck in six.iteritems(Extractor.enabledImagetypes)]
    ExtractorParams["featureClasses"] = [f"{cls.lower()}_ftclass" for cls, ft in six.iteritems(Extractor.enabledFeatures)]
    ExtractorParams["dngType"] = "binCount" if Extractor.settings.get('binCount') is not None else "binWidth"
    
    return ExtractorParams



def extract_radiomics_features(input_nifti, yaml, reextraction, output):
    """
    Extraction des paramètres de radiomique utilisant le fichier de prétraitement fourni à l'étape 3
    et générant ou complétant un fichier "radiomics_features.csv" dans le dossier du logiciel.
    """
    startTime = time.time()

    ### 1.Extraction des métadonnées DICOM de chaque série à partir de l'ensemble des fichiers
    # Se base sur "SeriesInstanceUID" pour trouver tous les fichiers communs à chaque série

    # Vérification 1 : le dossier d'entrée est-il bien spécifié ?
    if len(input_nifti) == 0:
        st.write("Dossier NIFTI non trouvé", "Vous devez sélectionner le dossier contenant les images et masques NIFTI "
                                                "(Étape 2)")
        return
    elif not os.path.exists(input_nifti):
        st.write("Dossier NIFTI non trouvé",
                    "Vous devez sélectionner un dossier contenant les fichiers NIFTI valide"
                    " (Étape 2)")
        return

    # Vérification 2 : le fichier de prétraitement est-il bien spécifié ?
    if yaml is None:
        st.write("Fichier YAML non trouvé", "Vous devez sélectionner le fichier de paramétrage du prétraitement YAML "
                                                "(Étape 3)")
        return
    elif len(yaml) == 0:
        st.write("Fichier YAML non trouvé",
                    "Vous devez sélectionner un fichier de paramétrage du prétraitement YAML valide"
                    " (Étape 3)")
        return

    # Initialisation de la barre de progression
    ProgBarSet = 0
    progress_bar = st.progress(ProgBarSet)

    open('..\log_extract.txt', 'w', encoding="utf-8").close()

    listMasks = []
    
    for (dirPath, dirNames, fileNames) in os.walk(input_nifti):
        listMasks += [os.path.join(dirPath, filename).replace("\\\\", "/").replace("\\","/") for filename in fileNames if ("_mask.nii.gz" in filename)]

    if input_nifti is not None :
        Extractor = featureextractor.RadiomicsFeatureExtractor()

    # Liste des paramètres activés :
    enabledFtList = []
    for cls, ft in six.iteritems(Extractor.enabledFeatures):
        if ft is None or len(ft) == 0:
            featureClasses = getFeatureClasses()
            ft = [f for f, deprecated in six.iteritems(featureClasses[cls].getFeatureNames()) if not deprecated]
        for f in ft:
            enabledFtList.append(f"{cls}_{f}")

    # Initialisation du dictionnaire des paramètres :
    features = {}
    cles = ["patient", "mask", "maskPath", "imgPath", "dngType", "dngBin", "intensityNormalization", "outliersRemoval"]

    # Si un fichier radiomics_features.csv existe et que l'option de réextraction est désactivée (Etape 1), on lit le fichier :
    file_path = os.path.join(output, 'radiomics_features.csv')
    if (reextraction == 0) and os.path.exists(file_path):
        with open(file_path, 'r', newline = '', encoding="utf-8") as csv_file:  
            csvreader = csv.DictReader(csv_file, delimiter = ';')
            for mask in csvreader:
                features[mask["maskPath"]] = OrderedDict(mask)

    # Boucle par masque pour l'extraction :
    for maskPath in listMasks:
        ProgBarSet += 1 / len(listMasks)

        numAno = maskPath.split('/')[-2]
        listImg = [os.path.join(os.path.dirname(maskPath), x) for x in os.listdir(os.path.dirname(maskPath)) if "img.nii.gz" in x]

        if len(listImg) > 0:
            imgPath = difflib.get_close_matches(maskPath, listImg)[0].replace("\\\\", "/").replace("\\", "/")
        else:
            with open('..\log_extract.txt', 'a', encoding="utf-8") as log_extract:
                log_extract.write(f"Image not found for mask {maskPath}.\n")
            progress_bar.progress(ProgBarSet)
            time.sleep(0.1)

            continue
        
        # Si l'option "réextraction" est activée (Etape 1) ou qu'aucun fichier radiomics_features.csv n'existe, on extrait tout :
        if (reextraction) or not os.path.exists(file_path):
            try:           
                features[maskPath] = Extractor.execute(imgPath, maskPath)
                features[maskPath]["patient"] = numAno
                features[maskPath]["mask"] = maskPath.replace(os.path.commonprefix([maskPath, imgPath]), '').replace('_mask.nii.gz', '')
                features[maskPath]["maskPath"] = maskPath
                features[maskPath]["imgPath"] = imgPath
                features[maskPath]["dngType"] = "FBN" if Extractor.settings.get("binCount") is not None else "FBS"
                features[maskPath]["dngBin"] = str(Extractor.settings["binCount"] if Extractor.settings.get("binCount") is not None else Extractor.settings["binWidth"])
                features[maskPath]["intensityNormalization"] = str(1 if Extractor.settings.get("normalize", False) is True else 0)
                features[maskPath]["outliersRemoval"] = str(Extractor.settings.get('removeOutliers') if Extractor.settings.get('removeOutliers') is not None else 0)

                features[maskPath].move_to_end("outliersRemoval", last = False)
                features[maskPath].move_to_end("intensityNormalization", last = False)
                features[maskPath].move_to_end("dngBin", last = False)
                features[maskPath].move_to_end("dngType", last = False)
                features[maskPath].move_to_end("imgPath", last = False)
                features[maskPath].move_to_end("maskPath", last = False)
                features[maskPath].move_to_end("mask", last = False)
                features[maskPath].move_to_end("patient", last = False)
                progress_bar.progress(ProgBarSet)
                time.sleep(0.1)
            except Exception as e:
                with open('..\log_extract.txt', 'a', encoding="utf-8") as log_extract:
                    log_extract.write(f"Pyradiomics extraction error for mask [{maskPath}] and image [{imgPath}] : {e}\n")
                progress_bar.progress(ProgBarSet)
                time.sleep(0.1)
                continue
        
        # Si un fichier radiomics_features.csv existe et que l'option de réextraction est désactivée (Etape 1), on extrait les paramètres et/ou masques manquants :
        else:
            # Si le masque existe déjà, on vérifie quels paramètres sont manquants :
            if (maskPath in features.keys()) and (str(features[maskPath].get("diagnostics_Configuration_Settings", 0)) == str(Extractor.settings)):
                listOfAlreadyExtractedFt = [f for f in features[maskPath].keys() if (f not in cles) and ("diagnostics" not in f) and (len(features[maskPath][f]) > 0)]
                for imageType, customKwargs in six.iteritems(Extractor.enabledImagetypes):
                    extractor2 = copy.deepcopy(Extractor)

                    sublistOfAlreadyExtractedFt = []
                    sublistOfFtNotWanted = []
                    sublistOfAlreadyExtractedFt = ['_'.join(x.split('_')[-2:]) for x in listOfAlreadyExtractedFt if any((x.lower().startswith(imageType.lower())) and (x.lower().endswith(ef.lower())) for ef in enabledFtList)]
                    sublistOfFtNotWanted = ['_'.join(x.split('_')[-2:]) for x in listOfAlreadyExtractedFt if (x.lower().startswith(imageType.lower())) and ('_'.join(x.split('_')[-2:]) not in enabledFtList)]

                    sublistOfFtToExtract = {}
                    sublistOfFtToDelete = {}
                    for ftClass in Extractor.enabledFeatures.keys():
                        sublistOfFtToExtract[ftClass] = [x.split('_')[-1] for x in enabledFtList if (x not in sublistOfAlreadyExtractedFt) and (x.lower().startswith(ftClass.lower()))]
                        sublistOfFtToDelete[ftClass] = [x.split('_')[-1] for x in sublistOfFtNotWanted if (x.lower().startswith(ftClass.lower()))]

                    sublistOfFtToExtract = {k: v for k, v in sublistOfFtToExtract.items() if len(v) > 0}
                    sublistOfFtToDelete = {k: v for k, v in sublistOfFtToDelete.items() if len(v) > 0}

                    # Modification des paramètres imageType et enableFeaturesByName
                    extractor2.disableAllImageTypes()
                    extractor2.disableAllFeatures()
                    extractor2.enableImageTypeByName(imageType, customArgs = customKwargs)
                    extractor2.enableFeaturesByName(**sublistOfFtToExtract)

                    # Extraction des paramètres manquants
                    try:           
                        features[maskPath] = {**features[maskPath], **(extractor2.execute(imgPath, maskPath))}
                        progress_bar.progress(ProgBarSet)
                        time.sleep(0.1)
                    except Exception as e:
                        with open('..\log_extract.txt', 'a', encoding="utf-8") as log_extract:
                            log_extract.write(f"Pyradiomics extraction error for mask [{maskPath}] and image [{imgPath}] : {e}\n")
                        progress_bar.progress(ProgBarSet)
                        time.sleep(0.1)
                        continue

            else:
                try:           
                    features[maskPath] = Extractor.execute(imgPath, maskPath)
                    features[maskPath]["patient"] = numAno
                    features[maskPath]["mask"] = maskPath.replace(os.path.commonprefix([maskPath, imgPath]), '').replace('_mask.nii.gz', '')
                    features[maskPath]["maskPath"] = maskPath
                    features[maskPath]["imgPath"] = imgPath
                    features[maskPath]["dngType"] = "FBN" if Extractor.settings.get("binCount") is not None else "FBS"
                    features[maskPath]["dngBin"] = str(Extractor.settings["binCount"] if Extractor.settings.get("binCount") is not None else Extractor.settings["binWidth"])
                    features[maskPath]["intensityNormalization"] = str(1 if Extractor.settings.get("normalize", False) is True else 0)
                    features[maskPath]["outliersRemoval"] = str(Extractor.settings.get('removeOutliers') if Extractor.settings.get('removeOutliers') is not None else 0)

                    features[maskPath].move_to_end("outliersRemoval", last = False)
                    features[maskPath].move_to_end("intensityNormalization", last = False)
                    features[maskPath].move_to_end("dngBin", last = False)
                    features[maskPath].move_to_end("dngType", last = False)
                    features[maskPath].move_to_end("imgPath", last = False)
                    features[maskPath].move_to_end("maskPath", last = False)
                    features[maskPath].move_to_end("mask", last = False)
                    features[maskPath].move_to_end("patient", last = False)
                    progress_bar.progress(ProgBarSet)
                    time.sleep(0.1)
                except Exception as e:
                    with open('..\log_extract.txt', 'a', encoding="utf-8") as log_extract:
                        log_extract.write(f"Pyradiomics extraction error for mask [{maskPath}] and image [{imgPath}] : {e}\n")
                    progress_bar.progress(ProgBarSet)
                    time.sleep(0.1)
                    continue
    
    # Enregistrement des résultats :
    with open(file_path, 'w', newline = '', encoding="utf-8") as csv_file:  
        csvwriter = csv.DictWriter(csv_file, features[list(features)[0]].keys(), delimiter = ";")
        csvwriter.writeheader()
        csvwriter.writerows([maskFt for maskFt in features.values()])

    # Opération terminée : affichage de l'information et déblocage du bouton
    progress_bar.progress(100)
    endTime = time.time()
    executionDuration = int(endTime - startTime)
    st.write(f'Opération terminée en {executionDuration} secondes.')
    return




def changeImageTypes(var, CustomImageTypes, sigma, level, start, valeur, selected_options):
    if "original" in var:
        if "on" in var:
            CustomImageTypes['Original'] = {}
    elif "log" in var:
        if "on" in var:
            CustomImageTypes['LoG'] = {'sigma' : [float(x) for x in sigma.split(',')] if len(sigma) > 0 else 1.0}

    elif "wavelet" in var:
        if "on" in var:
            CustomImageTypes['Wavelet'] = {'level': int(level) if len(level) > 0 else 1,
                                                'start_level': int(start) if len(start) > 0 else 0,
                                                'wavelet': selected_options + str(valeur if selected_options not in ["haar", "dmey"] else "")}
    # print(self.CustomImageTypes)
    return

def validateImageTypes(CustomImageTypes, ExtractorTemp, log_val, sigma, wavelet_val, level, start, valeur, selected_options):
    if log_val == "log_on":
        CustomImageTypes['LoG'] = {'sigma' : [float(x) for x in sigma.split(',')] if len(sigma) > 0 else 1.0}
    if wavelet_val == "wavelet_on":
        CustomImageTypes['Wavelet'] = {'level': int(level) if len(level) > 0 else 1,
                                            'start_level': int(start) if len(start) > 0 else 0,
                                            'wavelet': selected_options + str(valeur if selected_options not in ["haar", "dmey"] else "")}

    ExtractorTemp.enabledImagetypes = CustomImageTypes
    # print(self.ExtractorTemp.enabledImagetypes)
    return

def switchInterpolation(ExtractorTemp):
    interpol_Var = 1 if (ExtractorTemp.settings.get('resampledPixelSpacing', None) is not None) or (ExtractorTemp.settings.get('interpolator', None) is not None) else 0
    interpol_ValueVar = str(ExtractorTemp.settings.get('resampledPixelSpacing')) if ExtractorTemp.settings.get('resampledPixelSpacing', None) is not None else "[0, 0, 0]"
    interpol_TypeVar = ExtractorTemp.settings.get('interpolator', None) if ExtractorTemp.settings.get('interpolator', None) is not None else "sitkBSpline"
    if interpol_Var == 1:
        ExtractorTemp.settings['resampledPixelSpacing'] = [float(s) for s in interpol_ValueVar[1:-1].split(',')]
        ExtractorTemp.settings['interpolator'] = interpol_TypeVar
    else:
        ExtractorTemp.settings['resampledPixelSpacing'] = None
        ExtractorTemp.settings['interpolator'] = None
    return

def switchNormalization(ExtractorTemp):
    norm_Var = 1 if ExtractorTemp.settings.get('normalize', False) == True else 0
    if norm_Var == 1:
        ExtractorTemp.settings['normalize'] = True
    else:
        ExtractorTemp.settings['normalize'] = False
        normOutliers_Var = 0
        switchOutliers(normOutliers_Var, ExtractorTemp)
    return

def switchOutliers(normOutliers_Var, ExtractorTemp):
    normOutliers_Var = 0 if ExtractorTemp.settings.get('removeOutliers') is None else 1
    normOutliers_Value = ExtractorTemp.settings.get('removeOutliers') if ExtractorTemp.settings.get('removeOutliers') is not None else '3'
    if normOutliers_Var == 1:
        ExtractorTemp.settings['removeOutliers'] = int(normOutliers_Value)
    else:
        ExtractorTemp.settings['removeOutliers'] = None
    return


def changeExtractorFtClasses(ExtractorTemp, var):
    if str(var)[-8:] == "disabled":
        ExtractorTemp.enableFeatureClassByName(str(var)[:-9], enabled = False)
    else:
        ExtractorTemp.enableFeatureClassByName(str(var), enabled = True)
    return

def validateInterpolation(ExtractorTemp):
    interpol_Var = 1 if (ExtractorTemp.settings.get('resampledPixelSpacing', None) is not None) or (ExtractorTemp.settings.get('interpolator', None) is not None) else 0
    interpol_TypeVar = ExtractorTemp.settings.get('interpolator', None) if ExtractorTemp.settings.get('interpolator', None) is not None else "sitkBSpline"
    interpol_ValueVar = str(ExtractorTemp.settings.get('resampledPixelSpacing')) if ExtractorTemp.settings.get('resampledPixelSpacing', None) is not None else "[0, 0, 0]"
    if interpol_Var == 1:
        ExtractorTemp.settings['resampledPixelSpacing'] = [float(s) for s in interpol_ValueVar[1:-1].split(',')]
        ExtractorTemp.settings['interpolator'] = interpol_TypeVar
    return

def validate_preprocessing(CustomImageTypes, ExtractorTemp, ExtractorParams, log_var, sigma, wavelet_var, level, start, value_outlier, discretisation, bin_size, bin_number, valeur, selected_options):
    validateImageTypes(CustomImageTypes, ExtractorTemp, log_var, sigma, wavelet_var, level, start, valeur, selected_options)
    switchLabelDNG(discretisation, ExtractorTemp, ExtractorParams, bin_size, bin_number)
    switchOutliers(value_outlier, ExtractorTemp)
    validateInterpolation(ExtractorTemp)
    Extractor = copy.deepcopy(ExtractorTemp)


def switchLabelDNG(discretisation, ExtractorTemp, ExtractorParams, bin_size, bin_number):
    dng_parametrageVar = bin_size if "binWidth" in ExtractorParams["dngType"] else bin_number
    if discretisation == "Fixed Bin Size":
        ExtractorParams["dngType"] = "binWidth"
        ExtractorTemp.settings["binWidth"] = int(dng_parametrageVar) if int(dng_parametrageVar) > 0 else 1
        ExtractorTemp.settings["binCount"] = None
    else:
        ExtractorParams["dngType"] = "binCount"
        ExtractorTemp.settings["binWidth"] = None
        ExtractorTemp.settings["binCount"] = int(dng_parametrageVar) if int(dng_parametrageVar) > 0 else 16


########################################



# Exécution de l'application
if __name__ == "__main__":
    main()

