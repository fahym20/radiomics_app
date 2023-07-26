# -*- coding: utf-8 -*-

import os
import re
import csv
import pydicom as pdc
from random import randint
from collections import defaultdict
import time
import streamlit as st

dictionary = {}

def export_list_anonymisation(prefix, input, output):
    """ Lecture de l'ensemble des fichiers DICOM du dossier DicomInputFolderPath et export d'un fichier CSV dans
    le dossier du logiciel contenant une ligne par série avec : 
        - UID de la série (SeriesInstanceUID) 
        - ID du patient (PatientID) 
        - Nom du patient (PatientName) 
        - Date de naissance du patient (PatientBirthDate) 
        - Date de l'examen au format yyyyjjmm (StudyDate) 
        - Description de la série (SeriesDescription) 
        - Nombre de coupes par série (NumberOfSlices) 
        - ID de l'examen (AccessionNumber) 
        - Proposition de numéro d'anonymisation (AnonymizationID) basée sur le préfixe fourni par l'utilisateur

    Ce préfixe peut être modifié manuellement dans le fichier csv pour être pris en compte lors de l'anonymisation.
    """
    startTime = time.time()

    ### 1.Extraction des métadonnées DICOM de chaque série à partir de l'ensemble des fichiers
    # Se base sur "SeriesInstanceUID" pour trouver tous les fichiers communs à chaque série

    # Vérification : le dossier d'entrée est-il bien spécifié ?
    if len(input) == 0:
        st.write("Dossier DICOM non trouvé, vous devez sélectionner le dossier contenant les fichiers DICOM à anonymiser (Étape 2)")
        return
    elif not os.path.exists(input):
        st.write("Dossier DICOM non trouvé, vous devez sélectionner un dossier contenant les fichiers DICOM valide (Étape 2)")
        return

    # Initialisation de la barre de progression
    progress_bar = st.progress(0)

    # Création ou remise à zéro du fichier de log
    open('..\log_export.txt', 'w', encoding="utf-8").close()

    # Liste des fichiers DICOM dans le dossier d'entrée
    listDcm = []
    listSeries = {}
    
    for (dirPath, dirNames, fileNames) in os.walk(input):
        listDcm += [os.path.join(dirPath, filename) for filename in fileNames if (".dcm" in filename) or (is_file_a_dicom(os.path.join(dirPath, filename)) == True)]

    list_patients = []
    list_ano = {}
    for i, filepath in enumerate(listDcm):
        progress_bar.progress(i/len(listDcm))
        time.sleep(0.1)
        try:
            dcm = pdc.dcmread(filepath, stop_before_pixels = True)
            seriesDir = "_".join(re.sub(r"[^a-zA-Z0-9]", " ", str(dcm.SeriesDescription)).upper().split())

            if dcm.SeriesInstanceUID not in listSeries.keys() and seriesDir not in listSeries.keys():
                try:
                    if str(dcm.PatientID) not in list_patients:
                        list_patients.append(str(dcm.PatientID))
                        list_ano[str(dcm.PatientID)] = f"{prefix}{str(len(list_patients))}"

                    listSeries[str(dcm.SeriesInstanceUID), str(seriesDir)] = \
                        [list_ano[str(dcm.PatientID)],
                        str(dcm.PatientID),
                        " ".join(re.sub(r"[^a-zA-Z0-9]", " ", str(dcm.PatientName)).upper().split()),
                        str(dcm.PatientBirthDate),
                        str(dcm.PatientSex),
                        str(dcm.StudyDate),
                        str(dcm.InstitutionName),
                        str(dcm.Modality),
                        str(dcm.Manufacturer),
                        str(dcm.ManufacturerModelName),
                        str(dcm.SeriesInstanceUID),
                        str(dcm.SeriesDescription),
                        str(seriesDir)]
                except Exception as e:
                    with open('..\log_export.txt', 'a', encoding="utf-8") as log_export:
                        progress_bar.progress(i/len(listDcm))
                        time.sleep(0.1)
                        log_export.write(f"Failed to export: {filepath}: {e}\n")
                        st.write('Failed to export :', filepath, ':', e)
                    continue
            else:
                continue
        except Exception as f:
            with open('..\log_export.txt', 'a', encoding="utf-8") as log_export:
                progress_bar.progress(i/len(listDcm))
                time.sleep(0.1)
                log_export.write(f"Failed to export: {filepath}: {f}\n")
                st.write('Failed to export', filepath, ":", f)
            continue

    # Enregistrement du fichier csv contenant la liste des séries et certaines caractéristiques de patients
    
    file_path = os.path.join(output, 'liste_dicom.csv')

    with open(file_path, mode='w', newline='', encoding="utf-8") as csv_file:
        col_names = ["AnonymizationID",
                        "PatientID",
                        "PatientName",
                        "PatientBirthDate",
                        "Sex",
                        "StudyDate",
                        "InstitutionName",
                        "Modality",
                        "Manufacturer",
                        "ManufacturerModelName",
                        "SeriesInstanceUID", 
                        "SeriesDescription",
                        "SeriesOutputFolder"]
        writer = csv.writer(csv_file, delimiter=';')

        writer.writerow(col_names)
        for k, data in listSeries.items():
            writer.writerow(data)

    # Opération terminée : affichage de l'information et déblocage du bouton
    progress_bar.progress(100)
    endTime = time.time()
    executionDuration = int(endTime - startTime)
    st.write("Opération terminée en", executionDuration, "secondes.")



def deidentification(input, output, arborescence, reanonymisation, conserver_annee):
    """ Crée une copie anonymisée des fichiers DICOM présents dans le dossier DicomInputFolderPath
        dans dans le répertoire choisi à l'étape 2 par arborescence Patient > Date d'étude > Série
    """
    startTime = time.time()
    # Fichier de correspondance d'anonymisation
    cle = defaultdict(list)

    if os.path.isfile(os.path.join(output, "liste_dicom.csv")) :
        with open(os.path.join(output, "liste_dicom.csv"), 'r', encoding="utf-8") as data:
            for line in csv.DictReader(data, delimiter=';'):
                for k, v in line.items():
                    cle[k].append(v)
    else:
        st.write('Dossier \'liste_dicom.csv\' non trouvé (Étape 4) ')
        return


    # Vérification : le dossier de sortie est-il bien spécifié, différent du dossier d'entrée, et valide ?
    if len(output) == 0:
        st.write("Dossier de sortie non trouvé, sélectionnez le dossier qui contiendra les fichiers DICOM anonymisés")
        return
    elif input == output :
        st.write("Dossier DICOM de sortie non valide, sélectionnez un dossier de sortie différent du dossier d'entrée")
        return
    elif not os.path.exists(output):
        try:
            os.mkdir(output)
        except:
            st.write("Dossier DICOM de sortie non valide, sélectionnez un dossier de sortie valide")
            return

    # Initialisation de la barre de progression
    progress_bar = st.progress(0)

    # Liste des fichiers DICOM dans le dossier d'entrée
    listDcm = []
    for (dirPath, dirNames, fileNames) in os.walk(input):
        listDcm += [os.path.join(dirPath, filename) for filename in fileNames if (".dcm" in filename) or (is_file_a_dicom(os.path.join(dirPath, filename)) == True)]

    # Création ou remise à zéro du fichier de log
    open('..\log_anonymization.txt', 'w', encoding="utf-8").close()
    progbarset = 0
    progbarstep = 1 / len(listDcm)
    # Boucle d'anonymisation à partir de l'ensemble des fichiers présents dans le dossier d'entrée
    for i, file in enumerate(listDcm):
        try:
            progbarset += progbarstep
            dcm = pdc.dcmread(file)
            # Restriction aux seules séries sélectionnées dans le fichier liste_dicom.csv
            if dcm.SeriesInstanceUID in cle["SeriesInstanceUID"]:  
                numAno = str(cle["AnonymizationID"][cle["SeriesInstanceUID"].index(dcm.SeriesInstanceUID)])
                seriesDir = str(cle["SeriesOutputFolder"][cle["SeriesInstanceUID"].index(dcm.SeriesInstanceUID)])
                outputDir = createOutputDir(arborescence, output, numAno, seriesDir)

                new_SUID = [str(int(float(char)*3/5)+1) if char.isalnum() else char for char in dcm.SeriesInstanceUID]
                new_SUID = ''.join(new_SUID)

                new_SUID_filename = new_SUID.replace('.', '')[-1:2:-2]

                filename = f"{numAno}_{seriesDir}_{str(new_SUID_filename).replace('.', '')}_img_{str(i)}.dcm"
                annee = str(dcm.StudyDate)[:4]

                # Assure de ne pas réanonymiser si la case REANONYMISATION n'est pas cochée
                if os.path.exists(os.path.join(outputDir, filename)) and reanonymisation == False :
                    progress_bar.progress(i/len(listDcm))
                    time.sleep(0.1)
                else:
                    try:
                        dcm = dcm_anonymizator(dcm, file)
                        dcm.PatientName = numAno
                        dcm.SeriesInstanceUID = new_SUID
                        if conserver_annee :
                            dcm.StudyDate = str(annee + "0101")
                        dcm.save_as(os.path.join(outputDir, filename), write_like_original = False)
                        progress_bar.progress(i/len(listDcm))
                        time.sleep(0.1)
                    except Exception as e:
                        progress_bar.progress(i/len(listDcm))
                        time.sleep(0.1)
                        with open("..\log_anonymization.txt", "a", encoding="utf-8") as flogg:
                            flogg.write(f"{os.path.join(outputDir, filename)} error: {e} \n")
                            st.write(os.path.join(outputDir, filename), 'error', e)
                        continue
            else:
                progress_bar.progress(i/len(listDcm))
                time.sleep(0.1)   
        except Exception as e:
            progress_bar.progress(i/len(listDcm))
            time.sleep(0.1)
            with open("..\log_anonymization.txt", "a", encoding="utf-8") as flogg:
                flogg.write(f"{os.path.join(file)} error: {e} \n")
                st.write(os.path.join(file), 'error', e)

            continue   
    
    # Opération terminée : affichage de l'information et déblocage du bouton
    progress_bar.progress(100)
    endTime = time.time()
    executionDuration = int(endTime - startTime)
    st.write("Opération terminée en", executionDuration, "secondes.")





# Vérification de l'extension DICOM ou non de chaque fichier
def is_file_a_dicom(file):
    """ 
    Vérifie si file est un fichier DICOM (renvoie TRUE) ou pas (renvoie FALSE)

    """
    try:
        pdc.read_file(file, stop_before_pixels = True)
    except:
        return False
    return True 

# Création de l'arborescence de sortie
def createOutputDir(arborescence, outputParentPath, numAno, seriesDir):
    '''
    Crée l'arborescence de sortie en fonction de l'option ARBORESCENCE
    '''
    if arborescence == True :
        outputDir = os.path.join(outputParentPath, numAno, seriesDir)
    else:
        outputDir = os.path.join(outputParentPath, numAno)

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
        
    return outputDir

# Formatage du temps
def to_hms(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return '{}h{:0>2}m{:0>2}s'.format(h, m, s)

# Récupération des tags à anonymiser
def get_tags_from_file(active_lines, TAGS):
    try:
        tags_start_line, tags_end_line = [i for i, s in enumerate(active_lines) if s.startswith(f"{TAGS}_START") or s.startswith(f"{TAGS}_END")]
        TAGS_LINES = active_lines[tags_start_line + 1:tags_end_line]
        TAGS = [re.search(r"\([0-9a-zA-Z ,]+\)", x).group() for x in TAGS_LINES]
        TAGS = [eval(x) for x in TAGS]

        return TAGS
    except:
        raise AttributeError("{} n'existe pas".format(TAGS))


with open('new_champs_ano.txt') as f:
    active_lines = f.readlines()


D_TAGS = get_tags_from_file(active_lines, "D_TAGS")
Z_TAGS = get_tags_from_file(active_lines, "Z_TAGS")
U_TAGS = get_tags_from_file(active_lines, "U_TAGS")
X_TAGS = get_tags_from_file(active_lines, "X_TAGS")
Z_D_TAGS = get_tags_from_file(active_lines, "Z_D_TAGS")
X_Z_TAGS = get_tags_from_file(active_lines, "X_Z_TAGS")
X_D_TAGS = get_tags_from_file(active_lines, "X_D_TAGS")
X_Z_D_TAGS = get_tags_from_file(active_lines, "X_Z_D_TAGS")
X_Z_U_STAR_TAGS = get_tags_from_file(active_lines, "X_Z_U_STAR_TAGS")


# Modes d'anonymisation
def replaceElementUID(element):
    if element.value not in dictionary:
        new_chars = [str(randint(0, 9)) if char.isalnum() else char for char in element.value]
        dictionary[element.value] = ''.join(new_chars)
    element.value = dictionary.get(element.value)


def replaceElementDate(element):
    element.value = '00010101'


def replaceElementDateTime(element):
    element.value = '00010101010101.000000+0000'


def replaceElement(element):
    if element.VR == 'DA':
        replaceElementDate(element)
    elif element.VR == 'TM':
        element.value = '000000.00'
    elif element.VR in ('LO', 'SH', 'PN', 'CS'):
        element.value = 'Blinded'
    elif element.VR == 'UI':
        replaceElementUID(element)
    elif element.VR == 'UL':
        pass
    elif element.VR == 'IS':
        element.value = '0'
    elif element.VR == 'SS':
        element.value = 0
    elif element.VR == 'SQ':
        for subDataset in element.value:
            for subElement in subDataset.elements():
                replaceElement(subElement)
    elif element.VR == 'DT':
        replaceElementDateTime(element)
    else:
        pass
        #raise NotImplementedError('Non anonymisé. VR {} non encore implémenté.'.format(element.VR))


def replace(dataset, tag):
    """
    D - Remplacement par une valeur de longueur non nulle qui peut être une valeur fictive et qui est conforme à la
    VR
    """
    element = dataset.get(tag)
    if element is not None:
        replaceElement(element)


def emptyElement(element):
    if (element.VR in ('SH', 'PN', 'UI', 'LO', 'CS')):
        element.value = ''
    elif element.VR == 'DA':
        replaceElementDate(element)
    elif element.VR == 'TM':
        element.value = '000000.00'
    elif element.VR == 'UL':
        element.value = 0
    elif element.VR == 'SQ':
        for subDataset in element.value:
            for subElement in subDataset.elements():
                emptyElement(subElement)
    else:
        #raise NotImplementedError('Non anonymisé. VR {} non encore implémenté.'.format(element.VR))
        pass

def empty(dataset, tag):
    """
    Z - Remplacement par une valeur de longueur zéro, ou une valeur de longueur non nulle qui peut être une valeur
    fictive et conforme à la VR
    """
    element = dataset.get(tag)
    if element is not None:
        emptyElement(element)


def deleteElement(dataset, element):
    if element.VR == 'DA':
        replaceElementDate(element)
    elif element.VR == 'SQ':
        for subDataset in element.value:
            for subElement in subDataset.elements():
                deleteElement(subDataset, subElement)
    else:
        del dataset[element.tag]


def delete(dataset, tag):
    """X - Suppression"""

    def rangeCallback(dataset, dataElement):
        if dataElement.tag.group & tag[2] == tag[0] and dataElement.tag.element & tag[3] == tag[1]:
            deleteElement(dataset, dataElement)

    if (len(tag) > 2):  # Tag ranges
        dataset.walk(rangeCallback)
    else:  # Individual Tags
        element = dataset.get(tag)
        if element is not None:
            deleteElement(dataset, element)  # element.tag is not the same type as tag.


def keep(dataset, tag):
    """K - Garde"""
    pass


def clean(dataset, tag):
    """
    C - Remplacement par des valeurs de signification similaire connues pour ne pas contenir d'identification
    d'information et compatible avec le RV
    """
    if dataset.get(tag) is not None:
        raise NotImplementedError('Tag not anonymized. Not yet implemented.')


def replaceUID(dataset, tag):
    """
    U - Remplacement par un UID de longueur non nulle qui est cohérent au sein d'un ensemble d'instances
    """
    element = dataset.get(tag)
    if element is not None:
        replaceElementUID(element)


def emptyOrReplace(dataset, tag):
    """Z/D - Z sauf si D requis pour maintenir la conformité (Type 2 vs Type 1)"""
    replace(dataset, tag)


def deleteOrEmpty(dataset, tag):
    """X/Z - X sauf si Z requis pour maintenir la conformité (Type 3 vs Type 2)"""
    empty(dataset, tag)


def deleteOrReplace(dataset, tag):
    """X/D - X sauf si D requis pour maintenir la conformité (Type 3 vs Type 1)"""
    replace(dataset, tag)


def deleteOrEmptyOrReplace(dataset, tag):
    """
    X/Z/D - X sauf si Z ou D requis pour maintenir la conformité (Type 3 vs Type 2 vs Type 1)
    """
    replace(dataset, tag)


def deleteOrEmptyOrReplaceUID(dataset, tag):
    """
    X/Z/U* - X sauf si Z ou remplacement des UID requis pour maintenir la conformité (Type 3 vs Type 2 vs Type 1)
    """
    element = dataset.get(tag)
    if element is not None:
        if element.VR == 'UI':
            replaceElementUID(element)
        else:
            emptyElement(element)


# Actions d'anonymisation
DictionnaireActions = {
    "replace": replace,
    "empty": empty,
    "delete": delete,
    "replaceUID": replaceUID,
    "emptyOrReplace": emptyOrReplace,
    "deleteOrEmpty": deleteOrEmpty,
    "deleteOrReplace": deleteOrReplace,
    "deleteOrEmptyOrReplace": deleteOrEmptyOrReplace,
    "deleteOrEmptyOrReplaceUID": deleteOrEmptyOrReplaceUID,
    "keep": keep,
}

def generationActions(tagList, action):
    finalAction = action
    if not callable(action):
        finalAction = DictionnaireActions[action] if action in DictionnaireActions else keep
    return {tag: finalAction for tag in tagList}


def dcm_anonymizator(input_dicom, file):
    actions = generationActions(D_TAGS, replace)
    actions.update(generationActions(Z_TAGS, empty))
    actions.update(generationActions(X_TAGS, delete))
    actions.update(generationActions(U_TAGS, replaceUID))
    actions.update(generationActions(Z_D_TAGS, emptyOrReplace))
    actions.update(generationActions(X_Z_TAGS, deleteOrEmpty))
    actions.update(generationActions(X_D_TAGS, deleteOrReplace))
    actions.update(generationActions(X_Z_D_TAGS, deleteOrEmptyOrReplace))
    actions.update(generationActions(X_Z_U_STAR_TAGS, deleteOrEmptyOrReplaceUID))

    for tag, action in actions.items():
        try:
            action(input_dicom, tag)
        except:
            with open("log.txt", "a", encoding="utf-8") as flog:
                flog.write(';'.join([str(file), str(tag), str(action), "\n"]))
            continue

    # X - Private tags = (0xgggg, 0xeeee) avec 0xgggg impair
    input_dicom.remove_private_tags()

    return input_dicom
