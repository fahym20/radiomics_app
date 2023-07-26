import os
import pydicom as pdc
import numpy as np
import vtk
import trimesh
import SimpleITK as sitk

from tkinter.messagebox import *

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
    if arborescence == 1:
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


# Fonctions pour transformation OFF vers NIFTI --------------------------------------

def readImage(inputImagePath):
    # Lecture de l'image et de sa géométrie
    niirdr = vtk.vtkNIFTIImageReader()
    niirdr.SetFileName(inputImagePath)
    niirdr.Update()

    return niirdr

def readOff(inputOffPath):
    # Lecture du fichier .OFF
    vertices = []
    faces=  []
    edges = []

    with open(inputOffPath, 'r') as off:
        for cnt, line in enumerate(off):
            if "OFF" in line:
                continue
            if cnt == 1:
                nvertices, nfaces, nedges = line.split(sep=" ")
            if cnt < int(nvertices) + 2 and cnt != 1:
                vertices.append([float(x) for x in line.split(sep=" ")])
            if cnt > int(nvertices) + 1 and cnt < int(nvertices) + int(nfaces) + 2 :
                numvert, vert1, vert2, vert3 = line.split(sep=" ")
                faces.append([int(vert1), int(vert2), int(vert3)])

    verticesArray = np.asarray(vertices)
    facesArray = np.asarray(faces)

    return verticesArray, facesArray


def createPolyData(verticesArray, facesArray):
    # Création d'une structure PolyData avec vtk

    # Points
    VTKpoints = vtk.vtkPoints()
    VTKvertices = vtk.vtkCellArray()
    for i in range(0, len(verticesArray)):
        p = verticesArray[i].tolist()
        point_id = VTKpoints.InsertNextPoint(p)
        VTKvertices.InsertNextCell(1)
        VTKvertices.InsertCellPoint(point_id)

    # Liste id
    def mkVtkIdList(it):
        vil = vtk.vtkIdList()
        for i in it:
            vil.InsertNextId(int(i))
        return vil

    # Faces
    VTKpolys = vtk.vtkCellArray()
    for i in range(0, len(facesArray)):
        VTKpolys.InsertNextCell(mkVtkIdList(facesArray[i, :]))

    # PolyData
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(VTKpoints)
    polydata.SetVerts(VTKvertices)
    polydata.SetPolys(VTKpolys)
    polydata.Modified()

    return polydata

def voxelizePolyData(img, polydata):
    # Création d'une image blanche
    whiteImage = vtk.vtkImageData()
    whiteImage.SetSpacing(img.GetOutput().GetSpacing())
    whiteImage.SetDimensions(img.GetOutput().GetDimensions())
    whiteImage.SetExtent(img.GetOutput().GetExtent())
    whiteImage.SetOrigin(img.GetOutput().GetOrigin())
    whiteImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR,1)
    whiteImage.Modified()

    # Remplissage de l'image blanche par des 1
    inval = 1
    outval = 0
    count = whiteImage.GetNumberOfPoints()

    data = np.ones(count,dtype='B')
    data[:] = inval
    a = vtk.vtkUnsignedCharArray()
    a.SetArray(data, data.size, True)
    whiteImage.GetPointData().SetScalars(a)

    # Découpe du masque dans l'image blanche
    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(polydata)

    pol2stenc.SetOutputOrigin(whiteImage.GetOrigin())
    pol2stenc.SetOutputSpacing(whiteImage.GetSpacing())
    pol2stenc.SetOutputWholeExtent(whiteImage.GetExtent())
    pol2stenc.Update()

    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(whiteImage)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())

    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(outval)
    imgstenc.Update()

    # Masque final
    maskImage = imgstenc.GetOutput()

    return maskImage

def writeOutputNii(maskImage, img, outputNiiPath):
    # Enregistrement du masque en Nifti en s'assurant de copier le header de l'image
    writer = vtk.vtkNIFTIImageWriter()
    writer.SetFileName(outputNiiPath)
    writer.SetInputData(maskImage)
    writer.SetNIFTIHeader(img.GetNIFTIHeader())
    writer.Write()

def off2nii(inputImagePath, inputOffPath, outputNiiPath):
    # Fonction de transformation des fichiers .OFF en .NII
    img = readImage(inputImagePath)
    vertices, faces = readOff(inputOffPath)
    polydata = createPolyData(vertices, faces)
    maskImage = voxelizePolyData(img, polydata)
    writeOutputNii(maskImage, img, outputNiiPath)

# Fonction de transformation manuelle des OFF vers NIFTI --------------------------------
def off2nii_manual(inputImagePath, inputOffPath, outputNiiPath):
    mesh = trimesh.load(inputOffPath)
    img = sitk.ReadImage(inputImagePath)
    img_array = sitk.GetArrayFromImage(img)
    new_array = img_array.copy().astype(int)
    new_array[:] = 0

    v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, np.min(img.GetSpacing()))

    for x,y,z in v:
        xi = int(np.ceil(x / img.GetSpacing()[0]))
        yi = int(np.ceil(y / img.GetSpacing()[2]))
        zi = int(np.ceil(z / img.GetSpacing()[1]))

        new_array[yi, zi, xi] = 1

    new_array = np.flip(new_array, axis = 1).astype(int)

    new_img = sitk.GetImageFromArray(new_array)
    new_img = sitk.BinaryFillhole(new_img)
    new_img.CopyInformation(img)
    sitk.WriteImage(new_img, outputNiiPath)

def off2nii_manual_sag(inputImagePath, inputOffPath, outputNiiPath):
    mesh = trimesh.load(inputOffPath)
    img = sitk.ReadImage(inputImagePath)
    img_array = sitk.GetArrayFromImage(img)
    new_array = img_array.copy()
    new_array[:] = 0

    v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, np.min(img.GetSpacing()))
    
    for x,y,z in v:
        xi = int(x / img.GetSpacing()[2])
        yi = int(y / img.GetSpacing()[0])
        zi = int(z / img.GetSpacing()[1])

        new_array[xi, yi, zi] = 1

    new_array = np.rot90(new_array, axes=(1,2)).astype(int)

    new_img = sitk.GetImageFromArray(new_array)
    new_img = sitk.BinaryFillhole(new_img)
    new_img.CopyInformation(img)
    sitk.WriteImage(new_img, outputNiiPath)

def off2nii_manual_coro(inputImagePath, inputOffPath, outputNiiPath):
    mesh = trimesh.load(inputOffPath)
    img = sitk.ReadImage(inputImagePath)
    img_array = sitk.GetArrayFromImage(img)
    new_array = img_array.copy()
    new_array[:] = 0

    v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, np.min(img.GetSpacing()))
    
    for x,y,z in v:
        xi = int(x / img.GetSpacing()[1])
        yi = int(y / img.GetSpacing()[2])
        zi = int(z / img.GetSpacing()[0])

        new_array[yi, xi, zi] = 1

    new_array = np.rot90(new_array, axes=(1,2)).astype(int)

    new_img = sitk.GetImageFromArray(new_array)
    new_img = sitk.BinaryFillhole(new_img)
    new_img.CopyInformation(img)
    sitk.WriteImage(new_img, outputNiiPath)