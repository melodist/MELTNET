#!/usr/bin/env python
# coding: utf-8

# # Patch Extraction
# dicom -> Patient -> Study -> DICOMDIR

import pydicom
import os
import glob
from os.path import dirname, join
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import cv2
import shutil


def findDICOMDIR(path):
    # Find DICOMDIR path
    dicomdir_list = glob.glob(path, recursive=True)
    print(f"Number of DICOMDIR: {len(dicomdir_list)}")
    return dicomdir_list


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print(f'Error: Creating directory. ' + directory)


# ## Extract Slices from DICOM
# dicomdir_list[0] -> 146 ~ 184 / total 234 slices on CT
# 0.62 ~ 0.78

# Extract All Slices from DICOM
wl = -400
ww = 1600

img_min = wl - ww / 2
img_max = wl + ww / 2


def extractAllSlices(dicomdir_list, wl, ww):
    for dicomdir in dicomdir_list:
        # dicomdir = dicomdir_list[77]
        try:
            ds = pydicom.read_file(dicomdir)
        except IndexError:
            print(f'Error: dicomdir is empty: ' + dicomdir)
            continue
        print(f'Read {dicomdir}')

        base_dir = dirname(dicomdir)
        base2_dir = dirname(base_dir)

        createFolder(base2_dir + '/PT')
        createFolder(base2_dir + '/CT')

        # pixel_data_CT = list()
        # pixel_data_PET = list()

        # Adjust Window level : -400 and Window width : 1600
        # Min : (Level - Width/2) / Max : (Level + Width/2)

        for record in ds.DirectoryRecordSequence:
            if record.DirectoryRecordType == "IMAGE":
                # Extract the relative path to the DICOM file
                path = os.path.join(base_dir, *record.ReferencedFileID)
                dcm = pydicom.read_file(path)

                # Limit slices for lung
                dcm.InstanceNumber
                # Now get your image data
                if "PT" in dcm.Modality:
                    # pixel_data_PET.append(dcm.pixel_array)
                    img = dcm.pixel_array
                    cv2.imwrite(base2_dir + '/PT/'
                                + record.ReferencedFileID[-1] + '.png', img)
                else:
                    img = dcm.pixel_array
                    img_window = np.clip(img, 0, img_max)
                    img_norm = img_window / img_max * 255
                    # pixel_data_CT.append(img_norm.astype('uint8'))
                    cv2.imwrite(base2_dir + '/CT/'
                                + record.ReferencedFileID[-1] + '.png', img_norm)


# Copy Slices for lung to new folder
def copyDICOM(dicomdir_list, dst, max_point, min_point):
    for dicomdir in dicomdir_list:
        # dicomdir = dicomdir_list[0]

        try:
            ds = pydicom.read_file(dicomdir)
        except IndexError:
            print(f'Error: dicomdir is empty: ' + dicomdir)
            # continue
        print(f'Read {dicomdir}')

        record = ds.DirectoryRecordSequence[0]
        patientname = record.PatientName.components[0]
        abspath = dst + '\\' + record.PatientID + ' ' + patientname
        base_dir = os.path.relpath(abspath, ".")
        print(base_dir)
        base2_dir = dirname(dirname(dicomdir))

        createFolder(base_dir + '/PT_lung')
        createFolder(base_dir + '/CT_lung')

        num_slices = len(glob.glob(base2_dir + '/CT/*'))

        # check slices are in the range
        # Find Maximum and minimum and copy all the files through loop
        max_slices = np.ceil(max_point * num_slices).astype('uint16') - 1
        min_slices = np.floor(min_point * num_slices).astype('uint16') - 1
        print(f"Copy slice #{min_slices} to slice #{max_slices}")
        for i in range(min_slices, max_slices):
            try:
                shutil.copy(base2_dir + '/CT/I' + str(i) + '0.png',
                            base_dir + '/CT_lung/I' + str(i) + '0.png', )
            except FileNotFoundError:
                print(f"File Not Found: {base2_dir + '/CT/I' + str(i) + '0.png'}")

            try:
                shutil.copy(base2_dir + '/PT/I' + str(i) + '0.png',
                            base_dir + '/PT_lung/I' + str(i) + '0.png', )
            except FileNotFoundError:
                print(f"File Not Found: {base2_dir + '/CT/I' + str(i) + '0.png'}")


def main():
    max_point = 0.75
    min_point = 0.65
    dicomdir_list = findDICOMDIR('D:\\나 문서\\MyStudy\\Artificial Neural Network\\구로병원 Data\\dicom\\**\\DICOMDIR')
    dst_path = 'C:\\Users\\Junhwa\\PET-CT Images'
    copyDICOM(dicomdir_list, dst_path, max_point, min_point)


main()
