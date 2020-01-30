#!/usr/bin/env python
# coding: utf-8
"""
    Slice Extraction
    Extract raw PT and CT images from DICOM data
Usage:

    python SliceExtraction.py (path_src)
"""

# Import libraries
import pydicom
import os
import sys
import glob
from os.path import dirname
import numpy as np
import cv2


def SliceExtraction(path_src):
    """
    Extract raw PT and CT images from DICOM data and copy images to directory
    PT/CT images will be place on 'base_dir/patient1/CT', 'base_dir/patient1/PT'

    :param path_src: Path for DICOM Directory
    :return:
    """
    # Window leveling parameters for CT image
    wl_CT = 400
    ww_CT = 1600

    # Find DICOMDIR path
    dicomdir_list = glob.glob(f'{path_src}/dicom/**/DICOMDIR', recursive=True)
    print(f"Number of DICOMDIR: {len(dicomdir_list)}")

    # Extract All Slices from DICOM
    for dicomdir in dicomdir_list:

        try:
            ds = pydicom.read_file(dicomdir)
        except IndexError:
            print(f'Error: dicomdir is empty: ' + dicomdir)
        print(f'Read {dicomdir}')

        patient_dir = dirname(dicomdir)
        dst_dir = dirname(patient_dir)

        createFolder(dst_dir + '/PT')
        createFolder(dst_dir + '/CT')

        for record in ds.DirectoryRecordSequence:
            if record.DirectoryRecordType == "IMAGE":
                # Extract the relative path to the DICOM file
                path = os.path.join(patient_dir, *record.ReferencedFileID)
                try:
                    dcm = pydicom.read_file(path)
                except FileNotFoundError:
                    print('File Not Found Error!')
                    break

                # Separate PT and CT images
                if "PT" in dcm.Modality:
                    img = dcm.pixel_array
                    img_PT_min = img.min()
                    img_PT_max = img.max()
                    img_norm = (img - img_PT_min) / (img_PT_max - img_PT_min) * 255
                    cv2.imwrite(dst_dir + '/PT/'
                                + record.ReferencedFileID[-1] + '.png', 255 - img_norm)
                else:
                    # Adjust Window level : -400 and Window width : 1600
                    # Min : (Level - Width/2) / Max : (Level + Width/2)
                    img = dcm.pixel_array
                    img_window = np.clip(img + wl_CT, 0, ww_CT)
                    img_norm = img_window / ww_CT * 255
                    cv2.imwrite(dst_dir + '/CT/'
                                + record.ReferencedFileID[-1] + '.png', img_norm)

    print(f'Slice Extraction Completed!')


# Function for making directory
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print(f'Error: Creating directory. ' + directory)


if __name__ == '__main__':
    SliceExtraction(sys.argv[1])
