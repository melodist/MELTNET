#!/usr/bin/env python
# coding: utf-8
"""
    Slice Selection
    Load data from csv and 'pick' the slices which meets the condition
Usage:

    python SliceSelection.py (path_csv, path_src, path_dst)
"""

# Import libraries
import pandas as pd
import os
import sys
import shutil


def SliceSelection(path_csv, path_src, path_dst):
    """
    Load data from csv and 'pick' the slices which meets the condition
    Copy slices which meets the condition to the destination folder
    Lung slices will be place on 'path_dst/patient1/CT', 'path_dst/patient1/PT'

    :param path_csv: Path for csv file
    :param path_src: Path for whole slices
    :param path_dst: Path for lung slices
    :return:
    """
    # Load csv file
    df = pd.read_csv(path_csv)

    # Copy slices which meets the condition
    for i in range(53):
        # Select Patient
        patient = df['Patient'][i]

        # Make path
        path_patient = f'{path_src}/dicom/{patient}/'

        up_patient = df['Up'][i]
        down_patient = df['Down'][i]

        try:
            up_patient == 0 or down_patient == 0
        except ValueError:
            print(f'Slice number is empty: {patient}')
            continue

        print(f"Copy slice #{up_patient} to slice #{down_patient}")

        # Make directory
        try:
            os.makedirs(f'{path_dst}/{patient}/CT/')
        except FileExistsError:
            print(f"Directory exists!: {path_dst}/{patient}/CT/")

        try:
            os.makedirs(f'{path_dst}/{patient}/PT/')
        except FileExistsError:
            print(f"Directory exists!: {path_dst}/{patient}/PT/")

        # Check if directory is empty
        if len(os.listdir(f'{path_dst}/{patient}/CT/')) != 0:
            print(f"Directory is not empty.: {path_dst}/{patient}/CT/\n")
            continue

        for i in range(up_patient, down_patient + 1):
            # Copy CT slices
            try:
                shutil.copy(path_patient + '/CT/I' + str(i) + '0.png',
                            f'{path_dst}/{patient}/CT/I{i}0.png', )
            except FileNotFoundError:
                print(f"File Not Found: {path_patient + '/CT/I' + str(i) + '0.png'}")

            # Copy PT slices
            try:
                shutil.copy(path_patient + '/PT/I' + str(i) + '0.png',
                            f'{path_dst}/{patient}/PT/I{i}0.png', )
            except FileNotFoundError:
                print(f"File Not Found: {path_patient + '/PT/I' + str(i) + '0.png'}")

    print("Slice Selection Completed!")


if __name__ == '__main__':
    SliceSelection(sys.argv[1], sys.argv[2], sys.argv[3])
