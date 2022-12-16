import os
import math
import numpy as np
import pandas as pd
import scipy
import dit
from dit.other import tsallis_entropy
import pywt
from kymatio import Scattering2D
import cv2 as cv
import medpy.io
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means


def get_filepaths(folderpath, MRtype):
    """Recursively find all .mha files in the folder.
    MRtype is a list containing elements in ['MR_Flair', 'MR_T1', 'MR_T1c', 'MR_T2', 'OT'].
    Return all filepaths as a list
    """
    if len(MRtype) == 0:
        return []
    for i in MRtype:
        if i not in ['MR_Flair', 'MR_T1', 'MR_T1c', 'MR_T2', 'OT']:
            return []

    filepaths = []
    for (root, dirs, files) in os.walk(folderpath, topdown=True):
        if len(files) > 0:
            for i in files:
                if i[-4:] == '.mha' and i.split(".")[-3] in MRtype:
                    filepaths.append(os.path.join(root, i))
    filepaths.sort()
    return filepaths


def get_entropy(conti_data, q=5):
    """
    conti_data: 1D array of continuous data
    q: parameter to perform ccut to discretize the data
    """

    conti_data = np.array(conti_data).reshape(-1)
    data_df = pd.DataFrame(conti_data[:, np.newaxis], columns=['data'])
    data_df['cut'] = pd.cut(data_df['data'], q, labels=False, duplicates="drop")

    data_df['count'] = 0
    counts = data_df.groupby(by='cut').count()['count'].values
    total = len(conti_data)
    entropy = 0
    for i in counts:
        if i == 0 or i == total:
            continue
        prob = i / total
        entropy -= prob * math.log(prob, 2)

    return [entropy, counts]


def get_recommended_slices_id(OT_paths, file_id):
    """
    OT_paths: a list of file paths
    file_id: index in the OT_paths
    Return the recommended slice range [min_slice_id, max_slice_id] (inclusive)
    """
    # extract target slices in the current file
    tumor_data, _ = medpy.io.load(OT_paths[file_id])
    # tumor_area = [np.sum(tumor_data[:, :, i] > 0) for i in range(tumor_data.shape[2])]
    # tumor_area = np.array(tumor_area)
    # cutoff_threshold = np.quantile(tumor_area[tumor_area > 50], .25)
    # valid_slices = np.where(tumor_area >= cutoff_threshold)[0]
    # min_slice_id = valid_slices[0]  # inclusive
    # max_slice_id = valid_slices[-1]  # inclusive
    # find Z
    Z = 0
    max_sum = 0
    for i in range(0, tumor_data.shape[2]):
        cur_sum = np.sum(tumor_data[:, :, i])
        if cur_sum > max_sum:
            max_sum = cur_sum
            Z = i
    # return [min_slice_id, max_slice_id]
    return Z


def extract_feature_one_slice(OT_paths, MR_paths, file_id, slice_id):
    """
    MR_paths: a list of file paths
    file_id: index in the MR_pahts
    slice_id: index of slice in the file
    Return a 1D array of features
    """

    tumor_data, _ = medpy.io.load(OT_paths[file_id])
    mask = tumor_data[:, :, slice_id] != 0
    mask = np.array(mask).astype(int)
    image_data, _ = medpy.io.load(MR_paths[file_id])
    image_data = image_data[:, :, slice_id]
    image_data = image_data.astype(float)
    image_data_denoised = denoise_nl_means(image_data, h=15, fast_mode=True, preserve_range=True)  # denoise

    # Perform k-means clustering and plot the clusters
    # data = image_data_denoised.reshape((-1, 1))
    # data = np.float32(data)
    # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 4.0)
    # K = 15
    # ret, label, center = cv.kmeans(data, K, None, criteria, 100, cv.KMEANS_PP_CENTERS)
    # label += 1

    # Select the cluster for the tumor
    # index = np.argmax(center)  # index of the tumor cluster
    # temp_label = label.copy()
    # temp_label[temp_label != index + 1] = 0
    # res2 = temp_label.reshape(image_data.shape)
    # tumor_region_data = data.reshape(-1)
    # tumor_region_data[(temp_label == 0).reshape(-1)] = 0
    # tumor_region_data = tumor_region_data.reshape(image_data.shape)

    # index = np.array(center[:, 0]).argsort()[-10:][::-1]
    # std = np.std(center[index])
    # diff = 2.3842 * std - 17.5672
    # index = np.array([i for i in index if center[i] > center[index[0]] - diff])
    # temp_label = label.copy()
    # temp_label = np.array([1 if i in index + 1 else 0 for i in temp_label])
    # mask = temp_label.reshape(image_data.shape)
    #
    # kernel = np.ones((2, 2), np.uint8)
    # mask = cv.morphologyEx(mask.astype('uint8'), cv.MORPH_OPEN, kernel)
    # kernel = np.ones((2, 2), np.uint8)
    # mask = cv.morphologyEx(mask.astype('uint8'), cv.MORPH_CLOSE, kernel)

    tumor_region_data = np.ma.masked_array(image_data, mask)

    # Extract feature - part I
    data = tumor_region_data
    # feature extraction - part I
    non_zero_data = data.reshape(-1)
    non_zero_data = non_zero_data[non_zero_data > 0]
    skewness = scipy.stats.skew(non_zero_data, bias=True)
    kurtosis = scipy.stats.kurtosis(non_zero_data, fisher=True, bias=True)
    variance = scipy.stats.variation(non_zero_data, ddof=1)
    mean = np.mean(non_zero_data)
    [entropy, _] = get_entropy(non_zero_data, q=5)
    feature = [skewness, kurtosis, variance, mean, entropy]  # feature - part I

    # Extract feature - part II
    data = tumor_region_data  # tumor region data
    wavelet = 'coif5'  # chosen wavelet
    discretize_num = 10
    tsallis_entropy_order = 0.5

    # the first level
    coeffs = pywt.dwt2(data, wavelet=wavelet)
    first_level = [coeffs[0], coeffs[1][0], coeffs[1][1], coeffs[1][2]]

    # the second level
    second_level = []
    for i in first_level:
        coeffs = pywt.dwt2(i, wavelet=wavelet)
        second_level.append(coeffs[0])
        second_level.append(coeffs[1][0])
        second_level.append(coeffs[1][1])
        second_level.append(coeffs[1][2])

    # the thrid level
    third_level = []
    for i in second_level:
        coeffs = pywt.dwt2(i, wavelet=wavelet)
        third_level.append(coeffs[0])
        third_level.append(coeffs[1][0])
        third_level.append(coeffs[1][1])
        third_level.append(coeffs[1][2])

    # calculate tsallis entropy
    tsallis_entropy = []
    for i in third_level:
        [_, probs] = get_entropy(i, q=discretize_num)
        probs = probs / np.sum(probs)
        d = dit.Distribution([str(j) for j in range(len(probs))], probs)
        tsallis_entropy.append(dit.other.tsallis_entropy(d, order=tsallis_entropy_order))  # feature - part II
    feature = feature + tsallis_entropy

    # Extract feature III
    # Set the parameters of the scattering transform.
    J = int(np.floor(np.math.log(240, 2)))  # 7
    L = 8
    M, N = 240, 240

    # Define a Scattering2D object.
    S = Scattering2D(J, (M, N), L=8)

    # Calculate the scattering transform.
    Sx = S.scattering(data)
    feature = feature + Sx.reshape(-1).tolist()
    return np.array(feature)
