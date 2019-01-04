from skimage.feature import greycomatrix, greycoprops
import numpy as np
from typing import Any
from skimage.measure import shannon_entropy
from .enhancement import rescale

glcm = greycomatrix
gcpr = greycoprops
feature_names = ['contrast', 'dissimilarity', 'homogeneity',
                 'ASM', 'energy', 'correlation',
                 'sum_squares_var', 'inverse_diff_moment',
                 'sum_average', 'sum_var', 'sum_entropy',
                 'entropy', 'diff_var', 'diff_entropy',
                 'info_measure_corr_1', 'info_measure_corr_2',
                 'max_corr_coeff']


def glcm_variance(matrix: np.ndarray, eps: float = 10e-3) -> float:
    """
    :float eps:  ignored, it is here for consistency,
    """
    mu = np.mean(matrix)
    var = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            var += (i - mu) ** 2 * matrix[i, j]
    return var


def glcm_idm(matrix: np.ndarray) -> float:
    idm: int = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            idm += matrix[i, j] / (1 + (i - j) ** 2)
    return idm


def _p_sum_xy(matrix: np.ndarray) -> np.ndarray:
    ks = np.arange(0.0, 510.0, 1.0) + 2.0
    res = np.zeros_like(ks)
    # indices of nonzero elements in matrix
    nonzero_indices = np.transpose(np.where(matrix > 0))
    # pairs of indices and their sums
    useful_pairs = [(sum(i), i) for i in nonzero_indices]
    for each in useful_pairs:
        if each[0] in ks:
            res[each[0]-2] += matrix[tuple(each[1])]
    return res


def _p_diff_xy(matrix: np.ndarray) -> np.ndarray:
    ks = np.arange(0.0, 255.0, 1.0)
    res = np.zeros_like(ks)
    # indices of nonzero elements in matrix
    nonzero_indices = np.transpose(np.where(matrix > 0))
    # pairs of indices and their sums
    useful_pairs = [(abs(i - j), (i, j)) for (i, j) in nonzero_indices]
    for each in useful_pairs:
        if each[0] in ks:
            res[each[0]] += matrix[tuple(each[1])]
    return res


def glcm_sum_average(matrix: np.ndarray, eps: float = 10e-3) -> float:
    """
    :float eps: is here for consistency
    """
    p_sum_xy = _p_sum_xy(matrix)
    sm = 0
    for i in range(len(p_sum_xy)):
        sm += i * p_sum_xy[i]
    return sm


def glcm_sum_entropy(matrix: np.ndarray, eps: float = 10e-3) -> float:
    p_sum_xy = _p_sum_xy(matrix)
    entr = 0
    for i in range(len(p_sum_xy)):
        entr += -p_sum_xy[i] * np.log(eps + p_sum_xy[i])
    return entr


def glcm_sum_variance(matrix: np.ndarray,
                      eps: float = 10e-3,
                      entr: Any = 'compute') -> float:
    p_sum_xy = _p_sum_xy(matrix)
    sm = 0
    if entr == 'compute':
        entr = glcm_sum_entropy(matrix, eps)
        for i in range(len(p_sum_xy)):
            sm += (i - entr) ** 2 * p_sum_xy[i]
    else:
        for i in range(len(p_sum_xy)):
            sm += (i - entr) ** 2 * p_sum_xy[i]
    return sm


def glcm_entropy(matrix: np.ndarray, eps: float = 10e-3) -> np.ndarray:
    return -np.sum(matrix.ravel() * np.log(eps + matrix.ravel()))


def glcm_diff_variance(matrix: np.ndarray, eps: float = 10e-3) -> np.ndarray:
    return np.var(_p_diff_xy(matrix))


def glcm_diff_entropy(matrix: np.ndarray, eps: float = 10e-3) -> np.ndarray:
    p_diff_xy = _p_diff_xy(matrix)
    entr = 0
    for i in range(len(p_diff_xy)):
        entr += -p_diff_xy[i] * np.log(eps + p_diff_xy[i])
    return entr


def glcm_infmescor_1(matrix: np.ndarray, eps: float = 10e-3) -> np.ndarray:
    """
    Measures of Correlation
    """
    hxy = -np.sum(matrix * np.log(eps + matrix))
    hxy_1 = -np.sum(matrix * np.log(eps +
                                    np.sum(matrix, axis=0) *
                                    np.sum(matrix, axis=1)))
    hx = shannon_entropy(np.sum(matrix, axis=0))
    hy = shannon_entropy(np.sum(matrix, axis=1))
    measure_1 = (hxy - hxy_1) / max(hx, hy)

    # measure_2 = np.sqrt(np.abs(1-np.exp(-2*(hxy_2-hxy))))

    return measure_1  # , measure_2


def glcm_infmescor_2(matrix: np.ndarray, eps: float = 10e-3) -> np.ndarray:
    """
    Measures of Correlation
    """
    hxy = -np.sum(matrix * np.log(eps + matrix))
    hxy_2 = -np.sum(np.sum(matrix, axis=0) * np.sum(matrix, axis=1) *
                    np.log(eps + np.sum(matrix, axis=0) *
                           np.sum(matrix, axis=1)))
    measure_2 = np.sqrt(np.abs(1 - np.exp(-2 * (hxy_2 - hxy))))

    return measure_2


def glcm_max_cor_coef(matrix: np.ndarray, eps: float = 1.0) -> float:
    px = np.sum(matrix, axis=0).reshape((-1, 1))
    py = np.sum(matrix, axis=1).reshape((-1, 1))
    prod = (px.T*py)
    q = np.matmul(matrix.T, matrix)/(1+prod)
    eig = np.sqrt(np.abs(np.sort(np.linalg.eigvals(q))[1]))
    return eig


# noinspection PyArgumentEqualDefault
def feature_extraction_har(image: np.ndarray, levels: int = 256,
                           distances=None,
                           angles=None) -> np.ndarray:
    """Haralick features extraction"""
    if distances is None:
        distances = [1]
    if angles is None:
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    _feature_names = ('contrast', 'dissimilarity', 'homogeneity', 'ASM',
                      'energy', 'correlation',
                      'sum_squares_var', 'inverse_diff_moment',
                      'sum_average', 'sum_var', 'sum_entropy',
                      'entropy', 'diff_var', 'diff_entropy',
                      'info_measure_corr_1', 'info_measure_corr_2',
                      'max_corr_coeff')
    image = rescale(image, 0, 255).astype('uint8')
    features_df = np.zeros(len(distances) * len(angles) * len(_feature_names))
    matrices = glcm(image, levels=levels, normed=True,
                    distances=distances, angles=angles)
    props = ['contrast', 'dissimilarity',
             'homogeneity', 'ASM', 'energy', 'correlation']
    features_df[:len(distances) * len(angles) * len(props)] = np.concatenate(
        [gcpr(matrices, prop,).ravel() for prop in props])
    k = 0
    for i, feat in enumerate([glcm_variance, glcm_idm, glcm_sum_average,
                              glcm_sum_variance, glcm_sum_entropy,
                              glcm_entropy, glcm_diff_variance,
                              glcm_diff_entropy, glcm_infmescor_1,
                              glcm_infmescor_2, glcm_max_cor_coef]):
        for distance in range(len(distances)):
            for angle in range(len(angles)):
                feature = feat(matrices[:, :, distance, angle])
                if not np.isnan(feature):
                    features_df[len(distances) * len(angles)
                                * len(props) + k] = feature
                else:
                    features_df[len(distances) * len(angles)
                                * len(props) + k] = 0
                # print(feat(matrices[:,:,distance, angle]))
                k += 1

    return features_df
