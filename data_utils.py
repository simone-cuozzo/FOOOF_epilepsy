import numpy as np
import scipy
from fooof import FOOOF
import math
from tqdm import tqdm
from brainspace.gradient import GradientMaps
import enigmatoolbox
from pyentrp import entropy as ent
from mne.stats import fdr_correction
import pandas as pd
from scipy.stats import ttest_ind
from tqdm import tqdm

##########################################################################################
def permutation_ttest(matrix_EPI, matrix_HC, num_permutations = 5000, num_variables = 68):
    ''' 
    Perform a permutation t-test between two matrices (matrix_EPI, matrix_HC) for n permutations

    Arguments
    ---------
      matrix_EPI, matrix_HC: a list or an array. 
            The arrays must have the same first or second dimension of size num_variables
      num_permutations: an int.
            Number of permutations for the t-test
      num_variables: an int. 
            Number of variables on which the t-test is performed, that is the number of cortical parcels in the Desikan-Killiany atlas
    Returns
    -------
      actual_t_stats : a list.
        The calculated t-statistic from the 2 original matrices 
      permuted_p_values : a list.
        The p-value obtained from the permutations

    '''
    if np.shape(matrix_EPI)[1] != num_variables:
        matrix_EPI = np.transpose(matrix_EPI)
        if np.shape(matrix_EPI)[1] != num_variables:
            print("Matrix A has an incorrect shape for Desikan-Killiany Atlas")
    if np.shape(matrix_HC)[1] != num_variables:
        matrix_HC = np.transpose(matrix_HC)
        if np.shape(matrix_HC)[1] != num_variables:
            print("Matrix A has an incorrect shape for Desikan-Killiany Atlas")
    actual_t_stats = np.zeros(num_variables)
    permuted_p_values = np.zeros(num_variables)
    # Perform permutation t-test for each variable
    for variable_index in tqdm(range(num_variables), desc=f"ROIs EPI vs HC permuted t-value calculation"):
        # Extract the specific column from each matrix
        data_EPI = np.array(matrix_EPI)[:, variable_index]
        data_HC = np.array(matrix_HC)[:, variable_index]
        
        # Calculate the actual t-statistic for the original samples
        actual_t_stat, _ = ttest_ind(data_EPI, data_HC, equal_var=False)
        actual_t_stats[variable_index] = actual_t_stat
        # Initialize an array to store the permuted t-statistics
        permuted_t_stats = np.zeros(num_permutations)

        # Perform permutation test
        combined_data = np.concatenate([data_EPI, data_HC])
        for i in range(num_permutations):
            np.random.shuffle(combined_data)
            permuted_t_stats[i], _ = ttest_ind(combined_data[:len(data_EPI)], combined_data[len(data_EPI):])

        # Calculate the p-value by comparing the actual t-statistic with the permuted t-statistics
        perm_p_value = (np.abs(permuted_t_stats) >= np.abs(actual_t_stat)).mean()
        permuted_p_values[variable_index] = perm_p_value
    return actual_t_stats, permuted_p_values
############################################################################################

############################################################################################
'''def time_series_concat(data_mat, concat_data = list):
  for sub in data_mat:
    shape1 = np.shape(sub[0])[1]
    shape2 = np.shape(sub[1])[1]
    desired_len = 100000
    shape2_limit = desired_len - shape1
    time_series = np.concatenate((sub[0], sub[1][:, :shape2_limit]), axis=1)
    limited_time_series = time_series
    concat_data.append(limited_time_series)'''
############################################################################################

############################################################################################
def fooof_exponent_compute(data,  freq_range = list):
  '''
  calculate the power spectra of a time series matrix and use the fooof library to calculate the FOOOF exponents 

  Arguments
  ---------
    data: a list or an array.
      matrix of shape (n_subject x n_parcels) containing the time serie of each parcel of the Desikan-Killiany atlas for each subject
    freq_range: a list.
      contains the lower bound and the upper bound of the frequency range on which the FOOOF is calculated
  Returns
  -------
    fooof_dict: a dict.
      a dictionary that contains the subject ids as keys and the fooof exponents as values
  '''
  fooof_dict = {}
  for idx,sub in enumerate(data):
    fooof_exponent = []
    dict_key = 'sub' + str(idx+1)
    for i in range(len(sub)):
      f, Pxx_den = scipy.signal.welch(sub[i], fs=250, nperseg=1250, scaling = 'density')
      fm = FOOOF()
      fm.fit(freqs=f, power_spectrum=Pxx_den, freq_range=freq_range)
      #fooof_offset.append(fm.aperiodic_params_[0])
      fooof_exponent.append(fm.aperiodic_params_[1])
    fooof_dict[dict_key] = fooof_exponent
  print('FOOOF exponents calculation completed')
  return(fooof_dict)
############################################################################################

############################################################################################
def power_spectrum_and_aperiodic(data):
  '''
  Calculate the power spectra of a time series matrix with Welch alghorithm 

  Arguments
  ---------
    data: an array or list.
      matrix containing the subjects time series
  Returns
  -------
    power_spec_dict: a dict of tuples.
      a dict containing the frequencies and power for each subject
  '''
  power_spec_dict = {}
  for idx,sub in enumerate(data):
    frequencies = []
    power_spectrum = []
    dict_key = 'sub' + str(idx+1)
    for i in range(len(sub)):
      f, Pxx_den = scipy.signal.welch(sub[i], fs=250, nperseg=1250, scaling='density')
      frequencies.append(f)
      power_spectrum.append(Pxx_den)
    power_spec_dict[dict_key] = (frequencies, power_spectrum)
  return power_spec_dict
############################################################################################

############################################################################################
def nan_sanity_check(data_dict, group_name = str):
  '''
  For each subject in a dict check if a NaN value is present due to an error after FOOOF calculation

  Arguments
  ---------
    data_dict: a dict.
      a dict containing the parcels' FOOOF exponent for each group subject
    group_name: a string.
      the name of the group on which the function is operating
  Returns
  -------
    a printed string.
  '''
  for key, roi in data_dict.items():
    for idx, value in enumerate(roi):
      # Check if the value is NaN
      nan_check = False
      if isinstance(value, float) and math.isnan(value):
          nan_check = True
          print(f"The value at key '{key}' and roi #{idx} in {group_name} group is NaN.")
  if nan_check == False:
      print(f"No Nan values present in group {group_name}") 
############################################################################################
      
'''
############################################################################################
def p_value_epi_vs_hc(epi_data, hc_data, alpha = 0.05, fdr=False):
  
  Calculate the ROI-wise p-value between EPI group subjects and HC subjects

  Arguments
  ---------
    epi_data: a dict.
      a dict containing the parcels' FOOOF exponent for each group subject
    group_name: a string.
      the name of the group on which the function is operating
  Returns
  -------
    a printed string.
  
  # Function to calculate the ROI-wise p-value between subject an control groups 
  epi_data = np.array(epi_data)
  hc_data = np.array(hc_data)
  # transpose the data matrix if the first dimension is not the ROIs one
  if epi_data.shape[0] != 68:
    epi_data = epi_data.T
  if hc_data.shape[0] != 68:
    hc_data = hc_data.T
  # Two different procedures if FDR correction is applied or not
  if fdr == True:
    t_values, p_values = scipy.stats.ttest_ind(epi_data, hc_data, axis=1, equal_var=False)
    # Apply FDR correction
    _, corrected_p_values = fdr_correction(p_values, alpha=alpha, method="indep")
    roi_significant_p = (corrected_p_values <= alpha).astype(int)
    t_test_results_mask = t_values * roi_significant_p
    return t_values, corrected_p_values, t_test_results_mask
  else:
    t_values, p_values = scipy.stats.ttest_ind(epi_data, hc_data, axis=1, equal_var=False)
    roi_significant_p = (p_values <= alpha).astype(int)
    t_test_results_mask = t_values * roi_significant_p
  return t_values, p_values, t_test_results_mask
############################################################################################
'''

############################################################################################
def calculate_correlation(fooof_mat, score_vec, type = str, alpha = 0.01, fdr = False):
  '''
  Calculate the correlation between FOOOF exponents and clinical/neuropsychological scores

  Arguments
  ---------
    fooof_mat: an array or list.
      a matrix that contains the FOOOF exponents
    score_vec: an array or list.
      a vector that contains the clinical variable values or neuropsychological scores
    type: a string.
      specifies the type of correlation performed 
    group_name: a string.
      the name of the group on which the function is operating
    alpha: a float.
      the threshold for the p-values. Default is 0.01
    fdr: a boolean.
      defines if Roi-wise FDR correction is applied
  Returns
  -------
    correlation_vec: a list.
      contains Pearson/Spearman correlation coefficients
    p_values_vec/p_values_vec_corrected: a list.
      contains the p-values 
    corr_mask: a list.
      a mask that contains the correlation value if parcel p-value <= alpha
      and 0 if parcel p-value > alpha
    '''
  p_values_vec = []
  correlation_vec = []
  fooof_mat, score_vec = np.array(fooof_mat), np.array(score_vec)
  if np.shape(fooof_mat)[0] != 68:
    fooof_mat = fooof_mat.T
  for roi in fooof_mat:
    if type == "pearson":
      correlation, p_values = scipy.stats.pearsonr(roi, score_vec)
      correlation_vec.append(correlation)
      p_values_vec.append(p_values)
    elif type == 'spearman':
      correlation, p_values = scipy.stats.spearmanr(roi, score_vec)
      correlation_vec.append(correlation)
      p_values_vec.append(p_values)
  if fdr == True:
    reject, p_values_vec_corrected = fdr_correction(p_values_vec, alpha=alpha, method="indep")
    corr_mask = reject * correlation_vec
    return correlation_vec, p_values_vec_corrected, corr_mask
  else:
    significant_rois = (np.array(p_values_vec) <= alpha).astype(int)
    corr_mask = significant_rois * correlation_vec
    return correlation_vec, p_values_vec, corr_mask
############################################################################################

############################################################################################
def create_plot_mask(corr_values, p_values, alpha):
  '''
  Create the cortical masks used for brain plotting

  Arguments
  ---------
    corr_values: a list.
      correlation values list
    p_values: a list.
      p-values list
    alpha: a float.
      threshold for p-values
  Returns
  -------
    corr_mask: a list.
      a mask that contains the correlation value if p-value <= alpha
      and 0 if p-value > alpha
  '''
  significant_rois = (np.array(p_values) <= alpha).astype(int)
  corr_mask = significant_rois * corr_values
  return corr_mask
############################################################################################