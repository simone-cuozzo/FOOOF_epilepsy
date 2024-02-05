import numpy as  np
import pickle
from enigmatoolbox.utils.parcellation import parcel_to_surface, surface_to_parcel
from enigmatoolbox.plotting import plot_cortical

with open('data/plot_indeces_organization', 'rb') as file:
    new_indices = pickle.load(file)

############################################################################################
def reorganize_roi_gradient(original_data, new_indices=new_indices):
  '''
  A function that reorder the atlas parcels for plot purposes

  Arguments
  ---------
    original_data: a list.
      the original wrong ordered data, output of reorder_hemispheres function
    new_indices: a list.
      a list of the indeces in the new order
  Returns
  -------
    x: an array.
      the reordered data used for the cortical plot
  '''
  x = [original_data[i] for i in new_indices]
  return(np.array(x))
############################################################################################

############################################################################################
def reorder_hemispheres(original_data, first_left = True):
  '''
  A function that reorder the atlas parcels for plot purposes.

  Arguments
  ---------
    original_data: a list.
      the original wrong ordered data, output of reorder_hemispheres function
    first_left: a boolean.
      True if we want all left hemispheres parcels before right hemisphere ones,
      False otherwise
  Returns
  -------
    ordered_data: a list.
      the reordered data used for the cortical plot. This will become the input for
      the reorganize_roi_gradient function
  '''
  left_h = []
  right_h = []
  for idx, val in enumerate(original_data):
    idx +=1
    if idx%2==0:
      right_h.append(val)
    else:
      left_h.append(val)
  if first_left:
    ordered_data = np.concatenate((left_h, right_h))
  else:
    ordered_data = np.concatenate((right_h, left_h))
  return ordered_data.tolist()
############################################################################################