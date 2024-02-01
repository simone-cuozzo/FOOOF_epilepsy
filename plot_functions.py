import numpy as  np
import pickle
from enigmatoolbox.utils.parcellation import parcel_to_surface, surface_to_parcel
from enigmatoolbox.plotting import plot_cortical

with open('plot_indeces_organization', 'rb') as file:
    new_indices = pickle.load(file)

def reorganize_roi_gradient(original_data, new_indices=new_indices):
  x = [original_data[i] for i in new_indices]
  return(np.array(x))

def fill(gradient):
  #crete an intermediate value in the color scale for median wall in cortical plots
  return (max(gradient) + min(gradient))/2


def reorder_hemispheres(original_data, first_left = True):
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