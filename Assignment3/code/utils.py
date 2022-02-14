import torch
import math
import numpy as np

def nan_to_zero(output, depth) :
    nanMask = torch.ne(depth,depth)
    valid_elements = torch.sum(torch.eq(depth,depth).float())
    valid_output = output.clone()
    valid_depth = depth.clone()

    valid_output[nanMask] = 0
    valid_depth[nanMask] = 0

    return valid_output, valid_depth, nanMask, valid_elements

def evaluate_error(output, depth) :
    rel_error = 0
    lg10_error = 0
    valid_output, valid_depth, nanMask, valid_elements = nan_to_zero(output, depth) 

    if (valid_elements.data.cpu().numpy() > 0) : #data valid
        diff_matrix = torch.abs(valid_output - valid_depth)
        lg10_matrix = torch.abs(torch.log10(valid_output) - torch.log10(valid_depth))

        rel_error = torch.sum(torch.div(diff_matrix, valid_depth))/valid_elements #mean relative error
        rel_error = float(rel_error.data.cpu().numpy())
        lg10_error = torch.sum(lg10_matrix)/valid_elements #mean log10 error
        lg10_error = float(lg10_error.data.cpu().numpy())
    return rel_error, lg10_error