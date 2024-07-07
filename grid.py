import numpy as np
import torch
import itertools
from torch.autograd import Variable


def getGridMask(frame, dimensions, num_person, neighborhood_size, grid_size, is_occupancy = False):
    '''
    This function computes the binary mask that represents the
    occupancy of each ped in the other's grid
    params:
    frame : This will be a MNP x 3 matrix with each row being [pedID, x, y]
    dimensions : This will be a list [width, height]
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    num_person : number of people exist in given frame
    is_occupancy: A flag using for calculation of accupancy map

    '''
    mnp = num_person

    width, height = dimensions[0], dimensions[1]
    if is_occupancy:
        frame_mask = np.zeros((mnp, grid_size**2))
    else:
        frame_mask = np.zeros((mnp, mnp, grid_size**2))
    frame_np =  frame.data.numpy()

    #width_bound, height_bound = (neighborhood_size/(width*1.0)), (neighborhood_size/(height*1.0))
    width_bound, height_bound = (neighborhood_size/(width*1.0))*2, (neighborhood_size/(height*1.0))*2
    #print("weight_bound: ", width_bound, "height_bound: ", height_bound)

    #instead of 2 inner loop, we check all possible 2-permutations which is 2 times faster.
    list_indices = list(range(0, mnp))
    # P(list_indices, 2)를 추출함.
    # ex)list_indcies가 3 -> (0,1),(0,2),(1,0),(1,2),(2,0),(2,1)의 조합이 생김
    # (0, 1) -> 0번사람과 이웃후보1번 사람의 관계를 파악하고 이웃의 조건에 충족되면 cell에 1를 부여
    for real_frame_index, other_real_frame_index in itertools.permutations(list_indices, 2):
        current_x, current_y = frame_np[real_frame_index, 0], frame_np[real_frame_index, 1]

        width_low, width_high = current_x - width_bound/2, current_x + width_bound/2
        height_low, height_high = current_y - height_bound/2, current_y + height_bound/2

        other_x, other_y = frame_np[other_real_frame_index, 0], frame_np[other_real_frame_index, 1]
        
        #if (other_x >= width_high).all() or (other_x < width_low).all() or (other_y >= height_high).all() or (other_y < height_low).all():
        if (other_x >= width_high) or (other_x < width_low) or (other_y >= height_high) or (other_y < height_low):
                # Ped not in surrounding, so binary mask should be zero
                #print("not surrounding")
                continue
        # If in surrounding, calculate the grid cell
        cell_x = int(np.floor(((other_x - width_low)/width_bound) * grid_size))
        cell_y = int(np.floor(((other_y - height_low)/height_bound) * grid_size))

        if cell_x >= grid_size or cell_x < 0 or cell_y >= grid_size or cell_y < 0:
                continue
    
        if is_occupancy:
            frame_mask[real_frame_index, cell_x + cell_y*grid_size] = 1
        else:
            # Other ped is in the corresponding grid cell of current ped
            frame_mask[real_frame_index, other_real_frame_index, cell_x + cell_y*grid_size] = 1

    #Two inner loops aproach -> slower
    # # For each ped in the frame (existent and non-existent)
    # for real_frame_index in range(mnp):
    #     #real_frame_index = lookup_seq[pedindex]
    #     #print(real_frame_index)
    #     #print("****************************************")
    #     # Get x and y of the current ped
    #     current_x, current_y = frame[real_frame_index, 0], frame[real_frame_index, 1]

    #     #print("cur x : ", current_x, "cur_y: ", current_y)

    #     width_low, width_high = current_x - width_bound/2, current_x + width_bound/2
    #     height_low, height_high = current_y - height_bound/2, current_y + height_bound/2
    #     #print("width_low : ", width_low, "width_high: ", width_high, "height_low : ", height_low, "height_high: ", height_high)


    #     # For all the other peds
    #     for other_real_frame_index in range(mnp):
    #         #other_real_frame_index = lookup_seq[otherpedindex]
    #         #print(other_real_frame_index)


    #         #print("################################")
    #         # If the other pedID is the same as current pedID
    #         if other_real_frame_index == real_frame_index:
    #             # The ped cannot be counted in his own grid
    #             continue

    #         # Get x and y of the other ped
    #         other_x, other_y = frame[other_real_frame_index, 0], frame[other_real_frame_index, 1]
    #         #print("other_x: ", other_x, "other_y: ", other_y)
    #         if (other_x >= width_high).all() or (other_x < width_low).all() or (other_y >= height_high).all() or (other_y < height_low).all():
    #             # Ped not in surrounding, so binary mask should be zero
    #             #print("not surrounding")
    #             continue

    #         # If in surrounding, calculate the grid cell
    #         cell_x = int(np.floor(((other_x - width_low)/width_bound) * grid_size))
    #         cell_y = int(np.floor(((other_y - height_low)/height_bound) * grid_size))
    #         #print("cell_x: ", cell_x, "cell_y: ", cell_y)

    #         if cell_x >= grid_size or cell_x < 0 or cell_y >= grid_size or cell_y < 0:
    #             continue

    #         # Other ped is in the corresponding grid cell of current ped
    #         frame_mask[real_frame_index, other_real_frame_index, cell_x + cell_y*grid_size] = 1
    # #print("frame mask shape %s"%str(frame_mask.shape))

    return frame_mask

def getSequenceGridMask(sequence, dimensions, pedlist_seq, neighborhood_size, grid_size, using_cuda, is_occupancy=False):
    '''
    Get the grid masks for all the frames in the sequence
    params:
    sequence : A numpy matrix of shape SL x MNP x 3
    dimensions : This will be a list [width, height]
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    using_cuda: Boolean value denoting if using GPU or not
    is_occupancy: A flag using for calculation of accupancy map
    '''
    sl = len(sequence)
    sequence_mask = []

    for i in range(sl):
        # if not is_occupancy, getGridMask.shape = 이웃수(N) x 이웃수(N) x grid_size x grid_size
        # if is_occupancy, getGridMask.shape = 이웃수(N) x grid_size x grid_size
        # getGridMask : 보행자 - 이웃이 조건에 만족하면(가까우면), maks[보행자id][이웃id][이웃x][이웃y] = 1를 부여
        # is_occupancy과 다른 이유는 특정 grid_cell x grid_cell에 이웃이 여러명일때 OSLTM은 (grid_cell, grid_cell)만 주지만,
        # Social-LSTM은 해당 셀의 여러 이웃의 모든 정보를 추출해야하기 때문에 중복되게 저장을 방지함
        mask = Variable(torch.from_numpy(getGridMask(sequence[i], dimensions, len(pedlist_seq[i]), neighborhood_size, grid_size, is_occupancy)).float())

        if using_cuda:
            mask = mask.cuda()
        sequence_mask.append(mask)

    return sequence_mask
