"""Stereo matching."""
import numpy as np
from scipy.signal import convolve2d


class Solution:
    def __init__(self):
        pass

    @staticmethod
    def ssd_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:
        """Compute the SSDD distances tensor.

        Args:
            left_image: Left image of shape: HxWx3, and type np.double64.
            right_image: Right image of shape: HxWx3, and type np.double64.
            win_size: Window size odd integer.
            dsp_range: Half of the disparity range. The actual range is
            -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

        Returns:
            A tensor of the sum of squared differences for every pixel in a
            window of size win_size X win_size, for the 2*dsp_range + 1
            possible disparity values. The tensor shape should be:
            HxWx(2*dsp_range+1).
        """
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range+1)
        ssdd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))
        win_offset = int(win_size/2)

        left_image_pad = np.pad(left_image, ((win_offset, win_offset), (win_offset, win_offset), (0, 0)))
        for disparity_idx, disparity in enumerate(disparity_values):
            shifted_right_image = np.zeros(right_image.shape)
            if disparity > 0:
                shifted_right_image[:, :-disparity, :] = right_image[:, disparity:, :]
            elif disparity == 0:
                shifted_right_image = right_image
            else:
                shifted_right_image[:, -disparity:, :] = right_image[:, :disparity, :]
            shifted_right_image_pad = np.pad(shifted_right_image, ((win_offset, win_offset), (win_offset, win_offset), (0, 0)))
            diff_image = left_image_pad - shifted_right_image_pad
            disparity_image = np.zeros((num_of_rows, num_of_cols))
            for color in range(left_image.shape[2]):
                disparity_image_sliding_sum = np.lib.stride_tricks.sliding_window_view(diff_image[:, :, color], (win_size, win_size))
                disparity_image_sliding_sum_power = np.power(disparity_image_sliding_sum, 2)
                disparity_image += np.sum(np.sum(disparity_image_sliding_sum_power, axis=2), axis=2)
            disparity_image /= 3  # TODO: check if avg is required
            ssdd_tensor[:, :, disparity_idx] = disparity_image

        ssdd_tensor -= ssdd_tensor.min()
        ssdd_tensor /= ssdd_tensor.max()
        ssdd_tensor *= 255.0
        return ssdd_tensor

    @staticmethod
    def naive_labeling(ssdd_tensor: np.ndarray) -> np.ndarray:
        """Estimate a naive depth estimation from the SSDD tensor.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.

        Evaluate the labels in a naive approach. Each value in the
        result tensor should contain the disparity matching minimal ssd (sum of
        squared difference).

        Returns:
            Naive labels HxW matrix.
        """
        # you can erase the label_no_smooth initialization.
        label_no_smooth = np.zeros((ssdd_tensor.shape[0], ssdd_tensor.shape[1]))
        """INSERT YOUR CODE HERE"""
        num_of_dis = ssdd_tensor.shape[2]
        label_no_smooth = ssdd_tensor.argmin(axis=2)
        return label_no_smooth

    @staticmethod
    def dp_grade_slice(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
        """Calculate the scores matrix for slice c_slice.

        Calculate the scores slice which for each column and disparity value
        states the score of the best route. The scores slice is of shape:
        (2*dsp_range + 1)xW.

        Args:
            c_slice: A slice of the ssdd tensor.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Scores slice which for each column and disparity value states the
            score of the best route.
        """
        # Get the number of columns and labels in the slice
        num_of_cols, num_labels = c_slice.shape[0], c_slice.shape[1]

        # Initialize the scores array
        scores = np.zeros((num_of_cols, num_labels))

        # Iterate through each column in the slice
        for i in range(num_of_cols):
            # Initialize the minimum score for this column to a large value
            min_score = float('inf')

            # Iterate through each label in the slice
            for j in range(num_labels):
                # Calculate the score for this label
                score = c_slice[i, j]

                # If this is the first column, there is no previous column to consider
                if i == 0:
                    scores[i, j] = score
                elif j == 0:
                    scores[i, j] = score
                else:
                    # Calculate the minimum score for the previous column
                    prev_min = min(scores[i - 1, max(0, j - 2):j + 3])

                    score1 = scores[i, j - 1]
                    # cond1 = scores(i, j - 1)
                    #
                    # # cond2
                    #
                    # cond2 = min([score(i - 1, j - 1), score(i + 1, j - 1)])  + p1
                    # else:
                    #     cond2 = cost(i - 1, j - 1) + p1
                    #
                    # # cond3
                    # tmp_ls = []
                    # for k in range(2,num_labels-i-2):
                    #     tmp_ls.append(cost(i + k, j - 1))
                    # if len(tmp_ls):
                    #     cond3  = min(tmp_ls)  + p2
                    #     a = min([cond1, cond2, cond3])
                    # else:
                    #     a = min([cond1, cond2])

                    # Add the penalty for the offset between the current label and the
                    # label with the minimum score in the previous column
                    # offset = abs(j - np.argmin(scores[i - 1, max(0, j - 2):j + 3]))
                    # if offset == 1:
                    if i + 1 < scores.shape[0]:
                        score2 = p1 + min([scores[i - 1, j - 1], scores[i + 1, j - 1]])
                    else:
                        score2 = p1 + scores[i - 1, j - 1]

                    tmp_ls=[]
                    for k in range(2,num_labels-i-2):
                        tmp_ls.append(scores[i + k, j - 1])
                    if len(tmp_ls):
                        score3  = min(tmp_ls)  + p2

                    scoreMin = min([score1, score2, score3])

                    # Add the minimum score from the previous column to the current score
                    scores[i, j] = score + min(scores[:,j-1]) + scoreMin

            # Update the minimum score for the current column
            min_score = min(min_score, min(scores[i, :]))

        return scores
    #
    def dp_labeling(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.
        """
        # Get the number of rows and columns in the tensor
        num_rows, num_cols = ssdd_tensor.shape[:2]

        # Initialize the scores tensor
        scores = np.zeros(ssdd_tensor.shape)

        # Iterate through each row in the tensor
        for i in range(num_rows):
            # Calculate the scores slice for this row
            scores_slice = Solution.dp_grade_slice(ssdd_tensor[i, :, :], p1, p2)

            # Store the scores slice in the scores tensor
            scores[i, :, :] = scores_slice

        # Initialize the depth map to all zeros
        depth_map = np.zeros((num_rows, num_cols))

        # Iterate through each pixel in the tensor
        for i in range(num_rows):
            for j in range(num_cols):
                # Find the label with the minimum score for this pixel
                min_label = np.argmin(scores[i, j, :])

                # Set the depth value for this pixel to the corresponding disparity value
                depth_map[i, j] = min_label

        return depth_map.astype(np.int32)

    def dp_labeling_per_direction(self,
                                  ssdd_tensor: np.ndarray,
                                  p1: float,
                                  p2: float) -> dict:
        """Return a dictionary of directions to a Dynamic Programming
        etimation of depth.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Dictionary int->np.ndarray which maps each direction to the
            corresponding dynamic programming estimation of depth based on
            that direction.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        direction_to_slice = {}
        """INSERT YOUR CODE HERE"""
        return direction_to_slice

    def sgm_labeling(self, ssdd_tensor: np.ndarray, p1: float, p2: float):
        """Estimate the depth map according to the SGM algorithm.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Semi-Global Mapping depth estimation matrix of shape HxW.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""
        return self.naive_labeling(l)

