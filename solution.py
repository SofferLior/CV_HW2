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
        num_of_cols, num_labels = c_slice.shape[0], c_slice.shape[1]
        c_slice = c_slice.T
        l_slice = np.zeros((num_labels, num_of_cols))
        """INSERT YOUR CODE HERE"""

        def cost(i, j):
            if Map[i][j] != 0:
                return Map[i][j]
            a = 0
            # if j == 0:
            #     a = i * Occlusion
            # elif i == 0:
            #     a = j * Occlusion
            # else:
            #     a = min([cost(i - 1, j - 1) + c_slice(i, j),
            #              cost(i, j - 1) + Occlusion,
            #              cost(i - 1, j) + Occlusion])

            if j == 0:
                l = c_slice[i][j]
            elif i == 0:
                l = c_slice[i][j]
            else:
                ### a = M[i,j]
                a = min([cost(i, j - 1),
                         min([cost(i - 1, j - 1), cost(i + 1, j - 1)])  + p1,
                         min([cost(i + k, j - 1) for k in range(2,num_labels-i-2) if i<num_labels-2])  + p2])

                l = a + c_slice[i][j] - min(Map[:, j-1])

            return round(l, 3)

        Map = l_slice


        for i in range(1, num_labels):
            Map[i][0] = cost(i, 0)
        for j in range(1, num_of_cols):
            Map[0][j] = cost(0, j)
        for inx_lab in range(1, num_labels):
            for inx_col in range(1, num_of_cols):
                a = cost(inx_lab, inx_col)
                Map[inx_lab][inx_col] = a

        l_slice = Map
        return l_slice

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
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""

        for row in range(ssdd_tensor.shape[0]):
            Solution.dp_grade_slice(ssdd_tensor[row],p1,p2)

        return self.naive_labeling(l)

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

