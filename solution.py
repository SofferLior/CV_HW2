"""Stereo matching."""
import numpy as np
from scipy.signal import convolve2d


def create_diagonal_slice(ssdd_tensor):
    slices = []
    for row in range(1, ssdd_tensor.shape[0]):
        slices.append(ssdd_tensor.diagonal(-row))
    slices.append(ssdd_tensor.diagonal())
    for col in range(1, ssdd_tensor.shape[1]):
        slices.append(ssdd_tensor.diagonal(col))
    return slices


def create_slice_for_direction(ssdd_tensor, direction):
    slices = []
    if direction == 1 or direction == 5:
        for row in range(ssdd_tensor.shape[0]):
            slices.append(ssdd_tensor[row, :, :])
    elif direction == 3 or direction == 7:
        for col in range(ssdd_tensor.shape[1]):
            slices.append(ssdd_tensor[:, col, :])
    elif direction == 2:
        slices = create_diagonal_slice(ssdd_tensor)
    elif direction == 4:
        slices = create_diagonal_slice(np.fliplr(ssdd_tensor))
    elif direction == 6:
        slices = create_diagonal_slice(np.flipud(np.fliplr(ssdd_tensor)))
    elif direction == 8:
        slices = create_diagonal_slice(np.flipud(ssdd_tensor))
    return slices


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
        disparity_values = range(-dsp_range, dsp_range + 1)
        ssdd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))
        win_offset = int(win_size / 2)

        left_image_pad = np.pad(left_image, ((win_offset, win_offset), (win_offset, win_offset), (0, 0)))
        for disparity_idx, disparity in enumerate(disparity_values):
            shifted_right_image = np.zeros(right_image.shape)
            if disparity > 0:
                shifted_right_image[:, :-disparity, :] = right_image[:, disparity:, :]
            elif disparity == 0:
                shifted_right_image = right_image
            else:
                shifted_right_image[:, -disparity:, :] = right_image[:, :disparity, :]
            shifted_right_image_pad = np.pad(shifted_right_image,
                                             ((win_offset, win_offset), (win_offset, win_offset), (0, 0)))
            diff_image = left_image_pad - shifted_right_image_pad
            disparity_image = np.zeros((num_of_rows, num_of_cols))
            for color in range(left_image.shape[2]):
                disparity_image_sliding_sum = np.lib.stride_tricks.sliding_window_view(diff_image[:, :, color],
                                                                                       (win_size, win_size))
                disparity_image_sliding_sum_power = np.power(disparity_image_sliding_sum, 2)
                disparity_image += np.sum(np.sum(disparity_image_sliding_sum_power, axis=2), axis=2)
            disparity_image /= 3
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
        num_labels, num_of_cols = c_slice.shape[0], c_slice.shape[1]
        l_slice = np.zeros((num_labels, num_of_cols))

        # Iterate through each column in the slice
        for col in range(num_of_cols):

            # Iterate through each label in the slice
            for d in range(num_labels):
                # Calculate the score for this label
                score = c_slice[d, col]

                # If this is the first column, there is no previous column to consider
                if col == 0:
                    l_slice[d, col] = score
                else:
                    # Score 1 - previous col, same d
                    score1 = l_slice[d, col - 1]

                    # Score 2 - previous col, d +-1
                    if d + 1 == num_labels:
                        score2 = p1 + l_slice[d - 1, col - 1]
                    elif d == 0:
                        score2 = p1 + l_slice[d + 1, col - 1]
                    else:
                        score2 = p1 + min(l_slice[d - 1, col - 1], l_slice[d + 1, col - 1])

                    tmp_ls = []
                    for k in range(num_labels):
                        if k != d and k != d + 1 and k != d - 1:
                            tmp_ls.append(l_slice[k, col - 1])
                    score3 = p2 + min(tmp_ls)

                    l_slice[d, col] = score + min([score1, score2, score3]) - min(l_slice[:, col - 1])

        return l_slice

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
        l = np.zeros_like(ssdd_tensor)

        for i in range(ssdd_tensor.shape[0]):
            # Calculate the scores slice for this row
            l[i, :, :] = self.dp_grade_slice(ssdd_tensor[i, :, :].T, p1, p2).T

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
        for direction in range(1, num_of_directions + 1):
            print(f'direction {direction}')
            slices = create_slice_for_direction(ssdd_tensor, direction)
            l_score_direction = np.zeros_like(ssdd_tensor)
            for slice_idx in range(len(slices)):
                if direction == 1:
                    l_score_direction[slice_idx, :, :] = self.dp_grade_slice(slices[slice_idx].T, p1, p2).T

                elif direction == 3:
                    l_score_direction[:, slice_idx, :] = self.dp_grade_slice(slices[slice_idx].T, p1, p2).T

                elif direction == 5:
                    l_score_direction[slice_idx, :, :] = np.fliplr(
                        self.dp_grade_slice(np.fliplr(slices[slice_idx].T), p1, p2).T)

                elif direction == 7:
                    l_score_direction[:, slice_idx, :] = np.flipud(
                        self.dp_grade_slice(np.flipud(slices[slice_idx]).T, p1, p2).T)
                else:
                    slice_score = self.dp_grade_slice(slices[slice_idx], p1, p2).T
                    for d in range(ssdd_tensor.shape[2]):
                        if slice_idx < ssdd_tensor.shape[0]:
                            np.fill_diagonal(l_score_direction[slice_idx:, :, d], slice_score[:, d])
                        elif slice_idx == ssdd_tensor.shape[0]:
                            np.fill_diagonal(l_score_direction[:, :, d], slice_score[:, d])
                        else:
                            np.fill_diagonal(l_score_direction[:, slice_idx - ssdd_tensor.shape[0]:, d], slice_score[:, d])
            if direction == 4:
                l_score_direction = np.fliplr(l_score_direction)
            if direction == 6:
                l_score_direction = np.flipud(np.fliplr(l_score_direction))
            if direction == 8:
                l_score_direction = np.flipud(l_score_direction)
            direction_to_slice[direction] = self.naive_labeling(l_score_direction)

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
        for direction in range(1, num_of_directions + 1):
            slices = create_slice_for_direction(ssdd_tensor, direction)
            l_score_direction = np.zeros_like(ssdd_tensor)
            for slice_idx in range(len(slices)):
                if direction == 1:
                    l_score_direction[slice_idx, :, :] = self.dp_grade_slice(slices[slice_idx].T, p1, p2).T

                elif direction == 3:
                    l_score_direction[:, slice_idx, :] = self.dp_grade_slice(slices[slice_idx].T, p1, p2).T

                elif direction == 5:
                    l_score_direction[slice_idx, :, :] = np.fliplr(
                        self.dp_grade_slice(np.fliplr(slices[slice_idx].T), p1, p2).T)

                elif direction == 7:
                    l_score_direction[:, slice_idx, :] = np.flipud(
                        self.dp_grade_slice(np.flipud(slices[slice_idx]).T, p1, p2).T)
                else:
                    slice_score = self.dp_grade_slice(slices[slice_idx], p1, p2).T
                    for d in range(ssdd_tensor.shape[2]):
                        if slice_idx < ssdd_tensor.shape[0]:
                            np.fill_diagonal(l_score_direction[slice_idx:, :, d], slice_score[:, d])
                        elif slice_idx == ssdd_tensor.shape[0]:
                            np.fill_diagonal(l_score_direction[:, :, d], slice_score[:, d])
                        else:
                            np.fill_diagonal(l_score_direction[:, slice_idx - ssdd_tensor.shape[0]:, d],
                                             slice_score[:, d])
            if direction == 4:
                l_score_direction = np.fliplr(l_score_direction)
            if direction == 6:
                l_score_direction = np.flipud(np.fliplr(l_score_direction))
            if direction == 8:
                l_score_direction = np.flipud(l_score_direction)
            l += l_score_direction

        l /= num_of_directions
        return self.naive_labeling(l)
