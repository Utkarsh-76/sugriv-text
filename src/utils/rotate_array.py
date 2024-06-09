def rotate_array(arr, k):
    """
    Rotate an array to the right by k positions.

    Args:
        arr (list): Input array.
        k (int): Number of positions to rotate.

    Returns:
        list: Rotated array.
    """
    n = len(arr)
    # Handle case where k is larger than the length of the array
    k = k % n
    # Rotate the array using slicing
    rotated_arr = arr[-k:] + arr[:-k]
    return rotated_arr