def rank_binary_search(arr, l, r, x):
    if x < arr[0]:
        return 0

    if x > arr[len(arr) - 1]:
        return len(arr)

    if r >= l:
        mid = l + (r - l) / 2

        if arr[mid] == x:
            return mid + 1

        elif (arr[mid - 1] < x) and (arr[mid] > x):
            return mid

        elif arr[mid] > x:
            return rank_binary_search(arr, l, mid - 1, x)
        else:
            return rank_binary_search(arr, mid + 1, r, x)
    return 1


def binary_search(value, items, low=0, high=None):
    """
    Binary search function.
    Assumes 'items' is a sorted list.
    The search range is [low, high)
    """

    high = len(items) if high is None else high
    pos = low + (high - low) / len(items)

    if pos == len(items):
        return False
    elif items[pos] == value:
        return pos
    elif high == low:
        return False
    elif items[pos] < value:
        return binary_search(value, items, pos + 1, high)
    else:
        assert items[pos] > value
        return binary_search(value, items, low, pos)


def rank_serach(A, value):
    index = 0
    for i in range(0, len(A)):
        flag = binary_search(value, A[i], 0, len(A[i]))
        if flag is False:
            index += len(A[i])
        else:
            return index + flag
    if index is False:
        print "No rank of element ", value


if __name__ == '__main__':
    A = [1, 5, 8, 10, 15, 20]
    b = 20
    # print (binary_search(A, 15))
    # print binary_search(8, A, 0, len(A))
    print rank_binary_search(A, 0, len(A), b)

    # A1 = [1, 5, 8, 10]
    # A2 = [2, 6, 9, 15]
    # A = [A1, A2]
    # b = 15
    # print rank_serach(A, b)
