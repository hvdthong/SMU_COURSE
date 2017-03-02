def kthlargest(arr1, arr2, k):
    # if len(arr1) == 0:
    #     return arr2[k - 1]
    # elif len(arr2) == 0:
    #     return arr1[k - 1]

    # mida1 = len(arr1) / 2
    # mida2 = len(arr2) / 2
    #
    # if mida1 + mida2 < k:
    #     if arr1[mida1] > arr2[mida2]:
    #         return kthlargest(arr1, arr2[mida2 + 1:], k - mida2 - 1)
    #     else:
    #         return kthlargest(arr1[mida1 + 1:], arr2, k - mida1 - 1)
    # else:
    #     if arr1[mida1] > arr2[mida2]:
    #         return kthlargest(arr1[:mida1], arr2, k)
    #     else:
    #         return kthlargest(arr1, arr2[:mida2], k)

    if len(arr1) > len(arr2):
        return kthlargest(arr2, arr1, k)
    if len(arr1) == 0 and len(arr2) > 0:
        return arr2[k - 1]
    if k == 1:
        return min(arr1[0], arr2[0])

    mida1 = min(len(arr1), k / 2)
    mida2 = min(len(arr2), k / 2)

    if arr1[mida1 - 1] < arr2[mida2 - 1]:
        return kthlargest(arr1[mida1:], arr2, k - mida1)
    else:
        return kthlargest(arr1, arr2[mida2:], k - mida2)


if __name__ == '__main__':
    A = [3, 4, 5, 6, 9, 10]
    B = [6, 7, 8, 11, 15, 17, 20, 25]
    k = 2
    print(kthlargest(A, B, k))
