import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    padding_height = int((Hk-1)/2)
    padding_width = int((Wk-1)/2)
    ### YOUR CODE HERE
    for i in range(Hi):
        for j in range(Wi):
            for ki in range(Hk):
                for kj in range(Wk):
                    a = image[i-padding_height+ki, j-padding_width+kj] if i-padding_height+ki >= 0 and j-padding_width+kj >= 0 and i-padding_height+ki < Hi and j-padding_width+kj < Wi else 0
                    out[i, j] += a * kernel[Hk-ki-1, Wk-kj-1]
    # for i in range(0,Hi):
    #     for j in range(0,Wi):
    #         sum = 0
    #         for ki in range(int(-Hk/2),int(Hk/2)+1):
    #             for kj in range(int(-Wk/2),int(Wk/2)+1):
    #                 if i + ki < 0 or j + kj < 0 or i + ki >= Hi or j + kj >= Wi:
    #                     sum += 0
    #                 else:
    #                     sum += image[i + ki][j + kj] * kernel[ki + int(Hk/2)][kj + int(Wk/2)]
    #         out[i][j] = sum
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width),), 'constant', constant_values=0)
    # out = np.zeros((H+2*pad_height, W+2*pad_width))
    # for i in range(H+2*pad_height):
    #     for j in range(W+2*pad_width):
    #         if i in range(pad_height, pad_height+H) and j in range(pad_width, pad_width+W):
    #             out[i, j] = image[i-pad_height, j-pad_width]
            
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    image_pad = zero_pad(image, int((Hk-1)/2), int((Wk-1)/2))
    kernel_flip = np.flip(np.flip(kernel, 0), 1)
    kernel_flat = np.flip(np.flip(kernel, 0), 1).flatten()    
    # fast
    for i in range(Hi):
        for j in range(Wi):
            out[i][j] = np.sum(np.multiply(image_pad[i:i+Hk, j:j+Wk], kernel_flip))
            
    # faster 直接使用向量乘代替之前的矩阵乘再求和
    # for i in range(Hi):
    #     for j in range(Wi):
    #         out[i][j] = np.dot(image_pad[i:i+Hk, j:j+Wk].flatten(), kernel_flat)

    # fastest 先将图片展开成列向量矩阵，然后直接矩阵相乘得到最终结果（矩阵）
    # image_pad_col = np.zeros((Hi, Wi, Hk*Wk))
    # for i in range(Hi):
    #     for j in range(Wi):
    #         image_pad_col[i][j] = image_pad[i:i+Hk, j:j+Wk].flatten()
    # out = np.dot(image_pad_col, kernel_flat)
    
    ### END YOUR CODE

    return out
    # return np.convolve(image, kernel, "same")

def cross_correlation(f, g):
    """ Cross-correlation of image f and template g,
    cross correlation is equivalent to a convolution with out flip

    Hint: you can flip `g` at x-axis and y-axis first, 
    and use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    new_g = np.flip(np.flip(g, 0), 1)
    Hg, Wg = g.shape
    # 给定的template长宽不是奇数，调用conv_fast时padding会出问题
    # 所以需要将template填充为奇数长宽
    if Hg % 2 == 0:
        new_g = np.pad(new_g, ((1, 0), (0, 0)), "constant", constant_values=0)
    if Wg % 2 == 0:
        new_g = np.pad(new_g, ((0, 0), (1, 0)), "constant", constant_values=0)
    out = conv_fast(f, new_g)
    ### END YOUR CODE
    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of image f and template g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    # 计算template平均值并给每个元素减去平均值
    mean = np.mean(g)
    new_g = g - mean
    Hg, Wg = g.shape
    # 填充长宽
    if Hg % 2 == 0:
        new_g = np.pad(new_g, ((1, 0), (0, 0)), "constant", constant_values=0)
    if Wg % 2 == 0:
        new_g = np.pad(new_g, ((0, 0), (1, 0)), "constant", constant_values=0)
        
    out = conv_fast(f, new_g)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of image f and template g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    (you shall not use `conv_fast` above, for you need to normalize each subimage of f)

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    ### YOUR CODE HERE
    out = np.zeros((Hf, Wf))
    g_mean = np.mean(g)
    g_std = np.std(g)
    if Hg % 2 == 0:
        new_g = np.pad(g, ((1, 0), (0, 0)), "constant", constant_values=0)
    if Wg % 2 == 0:
        new_g = np.pad(new_g, ((0, 0), (1, 0)), "constant", constant_values=0)
    Hg, Wg = new_g.shape
    f_pad = zero_pad(f, int((Hg-1)/2), int((Wg-1)/2))
    for i in range(Hf):
        for j in range(Wf):
            f_area = f_pad[i:i+Hg, j:j+Wg]
            f_mean = np.mean(f_area)
            f_std = np.std(f_area)
            out[i, j] = np.dot(((f_area-f_mean)/f_std).flatten(), ((new_g-g_mean)/g_std).flatten())
            # 遍历的方式太慢了
            # for ki in range(Hg):
            #     for kj in range(Wg):
            #         out[i, j] += ((f_area[ki, kj]-f_mean) / f_std) * ((new_g[ki, kj]-g_mean) / g_std) 
    ### END YOUR CODE

    return out
