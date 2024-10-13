# import numpy as np
# import scipy.ndimage
# from scipy import signal
#
# # Create a 3D input volume (e.g., an image stack)
# input_tensor = np.array([
#                          [[1, 2, 3, 4, 0],
#                           [4, 2, 3, 4, 4],
#                           [7, 2, 3, 4, 6],
#                           [6, 2, 3, 4, 9],
#                           [0, 2, 3, 4, 1]],
#                          [[1, 2, 3, 4, 3],
#                           [2, 2, 3, 4, 8],
#                           [9, 2, 3, 4, 40],
#                           [8, 2, 3, 4, 10],
#                           [0, 2, 3, 4, 10]],
# [[1, 2, 3, 4, 3],
#                           [2, 2, 3, 4, 8],
#                           [9, 2, 3, 4, 40],
#                           [8, 2, 3, 4, 10],
#                           [0, 2, 3, 4, 10]],
# [[1, 2, 3, 4, 3],
#                           [2, 2, 3, 4, 8],
#                           [9, 2, 3, 4, 40],
#                           [8, 2, 3, 4, 10],
#                           [1, 3, 3, 4, 10]]
#
#                          ])
#
# # Create a 3D kernel (filter)
# kernel = np.array([
#     [[1,2,3],
#      [1,1,1],
#      [2,2,2]],
#     [[1, 2, 3],
#      [1, 1, 1],
#      [1, 2, 2]],
#     [[1, 2, 3],
#     [1, 1, 1],
#     [2, 1, 2]],
# [[1, 2, 3],
#     [1, 1, 1],
#     [0, 0, 0]]
# ])
#
# c= input_tensor.shape[0]
# print(c)
#
# # Perform 3D convolution using scipy's convolve function with 'SAME' padding
# output_volume = signal.correlate(input_tensor, np.flip(kernel, axis=(1,2) if 1>2 else axis = (1)), mode='same')
#
# o1 = np.sum(output_volume, axis = 0)
# print(output_volume)
#
#
# output_tensor = np.zeros((1,*input_tensor.shape[1:])).astype(int)
# for n_k in range(1):
#     for c in range(4):
#
#         output_tensor += signal.convolve(input_tensor[c],kernel[c], mode='same')
#     #output_tensor[b, n_k] = np.sum(signal.convolve(input_tensor[b], self.weights[n_k], 'same'), axis=0)
#     #output_tensor[b, n_k] += self.bias[n_k]
# #output_tensor = np.sum(output_tensor, axis=0)
# print(output_tensor)
#
# # # import numpy as np
# # # from scipy.signal import convolve
# # #
# # # # Create a 3D input volume (e.g., a stack of 3D images)
# # # input_volume = np.random.rand(5, 5, 5)
# # #
# # # # Define the number of kernels
# # # num_kernels = 3
# # #
# # # # Create 3D kernels (filters)
# # # kernels = [np.random.rand(3, 3, 3) for _ in range(num_kernels)]
# # #
# # # # Initialize an empty output volume
# # # output_volume = np.zeros((5, 5, num_kernels))
# # #
# # # # Perform 3D convolution for each kernel
# # # for i in range(num_kernels):
# # #     output_volume[:, :, i] = convolve(input_volume, kernels[i], mode='same', method='direct')
# # #
# # # # Print the input volume, kernels, and output volume
# # # print("Input Volume:")
# # # print(input_volume)
# # #
# # # print("\nKernels:")
# # # for i in range(num_kernels):
# # #     print(f"Kernel {i + 1}:")
# # #     print(kernels[i])
# # #
# # # print("\nOutput Volume:")
# # # print(output_volume)
# # # print(output_volume.shape)
# # #
# # # import numpy as np
# # #
# # # # Example data
# # # nk = 2
# # # h = 3
# # # m = 3
# # # n = 3
# # #
# # # # Create a random weight tensor
# # # weight_tensor = np.random.rand(nk, h, m, n)
# # #
# # # # Reshape and transpose the weight tensor
# # # reshaped_weight_tensor = weight_tensor.transpose((1, 0, 2, 3))
# # #
# # # # Check the shapes
# # # print("Original shape:", weight_tensor.shape)
# # # print(weight_tensor)
# # # print("Reshaped shape:", reshaped_weight_tensor.shape)
# # # print(reshaped_weight_tensor)
#
#
# #
# # import numpy as np
# #
# # # Create a sample 3D kernel
# # kernel_3d = np.random.rand(3, 3, 3)
# #
# # # # Flip the 3D kernel along each axis
# # # flipped_kernel_3d = np.flip(kernel_3d, axis=0)
# # # flipped_kernel_3d = np.flip(flipped_kernel_3d, axis=1)
# # flipped_kernel_3d = np.flip(kernel_3d, axis=(1,2))
# #
# # print("Original 3D Kernel:")
# # print(kernel_3d)
# #
# # print("\nFlipped 3D Kernel:")
# # print(flipped_kernel_3d)
