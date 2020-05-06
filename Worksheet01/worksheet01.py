import Patchingimport ConvolveAndKerneldef task1():    # load the image    image_name = "dog.jpg"    image = Patching.Image.open(image_name)    image.show()    print("Image.size: ", image.size)    print("Type of the Image.size: ", type(image.size))    data = Patching.asarray(image)    print("Shape of the Image data array: ", data.shape)    data_edges = Patching.select_random_patch(image.size[0], image.size[1], square_side=256)    data_edge_array = [data_edges[0][0], data_edges[0][1], data_edges[1][0], data_edges[1][1]]    print("Edges of the random patch: ", data_edges)    # print(data_edge_array)    crop_image = image.crop(data_edge_array)    # print(crop_image)    crop_image.save("cropImage.jpg")    crop_image.show()    crop_image = crop_image.convert(mode='L')    # print(crop_image.size)    crop_image.save("grayscaleCropImage.jpg")    crop_image.show()    # print(image.getpixel((0, 0)))    # print(crop_image.getpixel((0, 0)))    Patching.patch_in_image(data_edges[0], image, crop_image)    (new_width, new_height) = (image.width // 2, image.height // 2)    image_resized = image.resize((new_width, new_height))    image_resized.save("resizedImage.jpg")    image_resized.show()    # Just playing with code    # image = cv2.imread("dog.jpg")    # clone = image.copy()    # cv2.namedWindow("Image")    # cv2.setMouseCallback("image", click_and_patching())    #    # refPt = []    #    # while True:    #     cv2.imshow("dog.jpg", image)    #     key = cv2.waitKey(0) & 0xFF    #    #     if key == ord("r"):    #         image = clone.copy()    #    #     elif key == ord(c):    #         break;    #    #     if len(refPt) == 2:    #         roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]    #         cv2.imshow("ROI", roi)    #         cv2.waitKey(0)    #    #     cv2.destroyAllWindows()def task2():    T = ConvolveAndKernel.gaussian_kernel(n=201, sigma=500, mu=0)    print("Shape of matrix T={} and the sum is {}".format(T.shape, T.sum()))    # Normalize to the range 0-1. Resize by 255 and convert array to uint8    T = ConvolveAndKernel.np.uint8(T / (T.max() / 255))    print('Min: %3.f, Max: %3.f' % (T.min(), T.max()))    print("Dimensions of kernel: ", T.ndim)    print("Shape of the kernel: ", T.shape)    pil_filter = ConvolveAndKernel.Image.fromarray(T)    pil_filter.save("filter.jpg")    pil_filter.show()    image_name = "dog.jpg"    image = ConvolveAndKernel.Image.open(image_name)    image = image.convert(mode='L')    pixels = ConvolveAndKernel.asarray(image)    pixels_future_image_tensor = pixels    print("Dimension of the image: ", pixels.ndim)    print("Shape of the pixels", pixels.shape)    # print(pixels)    # Convert pixels from integers to float    pixels = pixels.astype('float32')    # Normalize to the range 0-1    pixels /= 255    print('Min: %3.f, Max: %3.f' % (pixels.min(), pixels.max()))    # convolved_image = ConvolveAndKernel.convolving(image=pixels, kernel=T)    # convolved_image.show()    convolved_image = ConvolveAndKernel.convolve(pixels=pixels, conv_filter=T)    conv_image = ConvolveAndKernel.Image.fromarray(convolved_image)    conv_image.show()def task3():    # Load the image    image = ConvolveAndKernel.Image.open("dog.jpg")    # image.show()    # Convert image to numpy array    data = ConvolveAndKernel.np.array(image)    image = image.convert(mode='L')    pixels = ConvolveAndKernel.np.array(image)    # From numpy array to torch tensor    torch_data_tensor = ConvolveAndKernel.torch.from_numpy(data)    print("Tensor shape: ", torch_data_tensor.shape)    # Change axis of tensor    permuted_torch_data_tensor = torch_data_tensor.permute(1, 0, 2)    print(permuted_torch_data_tensor.size())    # From torch tensor to numpy array    numpy_array = permuted_torch_data_tensor.numpy()    print(numpy_array.shape)    reversed_image = ConvolveAndKernel.Image.fromarray(ConvolveAndKernel.np.uint8(numpy_array))    reversed_image = ConvolveAndKernel.Image.fromarray((numpy_array * 255).astype(ConvolveAndKernel.np.uint8))    reversed_image.save("reversedImage.jpg")    reversed_image.show()    # 3.2    x = ConvolveAndKernel.np.random.rand(5, 5, 1)    # print(x)    w = ConvolveAndKernel.np.random.rand(2, 2, 1)    # Creating nn.Conv2d object with in_channels = 1, out_channels = 1, kernel_size = 2, and bias = false    conv = ConvolveAndKernel.torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False)    print(conv.weight.shape)    # Converting x and w to PyTorch tensors and swapping axes    x_PyTorch_tensor = ConvolveAndKernel.torch.from_numpy(x)    permuted_tensor_x = x_PyTorch_tensor.permute(2, 0, 1)    w_PyTorch_tensor = ConvolveAndKernel.torch.from_numpy(w)    permuted_tensor_w = w_PyTorch_tensor.permute(2, 0, 1)    # Adding dimension to x and w    permuted_tensor_x.unsqueeze_(0)    permuted_tensor_w.unsqueeze_(0)    print(permuted_tensor_w.shape)    print(permuted_tensor_x.shape)    # Adding weight with w tensor    conv.weight = ConvolveAndKernel.torch.nn.Parameter(permuted_tensor_w)    # Convolving image x    output_array = conv(permuted_tensor_x)    print("Output shape: ", output_array.shape)    # Convolving image with Gaussian filter    T = ConvolveAndKernel.gaussian_kernel(n=201, sigma=500, mu=0)    print("Shape of matrix T={} and the sum is {}".format(T.shape, T.sum()))    # Normalize to the range 0-1. Resize by 255 and convert array to uint8    T = ConvolveAndKernel.np.uint8(T / (T.max() / 255))    image_PyTorch_tensor = ConvolveAndKernel.torch.from_numpy(ConvolveAndKernel.np.float32(pixels))    print(image_PyTorch_tensor.shape)    gaussian_PyTorch_tensor = ConvolveAndKernel.torch.from_numpy(ConvolveAndKernel.np.float32(T))    print(gaussian_PyTorch_tensor.shape)    if image_PyTorch_tensor.dtype == gaussian_PyTorch_tensor.dtype:        # permuted_tensor_image = image_PyTorch_tensor.permute(2, 0, 1)        # permuted_tensor_gaussian_filter = gaussian_PyTorch_tensor.permute(2, 0, 1)        # permuted_tensor_image.unsqueeze_(0)        # permuted_tensor_gaussian_filter.unsqueeze_(0)        image_PyTorch_tensor.unsqueeze_(0)        image_PyTorch_tensor.unsqueeze_(0)                # To check        gaussian_PyTorch_tensor.unsqueeze_(0)        gaussian_PyTorch_tensor.unsqueeze_(0)        print(image_PyTorch_tensor.shape)        print(gaussian_PyTorch_tensor.shape)        # conv.weight = torch.nn.Parameter(permuted_tensor_gaussian_filter)        conv.weight = ConvolveAndKernel.torch.nn.Parameter(gaussian_PyTorch_tensor)        # output_image_array = conv(permuted_tensor_image)        # print("Output image.shape: ", output_image_array.shape)        output_image_array = conv(image_PyTorch_tensor)        print("Output image.shape: ", output_image_array.shape)def main():    task1()    task2()    task3()if __name__ == '__main__':    main()