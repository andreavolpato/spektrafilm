import numpy as np
import scipy.interpolate
import json
import OpenImageIO as oiio
import importlib.resources as pkg_resources


################################################################################
# 16-bit PNG I/O
################################################################################

def load_image_oiio(filename):
    # Open the image file
    in_img = oiio.ImageInput.open(filename)
    if not in_img:
        raise IOError("Could not open image file: " + filename)
    
    try:
        spec = in_img.spec()
        
        # Determine the native pixel format:
        # Use "uint16" for PNG and "half" for EXR if applicable.
        if spec.format == oiio.TypeDesc("uint8"): # for compatibility
            read_type = oiio.TypeDesc("uint8")
        elif spec.format == oiio.TypeDesc("uint16"):
            read_type = oiio.TypeDesc("uint16")
        elif spec.format == oiio.TypeDesc("half"):
            read_type = oiio.TypeDesc("half")
        elif spec.format == oiio.TypeDesc("float"):
            read_type = oiio.TypeDesc("float")
        else:
            # Fallback: use "uint16" by default. You might choose "float" if desired.
            read_type = oiio.TypeDesc("uint16")
        
        # Read the image data using the chosen type
        pixels = in_img.read_image(read_type)
        if pixels is None:
            raise Exception("Failed to read image data from " + filename)
        
        # Convert the raw data to a NumPy array and reshape it
        np_pixels = np.array(pixels)
        np_pixels = np_pixels.reshape(spec.height, spec.width, spec.nchannels)
        
        if spec.format == oiio.TypeDesc("uint16"):
            np_pixels = np.double(np_pixels)/(2**16-1)
        if spec.format == oiio.TypeDesc("uint8"):
            np_pixels = np.double(np_pixels)/(2**8-1)
        
        return np_pixels
    finally:
        in_img.close()

def save_image_oiio(filename, image_data, bit_depth=32):
    """
    Save a floating-point (double) image with 3 channels as a 16-bit image file.
    For PNG files, the image data is scaled to the [0,65535] range and saved as uint16.
    For JPEG files, the image data is scaled to the [0,255] range and saved as uint8.
    For EXR files, the image data is converted to 16-bit half floats.
    
    Parameters:
            filename (str): The output file name (e.g., "saved_image.png", "saved_image.jpg", or "saved_image.exr")
      image_data (np.ndarray): The input image data as a NumPy array with shape (height, width, 3).
    """
    # Extract image dimensions and number of channels
    height, width, nchannels = image_data.shape

    # Determine file type based on extension
    ext = filename.split('.')[-1].lower()
    
    # Create an ImageSpec with the proper data type
    if ext == "png":
        # Assume image_data is in [0, 1]: scale to 16-bit unsigned integers.
        img_uint16 = np.clip(image_data, 0, 1) * 65535.0
        img_uint16 = img_uint16.astype(np.uint16)
        spec = oiio.ImageSpec(width, height, nchannels, oiio.TypeDesc("uint16"))
        data_to_write = img_uint16
    elif ext in {"jpg", "jpeg"}:
        img_uint8 = np.clip(image_data, 0, 1) * 255.0
        img_uint8 = img_uint8.astype(np.uint8)
        spec = oiio.ImageSpec(width, height, nchannels, oiio.TypeDesc("uint8"))
        data_to_write = img_uint8
    elif ext=="exr" and bit_depth==16:
        # Convert the image data to 16-bit half precision.
        # Note: numpy's float16 is used here; OpenImageIO accepts "half" for 16-bit floats.
        img_half = image_data.astype(np.float16)
        spec = oiio.ImageSpec(width, height, nchannels, oiio.TypeDesc("half"))
        data_to_write = img_half
    elif ext=='exr' and bit_depth==32:
        # Convert the image data to 16-bit half precision.
        # Note: numpy's float16 is used here; OpenImageIO accepts "half" for 16-bit floats.
        img_float = image_data.astype(np.float32)
        spec = oiio.ImageSpec(width, height, nchannels, oiio.TypeDesc("float"))
        data_to_write = img_float
    else:
        raise ValueError("Unsupported file extension: " + ext)
    
    # Create an ImageOutput for writing the file
    out = oiio.ImageOutput.create(filename)
    if not out:
        raise IOError("Could not create output image: " + filename)
    
    try:
        out.open(filename, spec)
        # Write the image data; write_image accepts the NumPy array directly.
        out.write_image(data_to_write)
    finally:
        out.close()

################################################################################
# YMC filter values
################################################################################

def save_ymc_filter_values(ymc_filters):
    # to be launched only in the package not accessible by the user
    package = pkg_resources.files('spectral_film_lab.data.profiles')
    filename = 'enlarger_neutral_ymc_filters.json'
    resource = package / filename
    with resource.open("w") as file:
        json.dump(ymc_filters, file, indent=4)

def read_neutral_ymc_filter_values():
    filename = 'enlarger_neutral_ymc_filters.json'
    package_name = 'spectral_film_lab.data.profiles'
    package = pkg_resources.files(package_name)
    resource = package / filename
    with resource.open("r") as file:
        ymc_filters = json.load(file)
    return ymc_filters

################################################################################
# Profiles
################################################################################

def load_dichroic_filters(wavelengths, brand='thorlabs'):
    channels = ['y','m','c']
    filters = np.zeros((np.size(wavelengths), 3))
    for i, channel in enumerate(channels):
        package = pkg_resources.files('spectral_film_lab.data.filters.dichroics')
        filename = brand+'/filter_'+channel+'.csv'
        resource = package / filename
        with resource.open("r") as file:
            data = np.loadtxt(file, delimiter=',')
            unique_index = np.unique(data[:,0], return_index=True)[1]
            data = data[unique_index,:]
            # filters[:,i] = scipy.interpolate.CubicSpline(data[:,0], data[:,1]/100)(wavelengths)
            filters[:,i] = scipy.interpolate.Akima1DInterpolator(data[:,0], data[:,1]/100)(wavelengths)
    return filters

def load_filter(wavelengths, name='KG3', brand='schott', filter_type='heat_absorbing', percent_transmittance=False):
    transmittance = np.zeros_like(wavelengths)
    package = pkg_resources.files('spectral_film_lab.data.filters.'+filter_type)
    filename = brand+'/'+name+'.csv'
    resource = package / filename
    if percent_transmittance: scale = 100
    else: scale = 1
    with resource.open("r") as file:
        data = np.loadtxt(file, delimiter=',')
        unique_index = np.unique(data[:,0], return_index=True)[1]
        data = data[unique_index,:]
        # transmittance = scipy.interpolate.CubicSpline(data[:,0], data[:,1]/scale)(wavelengths)
        transmittance = scipy.interpolate.Akima1DInterpolator(data[:,0], data[:,1]/scale)(wavelengths)
    return transmittance
