from __future__ import annotations

import datetime
import importlib.resources as pkg_resources
import json
from dataclasses import dataclass

import exiv2
import numpy as np
import OpenImageIO as oiio
import scipy.interpolate

################################################################################
# Image metadata
################################################################################


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    exif: exiv2.ExifData
    iptc: exiv2.IptcData
    xmp: exiv2.XmpData


def read_image_metadata(filename: str) -> ImageMetadata | None:
    """Read content metadata (EXIF, IPTC, XMP) from an image file.

    Uses the Exiv2 library to read metadata from any format including RAW files.

    Parameters
    ----------
    filename : str
        Path to the image file.

    Returns
    -------
    ImageMetadata or None
        The image metadata, or ``None`` if the file cannot be opened.
    """
    try:
        image = exiv2.ImageFactory.open(filename)
        image.readMetadata()
    except Exception:
        return None

    return ImageMetadata(
        exif=image.exifData(),
        iptc=image.iptcData(),
        xmp=image.xmpData(),
    )


def write_image_metadata(
    filename: str,
    source_metadata: ImageMetadata | None = None,
    *,
    saving_color_space: str | None = None,
    saving_cctf_encoding: bool = True,
) -> None:
    """Write metadata to an image file after pixel data has been saved.

    Copies any source EXIF, IPTC and XMP tags, then sets overridden tags
    (Orientation, DateTime, Software, pixel dimensions). When
    ``saving_color_space`` is given, also tags the file with the EXIF
    ColorSpace / Interoperability fields that match the saved color space and
    records the human-readable profile name in ``Xmp.photoshop.ICCProfile``.

    Parameters
    ----------
    filename : str
        Path to the output image file (must already exist on disk).
    source_metadata : ImageMetadata, optional
        Metadata returned by ``read_image_metadata`` to copy from the original
        file. Pass ``None`` when there is no source file.
    saving_color_space : str, optional
        Human-readable name of the color space the pixels were encoded in
        (e.g. ``"sRGB"``, ``"Adobe RGB (1998)"``, ``"Display P3"``).
    saving_cctf_encoding : bool, default True
        Whether the saved pixels carry the color space's encoding transfer
        function. ``False`` (linear data) is appended to the recorded profile
        name so downstream tools can flag it.
    """
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "exr":
        return

    image_input = oiio.ImageInput.open(filename)
    if image_input is None:
        raise RuntimeError(f"Could not open image file with OpenImageIO: {filename}")

    try:
        spec = image_input.spec()
    finally:
        image_input.close()
    destination = exiv2.ImageFactory.open(filename)
    destination.readMetadata()

    if source_metadata is not None:
        destination.setExifData(source_metadata.exif)
        destination.setIptcData(source_metadata.iptc)
        destination.setXmpData(source_metadata.xmp)

    destination_exif = destination.exifData()

    destination_exif["Exif.Image.Orientation"] = 1
    destination_exif["Exif.Image.DateTime"] = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    destination_exif["Exif.Image.Software"] = "spektrafilm"
    destination_exif["Exif.Photo.PixelXDimension"] = spec.width
    destination_exif["Exif.Photo.PixelYDimension"] = spec.height

    if saving_color_space is not None:
        _set_color_space_tags(
            destination_exif,
            destination.xmpData(),
            saving_color_space,
            saving_cctf_encoding,
        )

    destination.writeMetadata()


# EXIF Photo.ColorSpace values per EXIF 2.32 spec.
_EXIF_COLORSPACE_SRGB = 1
_EXIF_COLORSPACE_UNCALIBRATED = 65535


def _set_color_space_tags(
    exif_data: "exiv2.ExifData",
    xmp_data: "exiv2.XmpData",
    saving_color_space: str,
    saving_cctf_encoding: bool,
) -> None:
    if saving_color_space == "sRGB" and saving_cctf_encoding:
        exif_data["Exif.Photo.ColorSpace"] = _EXIF_COLORSPACE_SRGB
        exif_data["Exif.Iop.InteroperabilityIndex"] = "R98"
    elif saving_color_space == "Adobe RGB (1998)" and saving_cctf_encoding:
        exif_data["Exif.Photo.ColorSpace"] = _EXIF_COLORSPACE_UNCALIBRATED
        exif_data["Exif.Iop.InteroperabilityIndex"] = "R03"
    else:
        exif_data["Exif.Photo.ColorSpace"] = _EXIF_COLORSPACE_UNCALIBRATED

    profile_name = saving_color_space if saving_cctf_encoding else f"{saving_color_space} (linear)"
    xmp_data["Xmp.photoshop.ICCProfile"] = profile_name


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
        img_uint16 = np.clip(image_data, 0, 1) * 255.0
        img_uint16 = img_uint16.astype(np.uint8)
        spec = oiio.ImageSpec(width, height, nchannels, oiio.TypeDesc("uint8"))
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
# Neutral filter values
################################################################################

NEUTRAL_PRINT_FILTERS_FILENAME = 'neutral_print_filters.json'


def save_neutral_print_filters(neutral_print_filters):
    package = pkg_resources.files('spektrafilm.data.filters')
    resource = package / NEUTRAL_PRINT_FILTERS_FILENAME
    with resource.open("w") as file:
        json.dump(neutral_print_filters, file, indent=4)


def read_neutral_print_filters():
    package = pkg_resources.files('spektrafilm.data.filters')
    resource = package / NEUTRAL_PRINT_FILTERS_FILENAME
    with resource.open("r") as file:
        return json.load(file)

################################################################################
# Profiles
################################################################################

def load_dichroic_filters(wavelengths, brand='thorlabs'):
    channels = ['c','m','y']
    filters = np.zeros((np.size(wavelengths), 3))
    for i, channel in enumerate(channels):
        package = pkg_resources.files('spektrafilm.data.filters.dichroics')
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
    package = pkg_resources.files('spektrafilm.data.filters.'+filter_type)
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
