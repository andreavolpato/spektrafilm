# `agx-emulsion`: simulation of color film photography from scratch
> [!IMPORTANT]
> At this stage, this project is very experimental and a work in progress. Things might change fast, and it is really just a playground for exploring the simulation model.

## Introduction

In this project we will experiment with the simulation of color film photography processes. The simulation emulates a negative emulsion starting from published data for film stocks. An example of curves for Kodak Portra 400 (data-sheet e4050, 2016) is in the following figure (note that CMY diffuse densities are generic ones, because they are usually not published).

![Data extracted from the datasheet of Kodak Portra 400](img/readme/example_data_kodak_portra_400.png)

An example of data from print paper Kodak Portra Endura (data-sheet e4021, 2009) is in the next figure.

![Data extracted from the datasheet of Kodak Ektacolor Edge](img/readme/example_data_kodak_portra_endura.png)

The left panel shows the spectral log sensitives of each color layer. The central panel shows the log-exposure-density characteristic curves for each layer when a neutral grey gradient illuminated with a reference light is used to expose the medium. The panel on the right shows the absorption spectra of the dyes formed on the medium upon chemical development. 'Min' and 'Mid' are the absorption for the unexposed processed medium and a neutral grey "middle" exposure medium, respectively. 

Starting from linear RGB data from a RAW file of a camera, the simulation recreates the spectral data, then we project the virtual light transmitted from the negative to print paper, and it uses a simplified color enlarger with dichroic filters for balancing the colors of the print. Finally we scan the virtual print using the light reflected from the print.

The pipeline is sketched in this figure adapted from [1]:
![The color photography process.](img/readme/pipeline_color_digital_management.png)
where the light from a scene (raw file from your camera) is exposed on a virtual negative with specific spectral sensitivities, then a chemical process create the dye densities (using density curves and more complex interactions modeling the couplers). The virtual negative is projected with a specific illuminant on paper that is developed again (simple density curves, no coupler in this case, print paper is already design to reduce cross-talk of channel since doesn't have to sample a scene but just the dyes on the negative).

The pipeline allow to add many characteristic in a physically sound way. For example:
- halation
- film grain generated on the negative (using a stochastic model)
- pre-flashing of the print to retain highlights

From my experience playing around with film simulation, data-sheet curves are really not enough to reproduce a decent film look. The key is to understand that in the film emulsion there are couplers (chemicals that are produced in the development along side the actual CMY dyes) that are very important to achieve the desired saturation. Mainly there are:
- masking couplers, that gives the typical color orange to the unexposed developed film. This couplers are consumed locally if density is formed and are used to reduce the effect of cross-talk of the absorption of the layers, thus increasing saturation.
The presence of masking couplers is simulated with negative absorption contribution in the isolated dye absorption spectra. See for example the data of Portra 400 updated to include the masking couplers and with unmixed print density characteristic curves:
![Portra 400 data modified for masking couplers and unmixing of densities.](img/readme/example_data_kodak_portra_400_couplers.png)

- direct inhibitor couplers, that are released locally when density is formed and inhibit the formation of density in nearby layers or in the same layer. This increases saturation and contrast. Also if we let the coupler diffuse in space they can increase local contrast and perceived sharpness. The simulation of coupler inhibitors is inspired by Michaelis–Menten inhiition kinetics (https://en.wikipedia.org/wiki/Michaelis%E2%80%93Menten_kinetics).

A more detailed description of colour couplers can be found in Chapter 15 of Hunt's book [2].


## Installation
Create and activate a python environment, I used `conda`: 
```
conda create -n agx-emulsion python=3.11.4
conda activate agx-emulsion
```
Install all the requirements in `requirements.txt` and the package `agx-emulsion` by going to the repository folder and running:
```
pip install -r requirements.txt
pip install -e .
```
Launch the GUI:
```
python agx_emulsion/gui/main.py
```
A `napari` window should appear. In `napari` I usually set the theme to `Light` because I find it easier to judge exposure with a white background. Go to `File >> Preferences >> Appearance >> Theme >> Light`. Also, `napari` is not color-managed and expects the video output to be treated as raw sRGB values. The way I am working is to set the color profile of screen and operating system to an sRGB profile.

## GUI
Launch the GUI by running the file `agx_emulsion/gui/main.py`. The gui is based on `napari` (Qt) and `magicgui` that are not color-managed, so probably a poor choice but a temporary quick solution for fast prototyping.
You should load an image that you converted from a raw file and kept linear, I usually save in PNG 16-bit sRGB to preserve the dynamic range. It is important to export in sRGB because the conversion from RGB to spectral data at the very beginning of the pipeline (using `colour.recovery.RGB_to_sd_Mallett2019`) uses the method in [3] for simplicity and computational efficiency. More advanced methods are required to recover spectral data from large color gamut RGB color spaces. Play with the parameters and press `Run` to run the simulation. In order to correctly load a 16-bit PNG file there is a small widget called `filepicker` that will import correctly the image as a new layer.

> [!TIP]
> Hover with the mouse on the widgets and controls to visualize an help tooltip.

![Example of GUI interface with color test image.](img/readme/gui_screenshot.png)

Please bear in mind that this is a highly experimental project and many controls are left exposed in the GUI with poor or no documentation. Make use of the help tooltips by overing the control boxes or explore the code.
Play with `exposure_compensation_ev` to change the exposure of the negative. You can visualize a virtual scan of the negative by pressing `compute_negative` and `Run`.
For fine tuning of halation play with `scattering size`, `scattering strength`, `halation size`, `halation strength`. There are three controls for each that define the effect on the three color channels (RGB). `scattering size` and `halation size` represent the value of sigma for Gaussian blurring. `scattering strength` and `halation strength` refers to the percentage of scattered of halation light.
`y filter shift` and `m filter shift` are the control for the virtual yellow and magenta filters of the color enlarger. They are the number of steps for the shift from a neutral position, i.e. starting settings that make an 18% gray target photographed with correct reference illuminant fully neutral in the final print. The enlarger has 170 steps.  
There are controls to apply lens blur in several stages of the pipeline. For example in the camera lens, in the color enlarger lens or the scanner. There is also a control for blurring the density as an effect of diffusion during development `grain > blur`. The scanner has also sharpness controls via a simple unsharp mask filter.

For example by magnifying the film, like a 0.7x0.7 mm sized crop, reveals the isolated dye clouds.

![Example of GUI interface with color test image.](img/readme/gui_grain_magnified.png)

This is one of the most appealing aspect for me, when I think of printing posters of high resolution simulated images retaining all this low level grain detail not available in the original picture.

## Preparing input images with darktable
The simulation expects linear scene-referred sRGB files as input.
I usually open RAW the files of digital cameras with darktable (https://www.darktable.org/) and deactivate the non linear mappings done by `filmic` or `sigmoid` modules and adjust the exposure to fit all the information avoiding clipping. Then I export the file as a 16-bit PNG, e.g. with the following export settings:

![Darktable export settings.](img/readme/darktable_export_settings.png)

## Example usage of the GUI

https://github.com/user-attachments/assets/9e9ac598-20e5-412a-975c-5ecd57325b74

Thank you Adam Severeid from discuss.pixls.us forum (https://discuss.pixls.us/t/have-a-seat-weve-been-waiting-for-you/44814) for providing the RAW in a Play Raw post that I used here.

## Things to consider
- The simulation is quite slow even for normal resolutions of 2-3K (a few megapixels), it takes several seconds on my laptop. I haven't tested on very high resolution images, mainly because they crash my computer because they require too much memory. I usually adjust most of the values with scaled down preview images, that is by default computed in the GUI, then when a final image is need I activate the `compute full image` checkbox that bypasses the image scaling.
- Fujifilm profiles are for now less trustworthy than Kodak ones because the data taken from a single data-sheet are not self-consistent, i.e. they do not work very well after the unmixing step. To make them work ok-ish, a small adjustment is performed to obtain neutral grayscales, this might change in the future.

## References
[1] Giorgianni, Madden, Digital Color Management, 2nd edition, 2008 Wiley  
[2] Hung, The Reproduction of Color, 6th edition, 2004 Wiley  
[3] Mallett, Yuksel, Spectral Primary Decomposition for Rendering with sRGB Reflectance, Eurographics Symposium on Rendering - DL-only and Industry Track, 2019, doi:10.2312/SR.20191216  

Sample images from signatureedits.com/free-raw-photos.
