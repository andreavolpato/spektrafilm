from dataclasses import dataclass

from spektrafilm_gui.params_manifest import (
    CAMERA_DIFFUSION_MANIFEST,
    CAMERA_MANIFEST,
    CHEMISTRY_MANIFEST,
    CONVERT_MANIFEST,
    DIR_COUPLERS_MANIFEST,
    ENLARGER_DIFFUSION_MANIFEST,
    FILM_BASE_MANIFEST,
    FILM_CHEMISTRY_MANIFEST,
    GLARE_MANIFEST,
    GRAIN_MANIFEST,
    HALATION_MANIFEST,
    INPUT_GAMUT_COMPRESS_MANIFEST,
    OUTPUT_GAMUT_COMPRESS_MANIFEST,
    PREFLASHING_MANIFEST,
    PRINT_BASE_MANIFEST,
    SCANNER_MANIFEST,
)
from spektrafilm_gui.widget_primitives import CollapsibleSection, platform_default_font
from spektrafilm_gui.widget_sections import (
    DisplaySection,
    EnlargerSection,
    FilePickerSection,
    GuiConfigSection,
    InputImageSection,
    LoadRawSection,
    OutputSection,
    ParamsGroupSection,
    PreviewCropSection,
    SimulationSection,
    SpecialSection,
    SpectralUpsamplingSection,
)

__all__ = (
    "CollapsibleSection",
    "WidgetBundle",
    "create_widget_bundle",
    "platform_default_font",
)


@dataclass(slots=True)
class WidgetBundle:
    filepicker: FilePickerSection
    gui_config: GuiConfigSection
    display: DisplaySection
    input_image: InputImageSection
    load_raw: LoadRawSection
    grain: ParamsGroupSection
    preflashing: ParamsGroupSection
    enlarger_diffusion: ParamsGroupSection
    camera_diffusion: ParamsGroupSection
    halation: ParamsGroupSection
    couplers: ParamsGroupSection
    chemistry: ParamsGroupSection
    print_base: ParamsGroupSection
    film_chemistry: ParamsGroupSection
    film_base: ParamsGroupSection
    convert: ParamsGroupSection
    glare: ParamsGroupSection
    input_gamut_compress: ParamsGroupSection
    output_gamut_compress: ParamsGroupSection
    special: SpecialSection
    simulation: SimulationSection
    preview_crop: PreviewCropSection
    camera: ParamsGroupSection
    enlarger: EnlargerSection
    scanner: ParamsGroupSection
    spectral_upsampling: SpectralUpsamplingSection
    output: OutputSection


def create_widget_bundle() -> WidgetBundle:
    filepicker = FilePickerSection()
    input_image = InputImageSection(filepicker)
    simulation = SimulationSection()
    special = SpecialSection(simulation)
    glare = ParamsGroupSection(GLARE_MANIFEST)
    scanner = ParamsGroupSection(SCANNER_MANIFEST)
    camera = ParamsGroupSection(CAMERA_MANIFEST)
    simulation.bind_scan_for_print_sections(glare=glare, scanner=scanner)

    return WidgetBundle(
        filepicker=filepicker,
        gui_config=GuiConfigSection(),
        display=DisplaySection(),
        input_image=input_image,
        load_raw=LoadRawSection(),
        grain=ParamsGroupSection(GRAIN_MANIFEST),
        preflashing=ParamsGroupSection(PREFLASHING_MANIFEST),
        enlarger_diffusion=ParamsGroupSection(ENLARGER_DIFFUSION_MANIFEST),
        camera_diffusion=ParamsGroupSection(CAMERA_DIFFUSION_MANIFEST),
        halation=ParamsGroupSection(HALATION_MANIFEST),
        couplers=ParamsGroupSection(DIR_COUPLERS_MANIFEST),
        chemistry=ParamsGroupSection(CHEMISTRY_MANIFEST),
        print_base=ParamsGroupSection(PRINT_BASE_MANIFEST),
        film_chemistry=ParamsGroupSection(FILM_CHEMISTRY_MANIFEST),
        film_base=ParamsGroupSection(FILM_BASE_MANIFEST),
        convert=ParamsGroupSection(CONVERT_MANIFEST),
        glare=glare,
        input_gamut_compress=ParamsGroupSection(INPUT_GAMUT_COMPRESS_MANIFEST),
        output_gamut_compress=ParamsGroupSection(OUTPUT_GAMUT_COMPRESS_MANIFEST),
        special=special,
        simulation=simulation,
        preview_crop=PreviewCropSection(input_image),
        camera=camera,
        enlarger=EnlargerSection(simulation),
        scanner=scanner,
        spectral_upsampling=SpectralUpsamplingSection(input_image),
        output=OutputSection(simulation),
    )
