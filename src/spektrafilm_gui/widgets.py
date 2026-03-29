from dataclasses import dataclass

from spektrafilm_gui.widget_editors import BoolEditor, EnumEditor, FloatEditor, FloatTupleEditor, IntEditor, IntTupleEditor
from spektrafilm_gui.widget_primitives import CollapsibleSection, platform_default_font
from spektrafilm_gui.widget_sections import (
    CameraSection,
    CouplersSection,
    DataclassSection,
    DisplaySection,
    EnlargerSection,
    ExposureControlSection,
    FilePickerSection,
    GlareSection,
    GrainSection,
    GuiConfigSection,
    HalationSection,
    InputImageSection,
    LoadRawSection,
    OutputSection,
    PreflashingSection,
    PreviewCropSection,
    ScannerSection,
    SimpleDataclassSection,
    SimulationSection,
    SpecialSection,
    SpectralUpsamplingSection,
    TuneSection,
    _build_auxiliary_label,
    _build_button,
    _build_button_row,
    _build_collapsible_form_section,
    _build_linked_form_section,
    _build_vertical_container,
    _build_widget_label,
    _enum_values,
    _format_label,
    _new_form_layout,
    _set_single_collapsible_layout,
    _spec_row,
)


@dataclass(slots=True)
class WidgetBundle:
    filepicker: FilePickerSection
    gui_config: GuiConfigSection
    display: DisplaySection
    input_image: InputImageSection
    load_raw: LoadRawSection
    grain: GrainSection
    preflashing: PreflashingSection
    halation: HalationSection
    couplers: CouplersSection
    glare: GlareSection
    special: SpecialSection
    simulation: SimulationSection
    preview_crop: PreviewCropSection
    camera: CameraSection
    exposure_control: ExposureControlSection
    enlarger: EnlargerSection
    scanner: ScannerSection
    spectral_upsampling: SpectralUpsamplingSection
    tune: TuneSection
    output: OutputSection


def create_widget_bundle() -> WidgetBundle:
    filepicker = FilePickerSection()
    input_image = InputImageSection(filepicker)
    simulation = SimulationSection()
    special = SpecialSection(simulation)

    return WidgetBundle(
        filepicker=filepicker,
        gui_config=GuiConfigSection(),
        display=DisplaySection(),
        input_image=input_image,
        load_raw=LoadRawSection(),
        grain=GrainSection(),
        preflashing=PreflashingSection(),
        halation=HalationSection(),
        couplers=CouplersSection(),
        glare=GlareSection(),
        special=special,
        simulation=simulation,
        preview_crop=PreviewCropSection(input_image),
        camera=CameraSection(simulation),
        exposure_control=ExposureControlSection(simulation),
        enlarger=EnlargerSection(simulation),
        scanner=ScannerSection(simulation),
        spectral_upsampling=SpectralUpsamplingSection(input_image),
        tune=TuneSection(special),
        output=OutputSection(simulation),
    )