import numpy as np
import scipy
import colour
import scipy.interpolate
import matplotlib.pyplot as plt
from spektrafilm.config import SPECTRAL_SHAPE
from spektrafilm.utils.io import load_dichroic_filters, load_filter

################################################################################
# Color Filter class
################################################################################

def create_combined_dichroic_filter(wavelength,
                                    transitions,
                                    edges,
                                    ):
    # data from https://qd-europe.com/se/en/product/dichroic-filters-and-sets/
    dichroics = np.zeros((np.size(wavelength),3))
    dichroics[:,2] = scipy.special.erf( (wavelength-edges[0])/transitions[0])
    dichroics[:,1][wavelength<=550] = -scipy.special.erf( (wavelength[wavelength<=550]-edges[1])/transitions[1])
    dichroics[:,1][wavelength>550] = scipy.special.erf( (wavelength[wavelength>550]-edges[2])/transitions[2])
    dichroics[:,0] = -scipy.special.erf( (wavelength-edges[3])/transitions[3])
    dichroics = dichroics/2 + 0.5
    return dichroics

class DichroicFilters():
    def __init__(self,
                 brand='thorlabs'):
        self.wavelengths = SPECTRAL_SHAPE.wavelengths
        self.filters = np.zeros((np.size(self.wavelengths), 3))
        if brand == 'custom':
            self.create_custom_filters()
        else:
            self.filters = load_dichroic_filters(self.wavelengths, brand)
            
    def plot(self):
        colors = ['tab:cyan', 'tab:pink', 'gold']
        _, ax = plt.subplots()
        for i in range(3):
            ax.plot(self.wavelengths, self.filters[:,i], color=colors[i])
        ax.set_ylabel('Transmittance')
        ax.set_xlabel('Wavelegnth (nm)')
        ax.set_ylim(0,1)
        ax.set_xlim(np.min(self.wavelengths), np.max(self.wavelengths))
        ax.legend(('C','M','Y'))
    
    def apply(self, illuminant, filter_transmittance_values=[1,1,1]):
        dimmed_filters = 1 - (1-self.filters)*(1-np.array(filter_transmittance_values)) # following durst 605 wheels values, with 170 max
        total_filter = np.prod(dimmed_filters, axis=1)
        filtered_illuminant = illuminant*total_filter
        return filtered_illuminant
    
    def apply_cc(self, illuminant, filter_cc_values=[0,0,0]):
        # Filter values are in Kodak CC units proportional to density, 100 units means 1.0 density, or 90% reduction in transmittance
        filter_transmittance_values = 10 ** -(np.array(filter_cc_values)/100.0)
        return self.apply(illuminant, filter_transmittance_values=filter_transmittance_values)
    
    def create_custom_filters(self,
                               edges=[516,500,610,607],
                               transitions=[12,8,8,8]):
        self.filters = create_combined_dichroic_filter(self.wavelengths,
                                                       transitions=transitions,
                                                       edges=edges)

class GenericFilter():
    def __init__(self,
                 name='KG3',
                 type='heat_absorbing',
                 brand='schott',
                 data_in_percentage=False,
                 load_from_database=True):
        self.wavelengths = SPECTRAL_SHAPE.wavelengths
        self.type = type
        self.brand = brand
        self.transmittance = np.zeros_like(self.wavelengths)
        if load_from_database:
            self.transmittance = load_filter(self.wavelengths, name, brand, type,
                                             percent_transmittance=data_in_percentage)
    
    def apply(self, illuminant, value=1.0):
        dimmed_filter = 1 - (1-self.transmittance)*value
        filtered_illuminant = illuminant*dimmed_filter
        return filtered_illuminant

################################################################################
# Band pass filter

def sigmoid_erf(x, center, width=1):
    return scipy.special.erf((x-center)/width)*0.5+0.5

def compute_band_pass_filter(filter_uv=[1, 410, 8], filter_ir=[1, 675, 15]):
    amp_uv = filter_uv[0]
    wl_uv = filter_uv[1]
    width_uv = filter_uv[2]
    
    amp_ir = filter_ir[0]
    wl_ir = filter_ir[1]
    width_ir = filter_ir[2]
    
    amp_uv = np.clip(amp_uv, 0, 1)
    amp_ir = np.clip(amp_ir, 0, 1)
    
    wl = SPECTRAL_SHAPE.wavelengths
    filter_uv  = 1-amp_uv + amp_uv*sigmoid_erf(wl, wl_uv, width=width_uv)
    filter_ir  = 1-amp_ir + amp_ir*sigmoid_erf(wl, wl_ir, width=-width_ir)
    band_pass_filter = filter_uv * filter_ir
    return  band_pass_filter
        

# color filter variables
dichroic_filters = DichroicFilters()
thorlabs_dichroic_filters = DichroicFilters(brand='thorlabs')
edmund_optics_dichroic_filters = DichroicFilters(brand='edmund_optics')
durst_digital_light_dicrhoic_filters = DichroicFilters(brand='durst_digital_light')
custom_dichroic_filters = DichroicFilters(brand='custom')
schott_kg1_heat_filter = GenericFilter(name='KG1', type='heat_absorbing', brand='schott')
schott_kg3_heat_filter = GenericFilter(name='KG3', type='heat_absorbing', brand='schott')
schott_kg5_heat_filter = GenericFilter(name='KG5', type='heat_absorbing', brand='schott')
generic_lens_transmission = GenericFilter(name='canon_24_f28_is', type='lens_transmission',
                                          brand='canon', data_in_percentage=True)


################################################################################

def color_enlarger(light_source, filter_cc_values=(0,65,55),
                   filters=custom_dichroic_filters):
    # Filter values are in Kodak CC units proportional to density, 100 units means 1.0 density, or 90% reduction in transmittance
    # cc_filter_values are in CMY order
    filter_cc_values = np.array(filter_cc_values)
    filtered_illuminant = filters.apply_cc(light_source, filter_cc_values=filter_cc_values)
    return filtered_illuminant

if __name__=="__main__":
    from spektrafilm.model.illuminants import standard_illuminant
    from spektrafilm.profiles.io import load_profile
    
    filters = DichroicFilters(brand='durst_digital_light')
    filters.plot()
    plt.title('Durst Digital Light Dichroic Filters')
    
    filters = DichroicFilters(brand='thorlabs')
    filters.plot()
    plt.title('Thorlabs Dichroic Filters')
    
    filters = DichroicFilters(brand='edmund_optics')
    filters.plot()
    plt.title('Edmund Optics Dichroic Filters')
    
    plt.figure()
    d65 = standard_illuminant('D55')
    # plt.plot(SPECTRAL_SHAPE.wavelengths, d65)
    # plt.plot(SPECTRAL_SHAPE.wavelengths, filters.apply(d65, [0.8,0.0,0])[:])
    plt.plot(SPECTRAL_SHAPE.wavelengths, schott_kg3_heat_filter.transmittance)
    plt.plot(SPECTRAL_SHAPE.wavelengths, generic_lens_transmission.transmittance)
    plt.plot(SPECTRAL_SHAPE.wavelengths, schott_kg3_heat_filter.transmittance*generic_lens_transmission.transmittance)
    plt.ylim((0,1))
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.legend(('Schott KG3', 'Canon Lens Transmittance', 'Combined'))


    filter_cc_values_a = (50.0, 50.0, 50.0)
    filter_cc_values_b = (0.0, 30.0, 20.0)
    paper_profile_name = 'kodak_portra_endura'

    def format_filter_cc_values(values):
        return f'({values[0]:g}, {values[1]:g}, {values[2]:g})'

    filter_label_a = format_filter_cc_values(filter_cc_values_a)
    filter_label_b = format_filter_cc_values(filter_cc_values_b)

    fig, (ax_filters, ax_spectra) = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={'height_ratios': [1, 3]},
    )
    th_kg3_l = standard_illuminant('TH-KG3')
    th_kg3_l_a = color_enlarger(th_kg3_l, filter_cc_values=filter_cc_values_a)
    th_kg3_l_b = color_enlarger(th_kg3_l, filter_cc_values=filter_cc_values_b)
    dichroich_cmy_filters = custom_dichroic_filters.filters
    filter_none = np.ones_like(SPECTRAL_SHAPE.wavelengths, dtype=float)
    filter_a = color_enlarger(filter_none, filter_cc_values=filter_cc_values_a)
    filter_b = color_enlarger(filter_none, filter_cc_values=filter_cc_values_b)
    paper_profile = load_profile(paper_profile_name)
    paper_profile_label = getattr(paper_profile.info, 'name', paper_profile_name)
    paper_log_sensitivity = np.asarray(paper_profile.data.log_sensitivity, dtype=float)
    if paper_log_sensitivity.ndim == 2 and paper_log_sensitivity.shape[0] == 3 and paper_log_sensitivity.shape[1] != 3:
        paper_log_sensitivity = paper_log_sensitivity.T
    paper_sensitivity = 10 ** paper_log_sensitivity
    paper_sensitivity /= np.nanmax(paper_sensitivity, axis=0, keepdims=True)
    ax_filters.plot(SPECTRAL_SHAPE.wavelengths, filter_none, color='0.7', linestyle=':', label='No filter')
    ax_filters.plot(SPECTRAL_SHAPE.wavelengths, dichroich_cmy_filters[:, 0], color='tab:cyan', linestyle='--', label='Pure C filter')
    ax_filters.plot(SPECTRAL_SHAPE.wavelengths, dichroich_cmy_filters[:, 1], color='tab:pink', linestyle='--', label='Pure M filter')
    ax_filters.plot(SPECTRAL_SHAPE.wavelengths, dichroich_cmy_filters[:, 2], color='goldenrod', linestyle='--', label='Pure Y filter')
    ax_filters.plot(SPECTRAL_SHAPE.wavelengths, filter_a, color='tab:orange', label=f'Filter {filter_label_a}')
    ax_filters.plot(SPECTRAL_SHAPE.wavelengths, filter_b, color='tab:purple', label=f'Filter {filter_label_b}')
    ax_filters.set_ylabel('Transmittance')
    ax_filters.set_ylim((0, 1.05))
    ax_filters.legend(loc='lower right')

    ax_spectra.plot(SPECTRAL_SHAPE.wavelengths, th_kg3_l, label='TH-KG3')
    ax_spectra.plot(SPECTRAL_SHAPE.wavelengths, th_kg3_l_a, label=f'TH-KG3 + {filter_label_a}')
    ax_spectra.plot(SPECTRAL_SHAPE.wavelengths, th_kg3_l_b, label=f'TH-KG3 + {filter_label_b}')
    ax_spectra.plot(SPECTRAL_SHAPE.wavelengths, dichroich_cmy_filters[:, 0], color='tab:cyan', linestyle=':', alpha=0.7, label='Pure C filter')
    ax_spectra.plot(SPECTRAL_SHAPE.wavelengths, dichroich_cmy_filters[:, 1], color='tab:pink', linestyle=':', alpha=0.7, label='Pure M filter')
    ax_spectra.plot(SPECTRAL_SHAPE.wavelengths, dichroich_cmy_filters[:, 2], color='goldenrod', linestyle=':', alpha=0.7, label='Pure Y filter')
    ax_spectra.plot(SPECTRAL_SHAPE.wavelengths, paper_sensitivity[:, 0], color='tab:red', linestyle='--', label=f'{paper_profile_label} R sensitivity')
    ax_spectra.plot(SPECTRAL_SHAPE.wavelengths, paper_sensitivity[:, 1], color='tab:green', linestyle='--', label=f'{paper_profile_label} G sensitivity')
    ax_spectra.plot(SPECTRAL_SHAPE.wavelengths, paper_sensitivity[:, 2], color='tab:blue', linestyle='--', label=f'{paper_profile_label} B sensitivity')
    ax_spectra.set_xlabel('Wavelength (nm)')
    ax_spectra.set_ylabel('Normalized intensity / sensitivity')
    ax_spectra.set_title(f'TH-KG3 with Color Enlarger Filters and {paper_profile_label}')
    ax_spectra.legend()
    fig.tight_layout()


    durst_pure_cmy_filters = durst_digital_light_dicrhoic_filters.filters
    custom_pure_cmy_filters = custom_dichroic_filters.filters
    fig_compare, ax_compare = plt.subplots()
    compare_colors = ['tab:cyan', 'tab:pink', 'goldenrod']
    compare_labels = ['C', 'M', 'Y']
    for index, (color, label) in enumerate(zip(compare_colors, compare_labels)):
        ax_compare.plot(
            SPECTRAL_SHAPE.wavelengths,
            durst_pure_cmy_filters[:, index]/np.nanmax(durst_pure_cmy_filters[:, index]),
            color=color,
            label=f'Durst {label}',
        )
        ax_compare.plot(
            SPECTRAL_SHAPE.wavelengths,
            custom_pure_cmy_filters[:, index]/np.nanmax(custom_pure_cmy_filters[:, index]),
            color=color,
            linestyle='--',
            label=f'Custom {label}',
        )
    ax_compare.set_xlabel('Wavelength (nm)')
    ax_compare.set_ylabel('Normalized Transmittance')
    ax_compare.set_ylim((0, 1.05))
    ax_compare.set_title('Durst Digital Light vs Custom Dichroic Filters')
    ax_compare.legend()
    fig_compare.tight_layout()


    band_pass = compute_band_pass_filter()
    negative_profile_name = 'kodak_portra_400'
    negative_profile = load_profile(negative_profile_name)
    negative_profile_label = getattr(negative_profile.info, 'name', negative_profile_name)
    negative_log_sensitivity = np.asarray(negative_profile.data.log_sensitivity, dtype=float)
    if negative_log_sensitivity.ndim == 2 and negative_log_sensitivity.shape[0] == 3 and negative_log_sensitivity.shape[1] != 3:
        negative_log_sensitivity = negative_log_sensitivity.T
    negative_sensitivity = 10 ** negative_log_sensitivity
    negative_sensitivity /= np.nanmax(negative_sensitivity, axis=0, keepdims=True)

    fig_bp, ax_bp = plt.subplots()
    ax_bp.plot(SPECTRAL_SHAPE.wavelengths, band_pass, color='black', label='Band pass filter')
    ax_bp.plot(SPECTRAL_SHAPE.wavelengths, negative_sensitivity[:, 0], color='tab:red', linestyle='--', label=f'{negative_profile_label} R sensitivity')
    ax_bp.plot(SPECTRAL_SHAPE.wavelengths, negative_sensitivity[:, 1], color='tab:green', linestyle='--', label=f'{negative_profile_label} G sensitivity')
    ax_bp.plot(SPECTRAL_SHAPE.wavelengths, negative_sensitivity[:, 2], color='tab:blue', linestyle='--', label=f'{negative_profile_label} B sensitivity')
    ax_bp.plot(SPECTRAL_SHAPE.wavelengths, band_pass * negative_sensitivity[:, 0], color='tab:red', alpha=0.5, label=f'{negative_profile_label} R after band pass')
    ax_bp.plot(SPECTRAL_SHAPE.wavelengths, band_pass * negative_sensitivity[:, 1], color='tab:green', alpha=0.5, label=f'{negative_profile_label} G after band pass')
    ax_bp.plot(SPECTRAL_SHAPE.wavelengths, band_pass * negative_sensitivity[:, 2], color='tab:blue', alpha=0.5, label=f'{negative_profile_label} B after band pass')
    ax_bp.set_xlabel('Wavelength (nm)')
    ax_bp.set_ylabel('Transmittance / normalized sensitivity')
    ax_bp.set_ylim((0, 1.05))
    ax_bp.set_xlim(np.min(SPECTRAL_SHAPE.wavelengths), np.max(SPECTRAL_SHAPE.wavelengths))
    ax_bp.set_title(f'Band Pass Filter and {negative_profile_label} Sensitivities')
    ax_bp.legend(loc='upper right', fontsize='small')
    fig_bp.tight_layout()

    plt.show()

