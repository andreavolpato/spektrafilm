import matplotlib.pyplot as plt
from agx_emulsion.profiles.factory import create_profile, process_negative_profile, process_paper_profile, plot_profile, replace_fitted_density_curves, adjust_log_exposure
from agx_emulsion.profiles.io import save_profile
from agx_emulsion.profiles.correct import correct_negative_curves_with_gray_ramp, align_midscale_neutral_exposures

process_print_paper = False
process_negative = True

print('----------------------------------------')
print('Paper profiles')
#               label,                               name,                                illu    sens, curv, dye,  dom
paper_info = [('kodak_ektacolor_edge',              'Kodak Ektacolor Edge',               'D50',  None, None, None, 1.0),
              ('kodak_ultra_endura',                'Kodak Professional Ultra Endura',    'D50',  None, None, None, 1.0),
              ('kodak_endura_premier',              'Kodak Professional Endura Premier',  'D50',  None, None, None, 1.0),
              ('kodak_portra_endura',               'Kodak Professional Portra Endura',   'D50',  None, None, None, 1.0),
              ('kodak_supra_endura',                'Kodak Professional Supra Endura',    'D50',  'kodak_portra_endura', None, 'kodak_portra_endura', 1.0),
              ('fujifilm_crystal_archive_typeii',   'Fujifilm Crystal Archive Type II',   'D65',  None, 'kodak_supra_endura', None, 1.0),
              ('kodak_2393',                        'Kodak Vision Premier 2393',          'K75P',  None, None, None, 1.0),
]


if process_print_paper:
    for label, name, illu, sens, curv, dye, dom in paper_info:
        profile = create_profile(stock=label,
                                name=name,
                                type='paper',
                                log_sensitivity_donor=sens,
                                denisty_curves_donor=curv,
                                dye_density_cmy_donor=dye,
                                densitometer='status_A',
                                viewing_illuminant=illu,
                                log_sensitivity_density_over_min=dom)
        save_profile(profile)
        profile = process_paper_profile(profile)
        save_profile(profile, '_uc')


print('----------------------------------------')
print('Negative profiles')

#               label,                 name,                  suffix   dye_donor,    d_over_min, ref_ill target_paper,                align_mid_exp  trustability
stock_info = [('kodak_vision3_50d',   'Kodak Vision3 50D',    '',      None       ,  0.2,        'D55',  'kodak_2393_uc',             None,          0.3),
              ('kodak_portra_400',    'Kodak Portra 400',     '',      'generic_a',  0.2,        'D55',  'kodak_portra_endura_uc',    None,          1.0),
              ('kodak_gold_200',      'Kodak Gold 200',       '',      'generic_a',  0.2,        'D55',  'kodak_portra_endura_uc',    None,          1.0),
              ('kodak_ultramax_400',  'Kodak Ultramax 400',   '',      'generic_a',  0.2,        'D55',  'kodak_portra_endura_uc',    None,          1.0),
              ('fujifilm_pro_400h',   'Fujifilm Pro 400H',    '',      'generic_a',  1.0,        'D65',  'kodak_portra_endura_uc',    'mid',         0.3),
              ('fujifilm_xtra_400',   'Fujifilm X-Tra 400',   '',      'generic_a',  1.0,        'D65',  'kodak_portra_endura_uc',    None,          0.3),
              ('fujifilm_c200',       'Fujifilm C200',        '',      'generic_a',  1.0,        'D65',  'kodak_portra_endura_uc',    'green',       0.3),
              ]

if process_negative:
    for label, name, suff, dye, d_over_min, ref_ill, target_paper, align_mid_exp, trustability in stock_info:
        profile = create_profile(stock=label,
                                 name=name,
                                 type='negative',
                                 densitometer='status_M',
                                 dye_density_cmy_donor=dye,
                                 reference_illuminant=ref_ill,
                                 log_sensitivity_density_over_min=d_over_min)
        save_profile(profile)
        suffix = '_'+suff
        if dye=='generic_a':
            suffix += 'a'
        profile = process_negative_profile(profile)
        save_profile(profile, suffix+'u')
        if align_mid_exp is not None:
            profile = align_midscale_neutral_exposures(profile, reference_channel=align_mid_exp)
        profile = correct_negative_curves_with_gray_ramp(profile, 
                                                        target_paper=target_paper, 
                                                        data_trustability=trustability)
        profile = replace_fitted_density_curves(profile)
        profile = adjust_log_exposure(profile)
        save_profile(profile, 'c')
        plot_profile(profile)

plt.show()