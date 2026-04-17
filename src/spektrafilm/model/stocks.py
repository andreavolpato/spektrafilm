from enum import Enum

class FilmStocks(Enum):
    # kodak pro
    kodak_ektar_100 = 'kodak_ektar_100'
    kodak_portra_160 = 'kodak_portra_160'
    kodak_portra_400 = 'kodak_portra_400'
    kodak_portra_800 = 'kodak_portra_800'
    kodak_portra_800_push1 = 'kodak_portra_800_push1'
    kodak_portra_800_push2 = 'kodak_portra_800_push2'
    # kodak consumer
    kodak_gold_200 = 'kodak_gold_200'
    kodak_ultramax_400 = 'kodak_ultramax_400'
    # kodak cine
    kodak_vision3_50d = 'kodak_vision3_50d'
    kodak_vision3_250d = 'kodak_vision3_250d'
    kodak_verita_200d = 'kodak_verita_200d'
    kodak_vision3_200t = 'kodak_vision3_200t'
    kodak_vision3_500t = 'kodak_vision3_500t'
    # fuji pro
    fujifilm_pro_400h = 'fujifilm_pro_400h'
    # fuji consumer
    fujifilm_c200 = 'fujifilm_c200'
    fujifilm_xtra_400 = 'fujifilm_xtra_400'
    # positive
    kodak_ektachrome_100 = 'kodak_ektachrome_100'
    kodak_kodachrome_64 = 'kodak_kodachrome_64'
    fujifilm_velvia_100 = 'fujifilm_velvia_100'
    fujifilm_provia_100f = 'fujifilm_provia_100f'

class PrintPapers(Enum):
    kodak_ultra_endura = 'kodak_ultra_endura' # problematic
    kodak_endura_premier = 'kodak_endura_premier'
    kodak_ektacolor_edge = 'kodak_ektacolor_edge'
    kodak_supra_endura = 'kodak_supra_endura'
    kodak_portra_endura = 'kodak_portra_endura'
    fujifilm_crystal_archive_typeii = 'fujifilm_crystal_archive_typeii'
    kodak_2383 = 'kodak_2383'
    kodak_2393 = 'kodak_2393'
