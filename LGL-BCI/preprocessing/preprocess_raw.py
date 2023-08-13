import Net.preprocessing as prep
from Net.preprocessing.config import CONSTANT
k_folds = 5
# pick_smp_freq = 250
pick_smp_freq = 1000
n_components = 4
bands = [3, 30]
save_path = './datasets'
scenario = "CV"
# scenario = "Holdout"

prep.BCIC2a.load_BCIC.subject_setting(k_folds=k_folds,
                                      pick_smp_freq=pick_smp_freq,
                                      bands=bands,
                                      save_path=save_path,
                                      num_class=4,
                                      scenario=scenario,
                                      sel_chs=CONSTANT['BCIC2a']['sel_chs'])


# prep.OpenBMI.load_OpenBMI.subject_setting(k_folds=k_folds,
#                                           pick_smp_freq=pick_smp_freq,
#                                           bands=bands,
#                                           save_path=save_path,
#                                           num_class=2,
#                                           sel_chs=CONSTANT['OpenBMI']['sel_chs'])