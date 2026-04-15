import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist


trackers = []

## 修改
# dataset_name = 'otb'
# dataset_name = 'uav'
# dataset_name = 'vasttrack' # lasot_extension_subset
# dataset_name = 'lasot_extension_subset'
dataset_name = 'lasot'

trackers.extend(trackerlist(name='m_step', parameter_name='baseline_mste_tpe_full_16_18_20', dataset_name=dataset_name,run_ids=299, display_name='m_step'))
# For VOT evaluate
dataset = get_dataset(dataset_name)
# dataset = get_dataset('otb', 'nfs', 'uav', 'tc128ce')
#plot_results(trackers, dataset,dataset_name, merge_results=True, plot_types=('success', 'norm_prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))
# print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec'))
# print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec'))

