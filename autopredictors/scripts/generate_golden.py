import sys
git_path = '/orions4-zfs/projects/lins2/Panos_Space/Git_Repos'
sys.path.insert(0, git_path)

n_points = 5000

segs_dir = '/orions4-zfs/projects/lins2/Panos_Space/DATA/SN_point_clouds_full_shape/'+str(n_points)+'/with_segmentations/03001627'
gold_file = '/orions4-zfs/projects/lins2/Panos_Space/DATA/gold_GT_cc_3_6_only.txt'

only_hks = False
only_cc = True
min_parts = 3
max_parts = 6

import autopredictors.scripts.prepare_data_for_torch as pdt
pdt.generate_gold_standar(segs_dir, gold_file, n_shapes='all', min_parts=min_parts, max_parts=max_parts, only_cc=only_cc, only_hks=only_hks, seed=0)
