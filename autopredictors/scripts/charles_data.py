'''
Created on Mar 23, 2017

@author: optas
'''


class CharlesFusedData(object):
    ''' Info regarding Charle's OBJ mesh files which are fused by multiple views and are posted on his webpage.
    The rotations corresponding to each object are those necessary to be applied so that the transformed point-clouds/meshes
    are aligned with ShapeNet (i.e., with our training data).
    '''
    chair_rotate_degrees = [-45, -150, -135, -115, -110, 0, -25, -10, 90, -30, -130, -45, -135, 90, 90, 60, 0]
    chair_model_names = ['chair_scan_0001', 'chair_scan_0010', 'chair_scan_0016', 'chair_scan_0007',
                         'chair_scan_0011', 'chair_scan_0006', 'chair_scan_0017', 'chair_scan_0012',
                         'chair_scan_0008', 'chair_scan_0003', 'chair_scan_0005', 'chair_scan_0014',
                         'chair_scan_0002', 'chair_scan_0009', 'chair_scan_0013', 'chair_scan_0015',
                         'chair_scan_0004']

    table_rotate_degrees = [30, -65, 65, -80, 20, 10, 120, -8, 40, 20, 30, 20, 10, 60, 25, 35, 10, -10]
    table_azimuth_angles = [240, 0, 50, 240, 240, 240, 240, 240, 240, 240, 20, 200, 200, 200, 240, 240, 240, 30]
    table_model_names = ['table_scan_0014', 'table_scan_0005', 'table_scan_0003', 'table_scan_0012',
                         'table_scan_0008', 'table_scan_0004', 'table_scan_0015', 'table_scan_0009',
                         'table_scan_0013', 'table_scan_0018', 'table_scan_0002', 'table_scan_0007',
                         'table_scan_0016', 'table_scan_0010', 'table_scan_0001', 'table_scan_0017',
                         'table_scan_0006', 'table_scan_0011']

    sofa_rotate_degrees = [0, 0, 100, -130, 10, -65, 60, -35, 60, 155, 120, 110, 70, 10, 0, 120, 67, 5, 5, 30, 0, 120, 5, 110, -83, -150]
    sofa_model_names = ['sofa_scan_0007', 'sofa_scan_0016', 'sofa_scan_0025', 'sofa_scan_0023',
                        'sofa_scan_0010', 'sofa_scan_0001', 'sofa_scan_0017', 'sofa_scan_0006',
                        'sofa_scan_0024', 'sofa_scan_0022', 'sofa_scan_0011', 'sofa_scan_0014',
                        'sofa_scan_0005', 'sofa_scan_0003', 'sofa_scan_0019', 'sofa_scan_0012',
                        'sofa_scan_0008', 'sofa_scan_0021', 'sofa_scan_0026', 'sofa_scan_0004',
                        'sofa_scan_0015', 'sofa_scan_0009', 'sofa_scan_0013', 'sofa_scan_0018',
                        'sofa_scan_0002', 'sofa_scan_0020']

    @staticmethod
    def permute_points(model_name):
        if model_name.startswith('chair'):
            return [2, 0, 1]
        else:
            return [0, 1, 2]

    @staticmethod
    def add_synced_lists_to_dict(in_dict, key_list, val_list):
        for idx, val in enumerate(val_list):
            in_dict[key_list[idx]] = val

    @classmethod
    def rotation_dict(cls):
        rotation_dict = {}
        cls.add_synced_lists_to_dict(rotation_dict, cls.chair_model_names, cls.chair_rotate_degrees)
        cls.add_synced_lists_to_dict(rotation_dict, cls.table_model_names, cls.table_rotate_degrees)
        cls.add_synced_lists_to_dict(rotation_dict, cls.sofa_model_names, cls.sofa_rotate_degrees)
        return rotation_dict