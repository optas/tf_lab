'''Created on December 22, 2016

@author: optas

General output specifications:
    Point-clouds are lex-sorted and centered in the unit sphere, dtype is np.float32.
    They are stored as the first 3 columns of each output file.
    The fourth column of an output file (when it exists) indicates for the corresponding point in which segment it belongs to.

Requires: in PYTHONPATH geo_tool, general_tools
'''

import sys
import glob
import numpy as np
import os.path as osp
import matplotlib.pylab as plt
from multiprocessing import Pool
from sklearn.cluster import KMeans

from general_tools.arrays.is_true import is_contiguous
from general_tools.arrays.transform import make_contiguous
from general_tools.in_out.basics import boot_strap_lines_of_file, create_dir
import general_tools.arrays.is_true

from geo_tool import Point_Cloud, Mesh, Laplace_Beltrami
import geo_tool.in_out.soup as gio
import geo_tool.solids.mesh_cleaning as cleaning
import geo_tool.signatures.node_signatures as ns

import helper   # TODO  Make it relative

segs_ext = helper.segs_extension
pts_ext = helper.points_extension


def only_uniform_sampling(mesh_file, out_folder, n_samples, swap_y_z=True, dtype=np.float32):
    ''' Given a mesh, it computes a point-cloud that is uniformly sampled
    from its area elements.
    '''

    in_mesh = Mesh(file_name=mesh_file)
    model_id = mesh_file.split('/')[-2]
    if swap_y_z:
        in_mesh.swap_axes_of_vertices([0, 2, 1])
    in_mesh = cleaning.clean_mesh(in_mesh)
    ss_points, _ = in_mesh.sample_faces(n_samples)
    pc = Point_Cloud(ss_points.astype(dtype))
    pc = pc.center_in_unit_sphere()
    pc, _ = pc.lex_sort()
    out_file = osp.join(out_folder, model_id + pts_ext)
    np.savetxt(out_file, pc.points)


def uniform_sampling_of_connected_components(file_name, out_folder, n_samples, area_bound, swap_y_z=True, seed=None):
    '''Assumes meshes from Shape-Net.
    '''

    if seed is not None:
        np.random.seed(seed)

    in_mesh = Mesh(file_name=file_name)
    model_id = file_name.split('/')[-2]
    if swap_y_z:
            in_mesh.swap_axes_of_vertices([0, 2, 1])

    in_mesh.center_in_unit_sphere()
    in_mesh = cleaning.clean_mesh(in_mesh)
    n_cc, cc = in_mesh.connected_components()

    if n_cc > 1:
        print 'Processing model: ' + model_id
        print in_mesh
        print 'Number of Connected Components: %i.' % (n_cc)

        areas = in_mesh.area_of_triangles()
        ftr = in_mesh.triangles[:, 0]   # First node of each triangle.
        accmap = cc[ftr]
        total_area_in_cc = np.bincount(accmap, weights=areas)  # How much is the area of the triangles of each CC.

        good_components = np.where(total_area_in_cc >= area_bound * np.sum(areas))[0]   # interesting components
        good_vertices = np.where(np.in1d(cc, good_components))
        cc_mesh = cleaning.filter_vertices(in_mesh, good_vertices)
        cc_mesh.name = model_id
        f_cc, cc_mesh.cc = cc_mesh.connected_components()

        save_dir = create_dir(osp.join(out_folder, cc_mesh.name))
        cc_mesh.save(osp.join(save_dir, 'model.cpkl'))

        all_pts = []
#         all_bboxs = []
        # Generate Point_Cloud for each part.
        for i in xrange(f_cc):
            nodes_i = np.where(cc_mesh.cc == i)
            part_i = cleaning.filter_vertices(cc_mesh.copy(), nodes_i)
            bi = part_i.bounding_box()
            ss_points, _ = part_i.sample_faces(n_samples)
            pc = Point_Cloud(ss_points)
            pc, _ = pc.lex_sort()
            out_file = osp.join(save_dir, 'part_' + str(i) + '.pts')
            header = 'bbox extrema = ' + str(bi.extrema) + '\n'
            header += 'bbox volume = ' + str(bi.volume())
            np.savetxt(out_file, pc.points, header=header)
            out_file = osp.join(save_dir, 'part_' + str(i) + '.bbox')
            all_pts.append(pc.points)
#             all_bboxs.append(bi)

        # Output image
        plt.ioff()
        colors = []
        for i, pt in enumerate(all_pts):
            colors.append(np.ones(len(pt)) * i)
        c = np.vstack(colors).ravel()
        pts = np.vstack(all_pts)
        fig = Point_Cloud(points=pts).plot(show=False, c=c)
        fig.suptitle('Number of filtered cc = %d (out of %d).' % (f_cc, n_cc))
        fig.savefig(osp.join(save_dir, 'image.png'))
        plt.close()
        return n_cc


def eric_prepare_io(data_top_dir, out_top_dir, synth_id, boot_n):
    points_top_dir = osp.join(data_top_dir, synth_id, 'points')
    segs_top_dir = osp.join(data_top_dir, synth_id, 'expert_verified', 'points_label')
    original_density_dir = osp.join(out_top_dir, synth_id, 'original_density')
    bstrapped_out_dir = osp.join(out_top_dir, synth_id, 'bootstrapped_' + str(boot_n))
    create_dir(original_density_dir)
    create_dir(bstrapped_out_dir)
    return segs_top_dir, points_top_dir, original_density_dir, bstrapped_out_dir


def eric_annotated(data_top_dir, out_top_dir, synth_id, boot_n=2700):
    ''' Writes out point clouds with a segmentation mask according to Eric's annotation.
    The point clouds are 1) the original point clouds that Eric sampled  2) a bootstrapped version of them.
    '''

    segs_top_dir, points_top_dir, original_density_dir, bstrapped_out_dir = eric_prepare_io(data_top_dir, out_top_dir, synth_id, boot_n)

    erics_seg_extension = '.seg'
    erics_points_extension = '.pts'

    for file_name in glob.glob(osp.join(segs_top_dir, '*' + erics_seg_extension)):
        model_name = osp.basename(file_name)[:-len(erics_seg_extension)]
        pt_file = osp.join(points_top_dir, model_name + erics_points_extension)
        points = gio.load_crude_point_cloud(pt_file, permute=[0, 2, 1])
        n_points = points.shape[0]
        pc = Point_Cloud(points=points)
        pc = pc.center_in_unit_sphere()
        pc, lex_index = pc.lex_sort()

        gt_seg = np.loadtxt(file_name, dtype=np.float32)
        gt_seg = gt_seg[lex_index]
        gt_seg = gt_seg.reshape((n_points, 1))
        seg_ids = np.unique(gt_seg)
        if seg_ids[0] == 0:
            seg_ids = seg_ids[1:]   # Zero is not a real segment.

        header_str = 'erics-annotated_segs\nseg_ids=%s' % (str(seg_ids.astype(np.int)).strip('[]'))
        out_data = np.hstack((pc.points, gt_seg))
        out_file = osp.join(original_density_dir, model_name + segs_ext)
        np.savetxt(out_file, out_data, header=header_str)
        boot_strap_lines_of_file(out_file, boot_n, osp.join(bstrapped_out_dir, model_name + segs_ext), skip_rows=2)


def laplacian_coloring(file_name, out_folder, n_samples, n_eigs, swap_y_z=True):
    in_mesh = Mesh(file_name=file_name)
    model_id = file_name.split('/')[-2]
    print 'Processing model: ' + model_id

    if swap_y_z:
            in_mesh.swap_axes_of_vertices([0, 2, 1])

    in_mesh = cleaning.clean_mesh(in_mesh)
    in_mesh.center_in_unit_sphere()

    try:
        lb = Laplace_Beltrami(in_mesh)
        evals, evecs = lb.spectra(n_eigs)

        out_file = osp.join(out_folder, model_id + '_spectra_' + str(n_eigs) + '.npz')
        np.savez(out_file, evals, evecs)

        ss_points, sample_face_id = in_mesh.sample_faces(n_samples)
        pcolors = np.zeros((n_samples, n_eigs))
        for ev_id in xrange(n_eigs):
            v_func = evecs[:, ev_id]
            pcolor = in_mesh.barycentric_interpolation_of_vertex_function(v_func, ss_points, sample_face_id)
            pcolors[:, ev_id] = pcolor

        out_file = osp.join(out_folder, model_id + '_spectra_' + str(n_eigs) + '_pcloud.npz')
        np.savez(out_file, ss_points, pcolors)
    except:
        print 'MODEL FAILED! : ' + model_id


def crude_shape_segmentations(file_name, out_folder, n_samples, swap_y_z=True, seed=None, save_img=False):
    max_eigs = 200
    min_vertices = 50   # Models with one CC and less than min_vertices are ignored.
    n_hks_segments = 20
    n_hks_time = 20
    max_cc_segments = 30

    if seed is not None:
        np.random.seed(seed)

    in_mesh = Mesh(file_name=file_name)
    model_id = file_name.split('/')[-2]
    hks_activated = False
    try:
        if swap_y_z:
            in_mesh.swap_axes_of_vertices([0, 2, 1])

        in_mesh = cleaning.clean_mesh(in_mesh)
        n_cc, cc = in_mesh.connected_components()

        print 'Processing model: ' + model_id
        print in_mesh
        print 'Number of Connected Components: %i.' % (n_cc)

        if n_cc == 1 and in_mesh.num_vertices > min_vertices:
            hks_activated = True
            n_eigs = min(max_eigs, in_mesh.num_vertices - 30)
            in_lb = Laplace_Beltrami(in_mesh)
            hks = ns.heat_kernel_embedding(in_lb, n_eigs, n_hks_time)
            estimator = KMeans(init='k-means++', n_clusters=n_hks_segments, n_init=10)
            estimator.fit(hks)
            segments_indicator = estimator.labels_ + 1
        elif n_cc > 2:
            areas = in_mesh.area_of_triangles()
            ftr = in_mesh.triangles[:, 0]   # First node of each triangle.
            accmap = cc[ftr]
            total_area_in_cc = np.bincount(accmap, weights=areas)  # How much is the area of the triangles of each CC.
            decreasing_index = [np.argsort(total_area_in_cc)[::-1]]
            segments_indicator = np.zeros(in_mesh.num_vertices, dtype=np.int)
            for s in xrange(min(n_cc, max_cc_segments)):
                segments_indicator[cc == decreasing_index[0][s]] = s + 1
        else:
            print "Skipping model: " + model_id
            return

        # Generate Point_Cloud for entire shape.
        ss_points, sample_face_idx = in_mesh.sample_faces(n_samples)
        pc = Point_Cloud(ss_points)
        pc = pc.center_in_unit_sphere()
        pc, lex_indices = pc.lex_sort()

        point_seg_ids = segments_indicator[in_mesh.triangles[sample_face_idx]][:, 1]
        point_seg_ids = point_seg_ids[lex_indices]
        point_seg_ids = make_contiguous(point_seg_ids, start=np.min(point_seg_ids)).reshape((pc.num_points, 1))

        output_data = np.hstack((pc.points, point_seg_ids))
        out_file = osp.join(out_folder, model_id + segs_ext)

        n_segs = np.unique(point_seg_ids)
        n_segs = len(n_segs[n_segs != 0])

        with open(out_file, 'w') as f_out:
            if hks_activated:
                np.savetxt(f_out, output_data, header='hks_based_segs\nn_segs=%d\n' % (n_segs,))
            else:
                np.savetxt(f_out, output_data, header='cc_based_segs\nn_segs=%d\n' % (n_segs,))

        if save_img:
            fig = pc.plot(show=False, c=point_seg_ids)
            fig.suptitle('#mesh nodes=%d #CC=%d' % (in_mesh.num_vertices, n_cc))
            fig.savefig(osp.join(out_folder, model_id + '.png'))
            plt.close()
    except:
        print 'MODEL FAILED! : ' + model_id


def restrict_mesh_on_specific_faces_only(in_mesh, sample_face_id):
    ''' Used currently for visualization of how bad a sampling of a point-cloud is.
    '''
    ref_vertex_soup = in_mesh.vertices[in_mesh.triangles[sample_face_id], :]
    ref_vertex_soup = ref_vertex_soup.squeeze()
    vertices = ref_vertex_soup.reshape(ref_vertex_soup.shape[0] * 3, 3)
    n_tr = ref_vertex_soup.shape[0]
    triangles = np.zeros(shape=(n_tr, 3), dtype=np.int32)
    for i in xrange(n_tr):
        tr_start = i * 3
        triangles[i, :] = np.arange(tr_start, tr_start + 3)

    mesh_from_pcloud = Mesh(vertices=vertices, triangles=triangles)
    return mesh_from_pcloud


def main():
    '''
    python tf_lab/autopredictors/scripts/pclouds_with_segments.py
        ~/DATA/Shapes/Shape_Net_Core/
        ~/DATA/Shapes/Shape_Net_Core/Point_Clouds/
        sample_pclouds
        2 1024 03001627
    '''

    top_shape_dir = sys.argv[1]     # e.g., '...Shape_Net_Core' where the class folders of shape_net are.
    out_top_dir = sys.argv[2]       # wherever you want to save the point-cloud data
    experiment_type = sys.argv[3]
    n_threads = int(sys.argv[4])

    if experiment_type == 'sample_pclouds':
        dispatch_f = only_uniform_sampling
        n_samples = int(sys.argv[5])
        arg_tuple = (n_samples,)
        out_top_dir = osp.join(out_top_dir, str(n_samples))
        if len(sys.argv) == 7:
            synth_id = sys.argv[6]
        else:
            synth_id = None

    elif experiment_type == 'sample_cc_modules':
        dispatch_f = uniform_sampling_of_connected_components
        n_samples = int(sys.argv[5])
        area_bound = float(sys.argv[6])
        arg_tuple = (n_samples, area_bound)
        out_top_dir = osp.join(out_top_dir, str(n_samples), 'area_b_' + str(area_bound))
        if len(sys.argv) == 8:
            synth_id = sys.argv[7]
        else:
            synth_id = None

    elif experiment_type == 'laplacian_coloring':
        dispatch_f = laplacian_coloring
        n_samples = int(sys.argv[5])
        n_eigs = int(sys.argv[6])
        arg_tuple = (n_samples, n_eigs)
        out_top_dir = osp.join(out_top_dir, str(n_samples), 'n_eigs_' + str(n_eigs))
        if len(sys.argv) == 8:
            synth_id = sys.argv[7]
        else:
            synth_id = None

    parallel = False
    if n_threads > 1:
        parallel = True

    create_dir(out_top_dir)
    synth_id_to_category = helper.shape_net_core_synth_id_to_category

    if parallel:
        pool = Pool(n_threads)
        print 'Starting a pool with %d workers.' % (n_threads, )

    if synth_id is not None:    # Run on single category.
        shape_iterator = [synth_id]
    else:
        shape_iterator = synth_id_to_category.keys()

    for synth_id in shape_iterator:
        out_folder = osp.join(out_top_dir, synth_id)
        create_dir(out_folder)
        for file_name in glob.glob(osp.join(top_shape_dir, synth_id, '*', 'model.obj')):
            final_arg = (file_name, out_folder) + arg_tuple
            if parallel:
                pool.apply_async(dispatch_f, final_arg)
            else:
                dispatch_f(*final_arg)

    if parallel:
        pool.close()
        pool.join()

if __name__ == '__main__':
    main()
