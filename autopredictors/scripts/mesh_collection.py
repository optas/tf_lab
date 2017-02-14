import glob
import os.path as osp
from geo_tool.in_out.soup import load_mesh_from_file 
from general_tools.in_out.basics import files_in_subdirs
import sys

def write_n_nodes_of_meshes(collection_dir, out_file, mesh_ext='.obj', verbose=False):
    '''
    Writes in a file the number of nodes each mesh under the collection_dir has.
    '''    
    with open(out_file, 'w') as f_out:
        f_out.write('# file_name - Number of Mesh Nodes\n')
        for file_name in files_in_subdirs(collection_dir, mesh_ext+'$'):                
            vertices = load_mesh_from_file(file_name)[0]
            line = file_name + ' %d' % (vertices.shape[0], ) + '\n'
            f_out.write(line)
            if verbose: 
                print line
        
    
if __name__ == '__main__':
    if sys.argv[1] == 'write_n_nodes_of_meshes':
        write_n_nodes_of_meshes(sys.argv[2], sys.argv[3], verbose=True)
    
    