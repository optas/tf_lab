
# coding: utf-8

# In[1]:


from general_tools.notebook.gpu_utils import setup_one_gpu
GPU = 3
setup_one_gpu(GPU)


# In[3]:


import numpy as np
import tensorflow as tf
import os
import os.path as osp
import argparse
from functools import partial

import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
from tflearn.layers.core import fully_connected

from general_tools.notebook.tf import reset_tf_graph
from general_tools.notebook.utils import top_data_dir
from general_tools.in_out.basics import create_dir, files_in_subdirs, pickle_data

from tf_lab.neural_net import MODEL_SAVER_ID
from tf_lab.data_sets.shape_net import snc_category_to_synth_id


# In[4]:


VGG_MEAN = [123.68, 116.78, 103.94]


# In[9]:


int_to_label = {}

def list_images(directory, in_view='image_p020_t337_r005'):
    """
    Get all the images and labels in directory/label/model_name/view.png
    """
    labels = os.listdir(directory)
    files_and_labels = []
    for label in labels:
        for f in os.listdir(os.path.join(directory, label)):
            files_and_labels.append((os.path.join(directory, label, f, in_view + '.png'), label))

    filenames, labels = zip(*files_and_labels)
    filenames = list(filenames)
    labels = list(labels)
    unique_labels = list(set(labels))

    label_to_int = {}
    for i, label in enumerate(unique_labels):
        label_to_int[label] = i
        int_to_label[i] = label
        
    labels = [label_to_int[l] for l in labels]    
    return filenames, labels

def check_accuracy(sess, correct_prediction, is_training, dataset_init_op):
    """Check the accuracy of the model on either train 
    or val (depending on dataset_init_op).
    """
    # Initialize the correct dataset.
    sess.run(dataset_init_op)
    num_correct, num_samples = 0, 0
    while True:
        try:
            correct_pred = sess.run(correct_prediction, {is_training: False})
            num_correct += correct_pred.sum()
            num_samples += correct_pred.shape[0]
        except tf.errors.OutOfRangeError:
            break

    # Return the fraction of datapoints that were correctly classified
    acc = float(num_correct) / num_samples
    return acc

def training_preprocess(image, label):
    ''' Preprocessing (for training)
        # (3) Take a random 224x224 crop to the scaled image
        # (4) Horizontally flip the image with probability 1/2
        # (5) Substract the per color mean `VGG_MEAN`
        # Note: we don't normalize the data here, as VGG was trained without normalization
    '''    
#     crop_image = tf.random_crop(image, [224, 224, 3]) #!
    crop_image = image
    flip_image = tf.image.random_flip_left_right(crop_image)
    means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
    centered_image = flip_image - means
    return centered_image, label


def val_preprocess(image, label):
    ''' Preprocessing (for validation)
    Take a central 224x224 crop to the scaled image
    Substract the per color mean `VGG_MEAN`
    # Note: we don't normalize the data here, as VGG was trained without normalization
    '''
    #crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)    # (3)
    crop_image = image
    means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
    centered_image = crop_image - means                                     # (4)
    return centered_image, label


def _parse_function(filename, label):
    ''' # Standard preprocessing for VGG on ImageNet taken from here:
        # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/vgg_preprocessing.py
        # Also see the VGG paper for more details: https://arxiv.org/pdf/1409.1556.pdf
        # Preprocessing (for both training and validation):
        # (1) Decode the image from PNG format
        # (2) Resize the image so its smaller side is 256 pixels long
        '''
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    resized_image = image
    return resized_image, label
#     smallest_side = 256.0
#     height, width = tf.shape(image)[0], tf.shape(image)[1]
#     height = tf.to_float(height)
#     width = tf.to_float(width)

#     scale = tf.cond(tf.greater(height, width),
#                     lambda: smallest_side / width,
#                     lambda: smallest_side / height)
#     new_height = tf.to_int32(height * scale)
#     new_width = tf.to_int32(width * scale)

#     resized_image = tf.image.resize_images(image, [new_height, new_width])
#     return resized_image, label


def real_data_parse_function(filename, label, channels=1):
    image_string = tf.read_file(filename)    
    image_decoded = tf.image.decode_png(image_string, channels=channels)
    image = tf.cast(image_decoded, tf.float32)
    if channels == 1: # gray-scale        
        image = tf.tile(image, [1, 1, 3])
    resized_image = image
    return resized_image, label


def make_dataset(filenames, labels, preprocess_f, shuffle=True, real_data=False):
    filenames = tf.constant(filenames)
    labels = tf.constant(labels)
    
    # Tf.13
    #dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, labels))
    
    # Tf.1.11
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    
    if real_data:
        # TF1.13
#         dataset = dataset.map(real_data_parse_function, num_threads=args.num_workers, output_buffer_size=args.batch_size)
        # TF 1.11
        rp = partial(real_data_parse_function, channels=REAL_CHANNELS)
        dataset = dataset.map(rp)
#         dataset = dataset.map(real_data_parse_function)
    else:    
        # TF1.13
        #dataset = dataset.map(_parse_function, num_threads=args.num_workers, output_buffer_size=args.batch_size)
        # TF 1.11
        dataset = dataset.map(_parse_function)
    
    #dataset = dataset.map(preprocess_f, num_threads=args.num_workers, output_buffer_size=args.batch_size)
    
    dataset = dataset.map(preprocess_f)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
    batched_dataset = dataset.batch(args.batch_size)
    return batched_dataset


def extract_features(vgg_net, filenames, labels, conv=False, real_data=False):
    '''returns features in same order as filenames.
    '''
    batched_test_dataset = make_dataset(filenames, labels, val_preprocess, shuffle=False, real_data=real_data)
    test_init_op = vgg_net.iterator.make_initializer(batched_test_dataset)
    vgg_net.sess.run(test_init_op)
    
    if conv :
        feat = vgg_net.end_points['vgg_16/conv5/conv5_3']
    else:
        feat = vgg_net.end_points['vgg_16/fc7']
    
    all_feats = []
    while True:
        try:
            res = vgg_net.sess.run(feat, {vgg_net.is_training: False})
            all_feats.append(np.squeeze(res))
        except tf.errors.OutOfRangeError:
            break
    all_feats = np.vstack(all_feats).astype(np.float32)
    return all_feats


# In[16]:


class VGG_Finetuner(object):
    def __init__(self, args, train_filenames, train_labels):
        self.args = args            
        n_classes = len(set(train_labels))
        print '# Classes:', n_classes
        batched_train_dataset = make_dataset(train_filenames, train_labels, training_preprocess)
        
        # Now we define an iterator that can operator on either dataset.
        # The iterator can be reinitialized by calling:
        #     - sess.run(train_init_op) for 1 epoch on the training set
        #     - sess.run(val_init_op)   for 1 epoch on the valiation set
        # Once this is done, we don't need to feed any value for images and labels
        # as they are automatically pulled out from the iterator queues.
        # A reinitializable iterator is defined by its structure. We could use the
        # `output_types` and `output_shapes` properties of either `train_dataset`
        # or `validation_dataset` here, because they are compatible.
        
        # TF1.3
        #self.iterator = tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types,
        #                                                   batched_train_dataset.output_shapes)
        
        # Tf.1.11
        self.iterator = tf.data.Iterator.from_structure(batched_train_dataset.output_types,
                                                        batched_train_dataset.output_shapes)
        

        self.images, self.labels = self.iterator.get_next()        
        self.train_init_op = self.iterator.make_initializer(batched_train_dataset)
        
        self.define_model(self.images, n_classes)
        self.create_optimizer()
        
        # Evaluation metrics
        prediction = tf.to_int32(tf.argmax(self.logits, 1))
        self.correct_prediction = tf.equal(prediction, self.labels)
        accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
    
        # Launch the session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.init_fn(self.sess)  # load the pretrained weights
        self.sess.run(self.fc8_init)  # initialize the new fc8 layer


    def define_model(self, images, n_classes):
        '''
        # ---------------------------------------------------------------------
        # For this example, we'll use VGG-16 pretrained on ImageNet. We will remove the
        # last fully connected layer (fc8) and replace it with our own, with an
        # output size n_classes.
        # We will first train the last layer for a few epochs.
        # Then we will train the entire model on our dataset for a few epochs.
        # Get the pretrained model, specifying the num_classes argument to create a new
        # fully connected replacing the last one, called "vgg_16/fc8"
        # Each model has a different architecture, so "vgg_16/fc8" will change in another model.
        # Here, logits gives us directly the predicted scores we wanted from the images.
        # We pass a scope to initialize "vgg_16/fc8" weights with he_initializer
        '''
        self.is_training = tf.placeholder(tf.bool)
        vgg = tf.contrib.slim.nets.vgg
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=args.weight_decay)):
            self.logits, self.end_points = vgg.vgg_16(self.images, num_classes=n_classes, is_training=self.is_training,
                               dropout_keep_prob=args.dropout_keep_prob)
            
            
        # Specify where the model checkpoint is (pretrained weights).
        model_path = args.model_path
        assert(os.path.isfile(model_path))

        # Restore only the layers up to fc7 (included)
        # Calling function `init_fn(sess)` will load all the pretrained weights.
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['vgg_16/fc8'])
        self.init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)

        # Initialization operation from scratch for the new "fc8" layers
        # `get_variables` will only return the variables whose name starts with the given pattern
        self.fc8_variables = tf.contrib.framework.get_variables('vgg_16/fc8')
        self.fc8_init = tf.variables_initializer(self.fc8_variables)

        # ---------------------------------------------------------------------
        # Using tf.losses, any loss is added to the tf.GraphKeys.LOSSES collection
        # We can then call the total loss easily
        tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.logits)
        self.loss = tf.losses.get_total_loss()

    def create_optimizer(self):
        # First we want to train only the reinitialized last layer fc8 for a few epochs.
        # We run minimize the loss only with respect to the fc8 variables (weight and bias).
        self.fc8_optimizer = tf.train.GradientDescentOptimizer(args.learning_rate1)
        self.fc8_train_op = self.fc8_optimizer.minimize(self.loss, var_list=self.fc8_variables)

        # Then we want to finetune the entire model for a few epochs.
        # We run minimize the loss only with respect to all the variables.
        self.full_optimizer = tf.train.GradientDescentOptimizer(args.learning_rate2)
        self.full_train_op = self.full_optimizer.minimize(self.loss)

    def train_last_layer(self, n_epochs):
        for epoch in range(n_epochs):
            # Run an epoch over the training data.
            print('Starting epoch %d / %d' % (epoch + 1, n_epochs))
            # Here we initialize the iterator with the training set.
            # This means that we can go through an entire epoch until the iterator becomes empty.            
            self.sess.run(self.train_init_op)            
            while True:
                try:
                    _ = self.sess.run(self.fc8_train_op, {self.is_training: True})
                except tf.errors.OutOfRangeError:
                    break
            # Check accuracy on the train sets every epoch.
            train_acc = check_accuracy(self.sess, self.correct_prediction, self.is_training, self.train_init_op)
            print('Train accuracy: %f' % train_acc)

    def train_all_layers(self, n_epochs):
        for epoch in range(n_epochs):
            print('Starting epoch %d / %d' % (epoch + 1, n_epochs))
            self.sess.run(self.train_init_op)
            while True:
                try:
                    _ = self.sess.run(self.full_train_op, {self.is_training: True})
                except tf.errors.OutOfRangeError:
                    break
            # Check accuracy on the train and val sets every epoch
            train_acc = check_accuracy(self.sess, self.correct_prediction, self.is_training, self.train_init_op)
            print('Train accuracy: %f' % train_acc)


# In[10]:


class Args():
    def __init__(self):
        pass


# In[11]:


top_data_dir()


# In[13]:


args = Args()
args.train_dir = osp.join(top_data_dir(), 'Images/Shape_Net_Core/2015_Summer_OUT/Rendered_Images/no_texture_blender/cropped')
args.model_path = osp.join(top_data_dir(), 'NN/vgg16_pretrained/vgg_16.ckpt')
args.batch_size = 92
args.num_workers = 10
args.learning_rate1 = 1e-3
args.learning_rate2 = 1e-5
args.weight_decay = 5e-4
args.dropout_keep_prob = 0.5
tr_perc = 0.95
random_seed = 42
last_layer_epochs = 15
all_layers_epochs = 15


# In[23]:


train_filenames, train_labels = list_images(args.train_dir)
n_data = len(train_filenames)
train_thr = int(n_data * tr_perc)
np.random.seed(random_seed)
perm = np.random.permutation(n_data)

train_labels = np.array(train_labels)[perm]
val_labels = train_labels[train_thr:]
train_labels = train_labels[:train_thr]
train_labels = train_labels.tolist()
val_labels = val_labels.tolist()

train_filenames = np.array(train_filenames)[perm]
val_filenames = train_filenames[train_thr:]
train_filenames = train_filenames[:train_thr]
train_filenames = train_filenames.tolist()
val_filenames = val_filenames.tolist()


# In[24]:


reset_tf_graph()
vgg_ft = VGG_Finetuner(args, train_filenames, train_labels)


# In[27]:


save_net_dir  = osp.join(top_data_dir(), 'OUT/3d_neighbs_as_context/nn_models/vgg_based_nets')
net_model_name = 'cropped_imgs_95prc_training_15_15_epochs'
out_dir = create_dir(osp.join(save_net_dir, net_model_name))


# In[28]:


restore_model = True
restore_epoch = 30
if restore_model:
    checkpoint_path = osp.join(out_dir, MODEL_SAVER_ID + '-' + str(int(restore_epoch)))
    vgg_ft.saver.restore(vgg_ft.sess, checkpoint_path)


# In[29]:


save_model = False
do_training = False
save_epoch = 30
if do_training:    
    batched_val_dataset = make_dataset(val_filenames, val_labels, val_preprocess)
    val_iterator = tf.contrib.data.Iterator.from_structure(batched_val_dataset.output_types, batched_val_dataset.output_shapes)
    val_init_op = vgg_ft.iterator.make_initializer(batched_val_dataset)    
    
    for i in range(last_layer_epochs):
        vgg_ft.train_last_layer(1)
        print check_accuracy(vgg_ft.sess, vgg_ft.correct_prediction, vgg_ft.is_training, val_init_op)

    for i in range(all_layers_epochs):
        vgg_ft.train_all_layers(1)
        print check_accuracy(vgg_ft.sess, vgg_ft.correct_prediction, vgg_ft.is_training, val_init_op)
        
    if save_model:
        checkpoint_path = osp.join(out_dir, MODEL_SAVER_ID)
        vgg_ft.saver.save(vgg_ft.sess, checkpoint_path, global_step=save_epoch)


# In[30]:


class_to_int = {'chair':3, 'table': 5, 'lamp': 1, 'sofa': 0, 'car': 6, 'airplane': 4, 'vessel': 7, 'rifle': 2}


# In[13]:


# Extract features for class (chair = 3, table = 5)
class_type = 'airplane'
class_label = class_to_int[class_type]
filenames, labels = list_images(args.train_dir)
filenames = np.array(filenames)
labels = np.array(labels)
class_idx = np.where(np.array(labels) == class_label)[0]
filenames = filenames[class_idx]
labels = labels[class_idx]
filenames = filenames.tolist()
labels = labels.tolist()


# In[36]:


check_acc = True
if check_acc:    
    batched_val_dataset = make_dataset(filenames, labels, val_preprocess, shuffle=False)
    val_iterator = tf.contrib.data.Iterator.from_structure(batched_val_dataset.output_types,
                                                           batched_val_dataset.output_shapes)        
    val_init_op = vgg_ft.iterator.make_initializer(batched_val_dataset)
    print check_accuracy(vgg_ft.sess, vgg_ft.correct_prediction, vgg_ft.is_training, val_init_op)


# In[37]:


save_feats = True
extract_feats = True
conv_feats = False

if extract_feats:
    features = extract_features(vgg_ft, filenames, labels, conv=conv_feats)
    feat_dict = dict()
    for i, f in enumerate(filenames):
        tokens = f.split('/')
        model_name = tokens[-2]
        syn_id = tokens[-3]
        feat_dict[syn_id + '_' + model_name] = features[i]


if save_feats:
    if conv_feats:
        feat_tag = 'conv5'
    else:
        feat_tag = 'fc7'
    
    save_dir = '/orions4-zfs/projects/optas/DATA/OUT/3d_neighbs_as_context/vgg_feats/'
    pickle_data(osp.join(save_dir, class_type + '_' + feat_tag + '_' + net_model_name + '.pkl'), feat_dict)


# In[32]:


# working with images of real-chairs.
from general_tools.in_out.basics import files_in_subdirs
REAL_CHANNELS = 3
top_real_dir  = osp.join(top_data_dir(), 'Images/real_chairs/scaled')
filenames = [f for f in files_in_subdirs(top_real_dir, '.png')]
n_images = len(filenames)
print n_images
labels = [class_to_int['chair']] * n_images
features = extract_features(vgg_ft, filenames, labels, conv=False, real_data=True)


# In[34]:


feat_tag = 'fc7'
feat_dict = dict()
for i, f in enumerate(filenames):
    feat_dict[f] = features[i]
save_dir = create_dir(osp.join(top_data_dir(), 'OUT/3d_neighbs_as_context/vgg_feats/real_chairs_new/'))
pickle_data(osp.join(save_dir, 'scaled_and_RGB' + '_' + feat_tag + '_' + net_model_name + '.pkl'), feat_dict)    


# In[35]:


batched_val_dataset = make_dataset(filenames, labels, val_preprocess, shuffle=False, real_data=True)
val_iterator = tf.contrib.data.Iterator.from_structure(batched_val_dataset.output_types, batched_val_dataset.output_shapes)        
val_init_op = vgg_ft.iterator.make_initializer(batched_val_dataset)
print check_accuracy(vgg_ft.sess, vgg_ft.correct_prediction, vgg_ft.is_training, val_init_op)


# In[14]:


# Below: working with parts.


# In[23]:


# Extract features for Chair-part-data (vary sub-dir)
part_dir = '/orions4-zfs/projects/optas/DATA/Images/Shape_Net_Core/2015_Summer_OUT/Rendered_Images/no_texture_blender/with_part_info/03001627'
# sub_dir = 'individual_parts_cropped'
sub_dir = 'without_one_part_cropped'
part_files = [f for f in files_in_subdirs(part_dir, '.*' + sub_dir + '.*\.png')]
for f in part_files:
    assert(sub_dir in f)
print len(part_files)    


# In[24]:


class_label = class_to_int['chair']
part_labels = (np.ones(len(part_files), dtype=int)*class_label).tolist()
all_f7_parts = extract_features(vgg_ft, part_files, part_labels, conv=False, real_data=False)


# In[25]:


feat_dict = dict()
features = all_f7_parts
filenames = part_files
save_feats = True
for i, f in enumerate(filenames):
    tokens = f.split('/')
    model_name = tokens[-3]
    syn_id = tokens[-4]
    part_id = tokens[-1].replace('.png', '')
    if sub_dir == 'without_one_part_cropped':
        p0, p1 = part_id.split('_')
        part_id = p1 + '_' + p0
    feat_dict[syn_id + '_' + model_name + '_' + part_id] = features[i]

if save_feats:
    top_out_dir = '/orions4-zfs/projects/optas/DATA/OUT/3d_neighbs_as_context/vgg_feats/'
    if sub_dir == 'without_one_part_cropped':        
        out_dir = osp.join(top_out_dir, 'exluding_one_part')
    else:
        out_dir = osp.join(top_out_dir, 'singleton_parts')
    create_dir(out_dir)
    pickle_data(osp.join(out_dir, 'fc7_' + net_model_name + '.pkl'), feat_dict)


# In[36]:


# Below next two cells: working with Kalogerakis parts.
# Extract features for Table-part-data
part_dir = '/orions4-zfs/projects/optas/DATA/Images/part-annoted-from-kalogerakis/Table/'
sub_dir = 'individual_parts/cropped'
part_files = [f for f in files_in_subdirs(part_dir, '.*' + sub_dir + '.*\.png')]
for f in part_files:
    assert(sub_dir in f)
print len(part_files)    

class_label = 5
part_labels = (np.ones(len(part_files), dtype=int)*class_label).tolist()
all_f7_parts = extract_features(vgg_ft, part_files, part_labels, conv=False, real_data=False)


# In[47]:


feat_dict = dict()
features = all_f7_parts
filenames = part_files
save_feats = True
syn_id = snc_category_to_synth_id()['table']

for i, f in enumerate(part_files):
    tokens = f.split('/')
    model_name = tokens[-5]
    part_id = tokens[-1].replace('.png', '')
    feat_dict[syn_id + '_' + model_name + '_' + part_id] = features[i]

if save_feats:
    top_out_dir = '/orions4-zfs/projects/optas/DATA/OUT/3d_neighbs_as_context/vgg_feats'
    out_dir = osp.join(top_out_dir, 'singleton_parts/kalogerakis/table')
    create_dir(out_dir)
    pickle_data(osp.join(out_dir, 'fc7_' + net_model_name + '.pkl'), feat_dict)


# In[ ]:


# Visualizing gradient
from general_tools.simpletons import iterate_in_chunks
def extract_grads(vgg_net, filenames, labels):
    '''returns features in same order as filenames.
    '''
    batched_test_dataset = make_dataset(filenames, labels, val_preprocess, shuffle=False)
    test_init_op = vgg_net.iterator.make_initializer(batched_test_dataset)
    vgg_net.sess.run(test_init_op)        
    all_feats = []    
    batch_idx = [b for b in iterate_in_chunks(np.arange(len(filenames)), 
                                              args.batch_size)] #!
    
    i = 0    
    while True:
        try:            
            if i == len(batch_idx):
                i = len(batch_idx) - 1
                
            res = vgg_net.sess.run(g_out, {vgg_net.is_training: False,
                                           grad_from_above: F[batch_idx[i]] #! F: the pre-computed (listener) grads.
                                          })
            all_feats.append(np.squeeze(res))
            i += 1
        except tf.errors.OutOfRangeError:
            break
    all_feats = np.vstack(all_feats).astype(np.float32)
    return all_feats

vgg_net = vgg_ft
f7 = vgg_net.end_points['vgg_16/fc7']
f7_squeezed = tf.squeeze(f7, axis=[1, 2])
grad_from_above = tf.placeholder(tf.float32, [None, 4096])
g_out = tf.gradients(f7_squeezed, vgg_net.images, grad_ys=grad_from_above)

fnames = []
for s in sorted_sn_models: # use sorted input.
    fnames.append('/orions4-zfs/projects/optas/DATA/Images/Shape_Net_Core/2015_Summer_OUT/Rendered_Images/no_texture_blender/default_out/03001627/' + 
    s + '/image_p020_t337_r005.png')

all_grads = extract_grads(vgg_ft, fnames, labels)

np.savez('grads_from_vgg_on_listener_projected_embedding', all_grads=all_grads, all_model_ids='03001627_' + np.array(sorted_sn_models, object))

http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_grabcut/py_grabcut.html
http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture12.pdf
    
def visualize_preprocess(image, label):
    ''' Preprocessing (for validation)
    Take a central 224x224 crop to the scaled image
    Substract the per color mean `VGG_MEAN`
    # Note: we don't normalize the data here, as VGG was trained without normalization
    '''
    crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)    # (3)
    centered_image = crop_image
    return centered_image, label

g_vals = []
for _ in grads_freezed:    
    g_part = grads_freezed
    g_op_listener = tf.gradients(f7_squeezed, vgg_net.images, grad_ys=g_part)
# #     g_op_listener = tf.gradients(f7_squeezed, vgg_net.images)    
    g_val = vgg_net.sess.run(g_op_listener, {vgg_net.is_training: False,
                                             vgg_net.images: feed_im})[0][0]
    
    g_val = np.max(np.abs(g_val), axis=2)
#     g_val[g_val < np.percentile(g_val, 50)] = 0
    g_vals.append(g_val)


import cv2
img = image

mask = np.zeros(img.shape[:2], np.uint8)

mask[g_vals[0] > pfg] = cv2.GC_PR_FGD
mask[g_vals[0] < bg] = 0
mask[g_vals[0] > fg] = 1

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

# rect = (1, 1, 224, 224)
# cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
cv2.grabCut(img, mask, None, bgdModel, fgdModel, 20, cv2.GC_INIT_WITH_MASK)

