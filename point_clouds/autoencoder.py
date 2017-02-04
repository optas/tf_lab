'''
Created on February 2, 2017

@author: optas
'''


class AutoEncoder(object):
    '''
    '''

    def restore_model(self, model_path):
        self.saver.restore(self.sess, model_path)

    def partial_fit(self, X, GT=None):
        '''Train models based on mini-batch of input data.
        Returns cost of mini-batch.
        If the AE is de-noising the GT needs to be provided.
        '''
        if GT is not None:
            _, loss, recon = self.sess.run((self.optimizer, self.loss, self.x_reconstr), feed_dict={self.x: X, self.gt: GT})
        else:
            _, loss, recon = self.sess.run((self.optimizer, self.loss, self.x_reconstr), feed_dict={self.x: X})
        return loss, recon

    def transform(self, X):
        '''Transform data by mapping it into the latent space.'''
        return self.sess.run(self.z, feed_dict={self.x: X})

    def reconstruct(self, X):
        '''Use AE to reconstruct given data.'''
        return self.sess.run((self.x_reconstr, self.loss), feed_dict={self.x: X, self.gt: X})
