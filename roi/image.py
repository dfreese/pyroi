#!/usr/bin/env python

import numpy as np

class Image:
    '''
    Base Image class
    '''
    def __init__(self, fov, data):
        '''
        Creates an image by calling set_fov and set_data from the given input.

        Parameters
        ----------
        fov : array_like, shape = (3,)
            The FOV size, in (X, Y, Z) technically dimension-less, as long as
            the units match those used by an ROI.
        data : array_like, shape = (n,m,o)
            The data for the image to be initialized with.
        '''
        self.set_fov(fov)
        self.set_data(data)

    def set_data(self, data):
        '''
        Used by init to set the voxel values of the image.  This can also be
        used to change the data after the image has been initialized.  This
        does not modify the FOV size.  The voxel dimension size is adjusted
        accordingly.

        Parameters
        ----------
        data : array_like, shape = (n,m,o)
            The data for the image to be initialized with.
        '''
        if data is None:
            raise ValueError('Image not provided')
        data = np.atleast_3d(np.asfarray(data))
        if data.ndim != 3:
            raise ValueError('Only 3 dimensional images are supported')
        self.data = data
        self.vsize = np.array(data.shape)
        self.init_grid()

    def set_fov(self, fov):
        '''
        Sets the size of the FOV as a (3,) float numpy array.

        Parameters
        ----------
        fov : array_like, shape = (3,)
            The FOV size, in (X, Y, Z) technically dimension-less, as long as
            the units match those used by an ROI.
        '''
        self.fov = np.asarray(fov).squeeze()
        if self.fov.shape != (3,):
            raise ValueError('FOV provided not the correct size')
        # Use self.x as an indicator that self.init_grid() has already been
        # called by self.set_data(), so that we should call this again.  This
        # causes self.init_grid() to not be called twice during __init__().
        if hasattr(self, 'x'):
            self.init_grid()

    def init_grid(self):
        '''
        creates self.x, y, z, X, Y, and Z, which are 1D and 3D representations
        of the center of the voxels based of of self.fov and self.vsize.
        '''
        self.x = np.linspace(0, self.fov[0], self.vsize[0], endpoint=False)
        self.y = np.linspace(0, self.fov[1], self.vsize[1], endpoint=False)
        self.z = np.linspace(0, self.fov[2], self.vsize[2], endpoint=False)
        self.x -= self.x.mean()
        self.y -= self.y.mean()
        self.z -= self.z.mean()
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z,
                                             indexing='ij')
