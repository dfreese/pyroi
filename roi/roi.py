#!/usr/bin/env python

import numpy as np
from image import Image

class ROI:
    '''
    Base ROI class
    '''
    def __init__(self):
        pass

    def get_mask(image):
        '''
        A base function to be overridden by derived classes.  The get_mask
        function for all derived classes shall return the mask of the ROI for
        that image, based on it's mesh coordinates.

        Calling this on the base class treats the entire image as the ROI by
        returning a mask of ones the same size as the image.
        '''
        if not isinstance(image, Image):
            raise TypeError('image is not an Image class')
        return np.ones(image.shape)

    def max(self, image):
        '''
        Finds the maximum value of the ROI.  The weights of the mask are not
        considered, and any non-negative value in the mask is used.
        '''
        return image.data[self.get_mask(image) > 0].max()

    def min(self, image):
        '''
        Finds the minimum value of the ROI.  The weights of the mask are not
        considered, and any non-negative value in the mask is used.
        '''
        return image.data[self.get_mask(image) > 0].min()

    def median(self, image):
        '''
        Finds the median value of the ROI.  The weights of the mask are not
        considered, and any non-negative value in the mask is used.
        '''
        return np.median(image.data[self.get_mask(image) > 0])

    def mean(self, image):
        '''
        Calculates a weighted average of the ROI calling numpy.average on
        the image weighted by the mask.
        '''
        return np.average(image.data, weights=self.get_mask(image))

    def var(self, image):
        '''
        Calculates a weighted variance of the ROI calling numpy.average on
        (I - I.mean())**2 weighted by the mask.
        '''
        return np.average((image.data - self.mean(image)) ** 2,
                         weights=self.get_mask(image))

    def std(self, image):
        '''
        Calculates a weighted standard deviation of the ROI by calling var()
        '''
        return np.sqrt(self.var(image))

    def sum(self, image):
        '''
        Returns a weighted sum of the mask times the image.
        '''
        return (image.data * self.get_mask(image)).sum()


class RectROI(ROI):
    '''
    Rectangular ROI class
    '''
    def __init__(self, size, center):
        ROI.__init__(self)
        self.set_size(size)
        self.set_center(center)

    def set_size(self, size):
        self.size = np.asfarray(size).squeeze()
        if self.size.shape != (3,):
            raise ValueError('Shape of ROI size provided not (3,)')

    def set_center(self, center):
        self.center = np.asfarray(center).squeeze()
        if self.center.shape != (3,):
            raise ValueError('Shape of ROI center provided not (3,)')

    def get_mask(self, image):
        if not isinstance(image, Image):
            raise TypeError('image is not an Image class')
        return ((np.abs(image.X - self.center[0]) <= self.size[0] / 2.0) &
                (np.abs(image.Y - self.center[1]) <= self.size[1] / 2.0) &
                (np.abs(image.Z - self.center[2]) <= self.size[2] / 2.0)
                ).astype(float)

class CylROI(ROI):
    '''
    Cylindrical ROI class
    '''
    def __init__(self, radius, height, center):
        ROI.__init__(self)
        self.set_radius(radius)
        self.set_height(height)
        self.set_center(center)

    def set_radius(self, radius):
        self.radius = np.float64(radius)
        if self.radius < 0:
            raise ValueError('Negative radius provided')

    def set_height(self, height):
        self.height = np.float64(height)
        if self.height < 0:
            raise ValueError('Negative height provided')

    def set_center(self, center):
        self.center = np.asfarray(center).squeeze()
        if self.center.shape != (3,):
            raise ValueError('Shape of ROI center provided not (3,)')

    def get_mask(self, image):
        if not isinstance(image, Image):
            raise TypeError('image is not an Image class')
        # TODO: Implement orientation, currently assuming height goes in Z
        return ((np.sqrt((image.X - self.center[0]) ** 2 +
                         (image.Y - self.center[1]) ** 2) <= self.radius) &
                (np.abs(image.Z - self.center[2]) <= self.height / 2.0)
                ).astype(float)


class SphereROI(ROI):
    '''
    Spherical ROI class
    '''
    def __init__(self, radius, center):
        ROI.__init__(self)
        self.set_radius(radius)
        self.set_center(center)

    def set_radius(self, radius):
        '''
        '''
        self.radius = np.float64(radius)
        if self.radius < 0:
            raise ValueError('Negative radius provided')

    def set_center(self, center):
        self.center = np.asfarray(center).squeeze()
        if self.center.shape != (3,):
            raise ValueError('Shape of ROI center provided not (3,)')

    def get_mask(self, image):
        if not isinstance(image, Image):
            raise TypeError('image is not an Image class')
        # TODO: Implement orientation, currently assuming height goes in Z
        return (np.sqrt((image.X - self.center[0]) ** 2 +
                        (image.Y - self.center[1]) ** 2 +
                        (image.Z - self.center[2]) ** 2) <= self.radius
                ).astype(float)
