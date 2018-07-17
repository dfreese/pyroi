#!/usr/bin/env python

import numpy as np
from image import Image

class ROI:
    '''
    Base ROI class
    '''
    def __init__(self):
        self._clear_cache()

    def get_mask(self, image):
        '''
        The main public function that checks the mask_cache first before calling
        the ROIs _get_mask function.  The _get_mask function should be
        overridden by the subclasses of ROIs.  The cache uses Image.get_key()
        to get key uniquely identifying that images mesh.

        Parameters
        ----------
        image : roi.Image
            The image for which the ROI should generate the mask.

        Returns
        -------
        res : numpy.ndarray
            A 3d ndarray of floats that indicating the contribution of that
            voxel to the ROI.
        '''
        key = image.get_key()
        if key not in self.mask_cache:
            self.mask_cache[key] = self._get_mask(image)
        return self.mask_cache[key]

    def _clear_cache(self):
        '''
        To be called if an property of the roi is modified such that the cache
        is no longer valid.
        '''
        self.mask_cache = dict()

    def _get_mask(self, image):
        '''
        A base function to be overridden by derived classes.  The _get_mask
        function for all derived classes shall return the mask of the ROI for
        that image, based on it's mesh coordinates (i.e. image.X, image.Y, and
        image.Z).

        Calling this on the base class treats the entire image as the ROI by
        returning a mask of ones the same size as the image.

        Parameters
        ----------
        image : roi.Image
            The image for which the ROI should generate the mask.

        Returns
        -------
        res : numpy.ndarray
            A 3d ndarray of floats that indicating the contribution of that
            voxel to the ROI.
        '''
        if not isinstance(image, Image):
            raise TypeError('image is not an Image class')
        return np.ones(image.shape)

    def max(self, image):
        '''
        Finds the maximum value of the ROI.  The weights of the mask are not
        considered, and any non-negative value in the mask is used.

        Parameters
        ----------
        image : roi.Image
            The image for which the ROI calculate the value.

        Returns
        -------
        res : numpy.scalar
            A numpy scalar indicating the statistic requested
        '''
        return image.data[self.get_mask(image) > 0].max()

    def min(self, image):
        '''
        Finds the minimum value of the ROI.  The weights of the mask are not
        considered, and any non-negative value in the mask is used.

        Parameters
        ----------
        image : roi.Image
            The image for which the ROI calculate the value.

        Returns
        -------
        res : numpy.scalar
            A numpy scalar indicating the statistic requested
        '''
        return image.data[self.get_mask(image) > 0].min()

    def median(self, image):
        '''
        Finds the median value of the ROI.  The weights of the mask are not
        considered, and any non-negative value in the mask is used.

        Parameters
        ----------
        image : roi.Image
            The image for which the ROI calculate the value.

        Returns
        -------
        res : numpy.scalar
            A numpy scalar indicating the statistic requested
        '''
        return np.median(image.data[self.get_mask(image) > 0])

    def mean(self, image):
        '''
        Calculates a weighted average of the ROI calling numpy.average on
        the image weighted by the mask.

        Parameters
        ----------
        image : roi.Image
            The image for which the ROI calculate the value.

        Returns
        -------
        res : numpy.scalar
            A numpy scalar indicating the statistic requested
        '''
        return np.average(image.data, weights=self.get_mask(image))

    def var(self, image):
        '''
        Calculates a weighted variance of the ROI calling numpy.average on
        (I - I.mean())**2 weighted by the mask.

        Parameters
        ----------
        image : roi.Image
            The image for which the ROI calculate the value.

        Returns
        -------
        res : numpy.scalar
            A numpy scalar indicating the statistic requested
        '''
        return np.average((image.data - self.mean(image)) ** 2,
                         weights=self.get_mask(image))

    def std(self, image):
        '''
        Calculates a weighted standard deviation of the ROI by calling var()

        Parameters
        ----------
        image : roi.Image
            The image for which the ROI calculate the value.

        Returns
        -------
        res : numpy.scalar
            A numpy scalar indicating the statistic requested
        '''
        return np.sqrt(self.var(image))

    def sum(self, image):
        '''
        Returns a weighted sum of the mask times the image.

        Parameters
        ----------
        image : roi.Image
            The image for which the ROI calculate the value.

        Returns
        -------
        res : numpy.scalar
            A numpy scalar indicating the statistic requested
        '''
        return (image.data * self.get_mask(image)).sum()

    def int_uniformity(self, image):
        '''
        Returns the integral uniformity for all voxel values with a non-zero
        weight.

        Parameters
        ----------
        image : roi.Image
            The image for which the ROI calculate the value.

        Returns
        -------
        res : numpy.scalar
            A numpy scalar indicating the statistic requested
        '''
        minimum = self.min()
        maximum = self.max()
        return (maximum - minimum) / (maximum + minimum)

    def __eq__(self, other):
        '''
        Class instances are considered equal.  This function should be
        overridden by subclasses with more specific properties.
        '''
        if isinstance(other, self.__class__):
            return True
        return NotImplemented

    def __ne__(self, other):
        '''
        Merely not the __eq__ function.  This should be consistent across all
        subclasses.
        '''
        return not self.__eq__(other)

class RectROI(ROI):
    '''
    Rectangular ROI subclass of ROI
    '''
    def __init__(self, size, center):
        '''
        Creates a Rectangular ROI with a given size.  Units are
        dimension-less as long as they match that of the image to be calculated
        on.

        Voxels with a center less than size / 2 units away from the roi center
        in x, y, and z are consiered as part of the ROI.  No consideration is
        given to partial voxels.

        Rectangle is currently assumed to be oriented in the same coordinate
        system as the image.

        Parameters
        ----------
        size : array_like, shape = (3,)
            The size of the sphere in (X, Y, Z) technically dimension-less,
            as long as the units match those used by an image.
        center : array_like, shape = (3,)
            The center of the sphere in (X, Y, Z) technically dimension-less,
            as long as the units match those used by an image.
        '''
        ROI.__init__(self)
        self.set_size(size)
        self.set_center(center)

    def set_size(self, size):
        '''
        Sets or changes the size of the ROI

        Parameters
        ----------
        size : array_like, shape = (3,)
            The size of the sphere in (X, Y, Z) technically dimension-less,
            as long as the units match those used by an image.
        '''
        self.size = np.asfarray(size).squeeze()
        if self.size.shape != (3,):
            raise ValueError('Shape of ROI size provided not (3,)')
        self._clear_cache()

    def set_center(self, center):
        '''
        Sets or changes the center of the roi

        Parameters
        ----------
        center : array_like, shape = (3,)
            The center of the sphere in (X, Y, Z) technically dimension-less,
            as long as the units match those used by an image.
        '''
        self.center = np.asfarray(center).squeeze()
        if self.center.shape != (3,):
            raise ValueError('Shape of ROI center provided not (3,)')
        self._clear_cache()

    def _get_mask(self, image):
        '''
        Creates a mask for the given image. Voxels with a center less than
        size / 2 units away from the roi center in x, y, and z are consiered as
        part of the ROI.  No consideration is given to partial voxels.

        Parameters
        ----------
        image : roi.Image
            The image for which the ROI should generate the mask.

        Returns
        -------
        res : numpy.ndarray
            A 3d ndarray of floats that indicating the contribution of that
            voxel to the ROI.
        '''
        if not isinstance(image, Image):
            raise TypeError('image is not an Image class')
        return ((np.abs(image.X - self.center[0]) <= self.size[0] / 2.0) &
                (np.abs(image.Y - self.center[1]) <= self.size[1] / 2.0) &
                (np.abs(image.Z - self.center[2]) <= self.size[2] / 2.0)
                ).astype(float)

    def __eq__(self, other):
        '''
        Classes are considered equal if the center and size are equal.
        '''
        if isinstance(other, self.__class__):
            return ((other.size == self.size).all() and
                    (other.center == self.center).all())
        return NotImplemented

class CylROI(ROI):
    '''
    Cylindrical ROI subclass of ROI
    '''
    def __init__(self, radius, height, center):
        '''
        Creates a Cylindrical ROI with a given radius.  Units are
        dimension-less as long as they match that of the image to be calculated
        on.

        Voxels with a center less than radius units away from the roi center
        in x and y, and less than height / 2 in z will be consiered as part of
        the ROI.  No consideration is given to partial voxels.

        Cylinder is currently assumed to be oriented in Z with circle in x and
        y.

        Parameters
        ----------
        radius : numpy.scalar like
            The radius of the cylinder
        height : numpy.scalar like
            The height of the cylinder
        center : array_like, shape = (3,)
            The center of the sphere in (X, Y, Z) technically dimension-less,
            as long as the units match those used by an image.
        '''
        ROI.__init__(self)
        self.set_radius(radius)
        self.set_height(height)
        self.set_center(center)

    def set_radius(self, radius):
        '''
        Sets or changes the radius of the cylinder.

        Parameters
        ----------
        radius : numpy.scalar like
            The radius of the cylinder
        '''
        self.radius = np.float64(radius)
        if self.radius < 0:
            raise ValueError('Negative radius provided')
        self._clear_cache()

    def set_height(self, height):
        '''
        Sets or changes the height of the cylinder.

        Parameters
        ----------
        height : numpy.scalar like
            The height of the cylinder
        '''
        self.height = np.float64(height)
        if self.height < 0:
            raise ValueError('Negative height provided')
        self._clear_cache()

    def set_center(self, center):
        '''
        Sets or changes the center of the cylinder.

        Parameters
        ----------
        center : array_like, shape = (3,)
            The center of the sphere in (X, Y, Z) technically dimension-less,
            as long as the units match those used by an image.
        '''
        self.center = np.asfarray(center).squeeze()
        if self.center.shape != (3,):
            raise ValueError('Shape of ROI center provided not (3,)')
        self._clear_cache()

    def _get_mask(self, image):
        '''
        Creates a mask for the given image. Voxels with a center less than
        radius units away from the roi center in x and y, and less than
        height / 2 in z will be consiered as part of the ROI.  No consideration
        is given to partial voxels.

        Parameters
        ----------
        image : roi.Image
            The image for which the ROI should generate the mask.

        Returns
        -------
        res : numpy.ndarray
            A 3d ndarray of floats that indicating the contribution of that
            voxel to the ROI.
        '''
        if not isinstance(image, Image):
            raise TypeError('image is not an Image class')
        # TODO: Implement orientation, currently assuming height goes in Z
        return ((np.sqrt((image.X - self.center[0]) ** 2 +
                         (image.Y - self.center[1]) ** 2) <= self.radius) &
                (np.abs(image.Z - self.center[2]) <= self.height / 2.0)
                ).astype(float)

    def __eq__(self, other):
        '''
        Classes are considered equal if the center, height, and radius are
        equal.
        '''
        if isinstance(other, self.__class__):
            return ((other.radius == self.radius) and
                    (other.height == self.height) and
                    (other.center == self.center).all())
        return NotImplemented

class SphereROI(ROI):
    '''
    Spherical ROI subclass of ROI
    '''
    def __init__(self, radius, center):
        '''
        Creates a spherical ROI with a given radius.  Units are dimension-less
        as long as they match that of the image to be calculated on.

        Voxels with a center less than radius units away from the roi center
        will be consiered as part of the ROI.  No consideration is given to
        partial voxels.

        Parameters
        ----------
        radius : numpy.scalar like
            The radius of the spherical roi
        center : array_like, shape = (3,)
            The center of the sphere in (X, Y, Z) technically dimension-less,
            as long as the units match those used by an image.
        '''
        ROI.__init__(self)
        self.set_radius(radius)
        self.set_center(center)

    def set_radius(self, radius):
        '''
        Sets or changes the radius of the ROI.

        Parameters
        ----------
        radius : numpy.scalar like
            The radius of the spherical ROI
        '''
        self.radius = np.float64(radius)
        if self.radius < 0:
            raise ValueError('Negative radius provided')
        self._clear_cache()

    def set_center(self, center):
        '''
        Sets or changes the center of the ROI.

        Parameters
        ----------
        center : array_like, shape = (3,)
            The center of the sphere in (X, Y, Z) technically dimension-less,
            as long as the units match those used by an image.
        '''
        self.center = np.asfarray(center).squeeze()
        if self.center.shape != (3,):
            raise ValueError('Shape of ROI center provided not (3,)')
        self._clear_cache()

    def _get_mask(self, image):
        '''
        Creates a mask for the given image. Voxels with a center less than
        radius units away from the roi center are  consiered as part of the ROI.
        Currently, no consideration is given to partial voxels.

        Parameters
        ----------
        image : roi.Image
            The image for which the ROI should generate the mask.

        Returns
        -------
        res : numpy.ndarray
            A 3d ndarray of floats that indicating the contribution of that
            voxel to the ROI.
        '''
        if not isinstance(image, Image):
            raise TypeError('image is not an Image class')
        # TODO: Implement orientation, currently assuming height goes in Z
        return (np.sqrt((image.X - self.center[0]) ** 2 +
                        (image.Y - self.center[1]) ** 2 +
                        (image.Z - self.center[2]) ** 2) <= self.radius
                ).astype(float)

    def __eq__(self, other):
        '''
        Classes are considered equal if the center and radius are equal.
        '''
        if isinstance(other, self.__class__):
            return ((other.radius == self.radius) and
                    (other.center == self.center).all())
