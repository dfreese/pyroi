import roi
import numpy as np

def test_io_cyl():
    cyl_roi_base = roi.CylROI(13, 30, (1, 8, 0))
    cyl_roi_read = roi.json_to_roi('./test/cyl_roi.json')
    assert(cyl_roi_base == cyl_roi_read)

def test_io_rect():
    roi_base = roi.RectROI((34, 52, 10), (0, 0, 0))
    roi_read = roi.json_to_roi('./test/rect_roi.json')
    assert(roi_base == roi_read)

def test_io_sphere():
    roi_base = roi.SphereROI(1, (0, 0, 0))
    roi_read = roi.json_to_roi('./test/sphere_roi.json')
    assert(roi_base == roi_read)

def test_sphere():
    sph_roi = roi.SphereROI(0.434, (0, 0, 0))
    image_vsize = (320, 168, 30)
    image_fov = [x / 2.0 for x in image_vsize]
    img = roi.Image(image_fov, np.ones(image_vsize))
    assert(sph_roi.sum(img) == 8.)
    assert(sph_roi.mean(img) == 1.)
    assert(sph_roi.std(img) == 0.)

def test_sum_rect():
    rect_roi = roi.RectROI((1.0, 1.0, 1.0), (0, 0, 0))
    image_vsize = (320, 168, 30)
    image_fov = [x / 2.0 for x in image_vsize]
    img = roi.Image(image_fov, np.ones(image_vsize))
    assert(rect_roi.sum(img) == 8.)
    assert(rect_roi.mean(img) == 1.)
    assert(rect_roi.median(img) == 1.)

def test_cylinder():
    cyl_roi = roi.CylROI(0.354, 0.5, (0, 0, 0))
    image_vsize = (320, 168, 30)
    image_fov = [x / 2.0 for x in image_vsize]
    img = roi.Image(image_fov, np.ones(image_vsize))
    assert(cyl_roi.sum(img) == 8.)
    assert(cyl_roi.mean(img) == 1.)
    assert(cyl_roi.std(img) == 0.)
