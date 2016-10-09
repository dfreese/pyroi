# PyROI
PyROI is a barebones python ROI analysis module that can be used to do basic image analysis. Emphasis was put on quickly performing the same analysis across a broad set of images describing the same scene, such as iterations of a PET image.

## Usage
A system level install isn't supported currently.  To use, add the PyROI folder to your PYTHONPATH

Next specify an ROI, for example, a spherical ROI with a radius 3, centered at (0,0,0).  The units don't matter as long as your consistent.

```
import roi
sphere_roi = roi.SphereROI(3, (0,0,0))
```

Then load in an image with a certain size of FOV, in this case we'll use all ones

```
img = roi.Image((64,64,10), np.ones((128, 128, 20))
```

The combination of the two can then be used to calculate max, min, median, mean, var, std, and sum by doing the following

```
total = roi.sum(img)
```

## Storing ROIs

ROI properties can also be stored and loaded from a json file such as this:
```
{
    "type": "cylinder",
    "center": [1, 8, 0],
    "radius": 13,
    "height": 30
}
```

This is done by calling the following:
```
cyl_roi = roi.json_to_roi('cyl_roi.json')
```
