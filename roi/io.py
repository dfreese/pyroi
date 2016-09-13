#!/usr/bin/env python
import json
from roi import (RectROI, CylROI, SphereROI)

def json_to_roi(string):
    '''
    Decodes JSON entry for an ROI and returns the proper type initialized with
    the given specs.  Assumes first that string is a json entry.  If that fails
    it then assumes it names a JSON file with the json entry.  Raises a value
    error if both fail.  Assumes the JSON file has the following entries:
        - 'type'
        - 'center'

    Type can be the following values:
        - 'rectangle', requires 'size' entry (3,) array
        - 'cylinder', requires 'radius' and 'height' entries as scalars
        - 'sphere', requires 'radius' entry as a scalar
    '''
    try:
        # Assume it's a string first
        config = json.loads(string)
    except:
        # If that fails, treat it like a file
        try:
            with open(string, 'r') as fid:
                config = json.load(fid)
        except:
            raise ValueError('JSON string or filename not valid')

    if 'type' not in config:
        raise KeyError('key specifying type of ROI was not specified')

    if 'center' not in config:
        raise KeyError('key specifying center of ROI was not specified')

    if config['type'] == 'rectangle':
        if 'size' not in config:
            raise KeyError('key specifying size of ROI was not specified')
        return RectROI(config['size'], config['center'])
    elif config['type'] == 'cylinder':
        if 'radius' not in config:
            raise KeyError('key specifying radius of ROI was not specified')
        if 'height' not in config:
            raise KeyError('key specifying height of ROI was not specified')
        return CylROI(config['radius'], config['height'], config['center'])
    elif config['type'] == 'sphere':
        if 'radius' not in config:
            raise KeyError('key specifying radius of ROI was not specified')
        return SphereROI(config['radius'], config['center'])
    else:
        raise ValueError('ROI type, "%s" not recognized' % config['type'])
