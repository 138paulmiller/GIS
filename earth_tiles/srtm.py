#!/usr/env python
'''
    Tool for Querying and reformatting tiles from GIS Datasets 

    
Example Query all tiles, normalize the range and generate corresponding color tiles for all elevations within Longitude Latiutude 34N 2W to 35N 1E
    Notice for Longitude that 0W is truly 0E ... W-1 E0 E1
    
    $ python srtm.py            \ 
        SRTM_HGTs/              \   # specify that dataset can found in the local dir SRTM_HGTs. If no such file exists, it will be downloaded from the given server and placed in this directory.
        SRTM1                   \   # specify that the dataset contains SRTM1 data files
        r16                     \   # the output tile format is .r16 
        --bbox 34 -2 35 1       \   # the area of interest is specified by the following bounding box
        --normalize 32768 65535 \   # normalize the height values between 32768 65535
        --colors                \   # generate corresponding color images

'''



TODO = \
'''
    
    TODO 
    - Alternative methods for extracting area of interest
    - Gaussian and Median Filters on entire raster grid
    - Create a utility to grab polygons, incomplete tiles
    - Prevent Output file format from normalizeing 
    - Allow for partial long,lat queries. Not jusr whole numbers
'''


OUTLINE =\
'''
    As of now, this tool queries raw hgt (NASA height format) tiles form SRTM1 datasets into a raster grid (a matrix of rasters). 
    Then processes the grid and exports the grid into indivual files whose names are formatted according to their grid position 
    
    Grid Coordinate system and its correspondance with Tile Longitude and Latitude
    
                  +y (N)
                    ^
                    | 
                    |
    -x (W) <--------.--------> +x (E)
                    |          
                    |
                    v 
                   -y (S)
    
    
    As of now, Output fileformat only export +x, and +y tiles. where x,range is the range of latitude, and y is the range for the longitude.
    
    
'''


'''
UE4 Tiled Landscape Resolutions
8128
4033
2017
1009
505
253
127

For Unreal Landscape XY Scalar use 40075036.0/360/RES*100 where RES is any of the above resolutions 


'''

# All imports are at the top level. As they are used throughout
import sys, os 
import urllib
import argparse
import skimage

import numpy as np 
import matplotlib as mpl
import numpy as np

from PIL import Image


#'globals'
args = None
input_raster_res = { 'srtm1': 3601, 'srtm3': 1201}
supported_dataset_types = ['srtm1', 'srtm3' ]
supported_output_formats = ['r16' ] 
possible_subdirs = []

'''
Directory structure 
 SRTM_GL1/
    SRTM_GL1_srtm/
        North/
            North_0_29/
                N30E006.hgt
                ...
            North_30_60/
                N30E000.hgt
                ... 
        South/
            S01E006.hgt
            ...

 SRTM_GL3 is same as SRTM_GL1 just replace 1 with 3
  
'''
srtm_server="https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/"


## ------------------------ File System  Utils 
def get_ext(file):
    return file[file.rfind('.'):]
    
def walkfiles(dir, callback):
    for root, subdirs, files in os.walk(dir):  
        for file in files:
            callback(os.path.join(root, file))
            
def walkdirs(dir, callback):
    for root, subdirs, files in os.walk(dir):  
        for subdir in subdirs:
            callback(os.path.join(root, subdir).replace('\\', '/'))


#def get_file_from_url(url):
#    webfile = urllib.urlopen(url)
#    data = webfile.read()
#    webfile.close()
            
## ------------------------ Command line Utils 
def parse_cmd_args():
     
    
    global args 
    global supported_dataset_types 
    global supported_output_formats 
    global possible_subdirs
    global input_raster_res

    parser = argparse.ArgumentParser(description='GIS Util')
    
    parser.add_argument('dataset_path', metavar='Path', nargs=1, 
                        help='Path to dataset')
                        
    parser.add_argument('dataset_type', metavar='Format', nargs=1, 
                        help='''
                            SRTM1 : Use the SRTM GL1 dataset 30 meter (1 Arc Second)
                            SRTM3 : Use the SRTM GL3 dataset 90 meter (3 Arc second) 
                            ''')
    parser.add_argument('format', metavar='format', nargs=1,
                        help='r16: Format for World Machine and Unreal Engine 4 Heightmaps')
    #-----Optional
    parser.add_argument('-b', '--bbox', metavar='Lat Long', type=int, nargs=4,
                        help='Latitude Longitude  coordinates for Bounding box query Order from Minimum point to Maximum point')
    parser.add_argument('-r', '--tile_res', metavar='Resolution', type=int, nargs=1, default=1,
                        help='Target Resolution of raster for each tile ')
    
    parser.add_argument('-s', '--sigma', metavar='Sigma', type=float, nargs=1, default=0,
                        help='Sigma value for gaussian smoothing of tiles')
                        
    parser.add_argument('-n', '--normalize', metavar='Min Max', type=int, nargs=2, default=0,
                        help='Normalize output raster values within the range [Min-Max]')
   
    parser.add_argument('-p', '--tileprefix', metavar='Tile Prefix', nargs=1, default='Tiles/Tile',
                        help='Prefix for each tile filename {prefix}_x{x}_y{y}')
    
    parser.add_argument('-q', '--colorprefix', metavar='Color Prefix', nargs=1, default='Colors/Color',
                        help='Prefix for each color filename {prefix}_x{x}_y{y}')

                        
    # ---- Flags
    parser.add_argument('-c', '--colors',  action='store_true',
                        help='If set generate colors for ')
      
   
  
    args = parser.parse_args()
    # reformat args as datum not lists
    args.dataset_type = args.dataset_type[0].lower()
    args.format = args.format[0].lower()
    args.dataset_path = args.dataset_path[0]
    args.tile_res = args.tile_res[0] 
    args.sigma = args.sigma[0] 
    
    # check args
    assert( args.dataset_type in supported_dataset_types)
    assert( args.format in supported_output_formats)

    # set tile_res to be native tile, or argument
    if args.tile_res  == 0:
        args.tile_res = input_raster_res[args.dataset_type]
    # get information about folder structure
    possible_subdirs.append(args.dataset_path)
    walkdirs(args.dataset_path, lambda subdir : possible_subdirs.append(subdir) )
    
    print(args)
    return 

    
    
## ----------------------------- Getters 

def get_tile_filename(latlong, dataset_type):
    if dataset_type == 'srtm1':         # format [NS]Y[EW]XXX.hgt
        lat,long = latlong
        if lat >= 0:
            prefix_lat = 'N' 
        else: # if less then zero query southern tiles. remove sign
            lat *= -1
            prefix_lat = 'S'

        if long >= 0:
            prefix_long = 'E' 
        else: # if less then zero query westward tiles. remove sign
            long *= -1
            prefix_long = 'W'
            
        return f'{prefix_lat}{lat:02}{prefix_long}{long:03}.hgt'
    
 
def get_files_within_bbox(bbox):
    '''
    Returns a listing of all the filenames that contain values withing the given bbox.
    They are returned in row-major order
    '''

    # long lat == N/S E/W  
    if not bbox: return None
    min_lat, min_long, max_lat, max_long = bbox
    assert(min_long <= max_long)
    assert(min_lat <= max_lat)
    
    files = []
    for lat in range(min_lat, max_lat+1):
        for long in range(min_long, max_long+1):
            files.append(get_tile_filename(( lat,long ), args.dataset_type ))
    return files

def get_import_raster_res():
    global input_raster_res
    return input_raster_res [ args.dataset_type]

# grid class
class Grid(dict):
    def __init__(self, shape):
        self.shape =shape
        for y in range(shape[0]):
            self.__setitem__(y, {})
            for x in range(shape[1]):
                self.__getitem__(y).__setitem__(x, None)
            
    def __getitem__(self, key):
        return super(Grid, self).__getitem__(key)
    def __setitem__(self, key, value):
        return super(Grid, self).__setitem__(key, value)
    
def get_raster_grid(bbox):
    global args
    global possible_subdirs
    
    
    filenames = get_files_within_bbox(args.bbox)
    shape = bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1, args.tile_res,args.tile_res
    # use a dict, prevent memory aloloc error if using np array as block grid (longn perf hot)
    raster_grid = Grid(shape)
    filepath_grid = Grid(shape)

    
    raster_grid.shape = shape
    
    x= 0 
    y= 0        
    for filename in filenames:
        fullpath  = None
        for subdir in possible_subdirs:
            test_file = os.path.join(subdir,filename).replace('\\', '/')
            if os.path.isfile( test_file):   
                
                fullpath = test_file
                
                break 
        if  fullpath :
            filepath_grid[y][x] = fullpath
            raster =  np.array(import_raster(fullpath))
            raster_grid[y][x] = raster
        else:
            print(f'WARNING: Could not find file {filename}' )
        x+=1        
        if x >= shape[1]:
            x = 0;
            y += 1 

    return raster_grid, filepath_grid

def get_minmax(raster_grid):
    minz, maxz = sys.maxsize, -1*sys.maxsize
   
    for y in range(raster_grid.shape[0]):
        for x in range(raster_grid.shape[1]):
            raster = raster_grid[y][x]          
            if raster is None:
                continue
            minz = min(minz, np.amin(raster))
            maxz = max(maxz, np.amax(raster))
    return minz, maxz
   

'''
 SRTM_GL1/
    SRTM_GL1_srtm/
        North/
            North_0_29/
                N00E006.hgt
                ...
            North_30_60/
                N30E000.hgt
                ... 
        South/
            S01E006.hgt
            ...
'''
def get_url_from_server(in_filepath):
    global args
    global srtm_server
    filename = in_filepath[in_filepath.rfind('/'):]
    # expects [NS]yy[WE]xxx.hgt
    lon =  int(filename[1:3])
    lat =  int(filename[3:7])
    
    if args.dataset_type == 'srtm1':
        preamble = 'SRTM_GL1/SRTM_GL1_srtm/'
    elif args.dataset_type == 'srtm3':
        preamble = 'SRTM_GL3/SRTM_GL3_srtm/'
    if lon >= 0:
        preamble += 'North/'
        if lon < 30:
            preamble += 'North_0_29'
        else:
            preamble += 'North_30_60'
    else:
        preamble += 'South/'
    
    return preamble+filename
    # else should never occur
# -------------------------------- IO Utils
def import_raster(in_filepath):
    global args
    global possible_subdirs
    res = get_import_raster_res()
    num_hwords=res*res
    dtype = [('data', '>i2', (res,res))]
    if get_ext(in_filepath) == '.hgt':
        # if file doesnt exist, download from server then open
        if not os.path.isfile(in_filepath):
            urllib.retrieve(get_url_from_server(in_filename), in_filename)
        raster = np.memmap(in_filepath, np.dtype('>i2'), shape =num_hwords, mode = 'r').reshape((res,res))
    else:
        return None
    new_res = args.tile_res
    if new_res != res:
        scale_factor = float(new_res)/res, float(new_res)/res
        raster = skimage.transform.rescale(raster, scale=scale_factor, mode='wrap', preserve_range=True, multichannel=False, anti_aliasing=True)
    return raster
    
 
def export_raster(raster, out_filepath):
    global args
        
    if args.format == 'r16':    
        # sea level for ue4 is 32768
        raster = raster.astype('u2')
    else:
        print(f'Error: Could not export {args.format} Nopt supported')
        return False
    out_filepath = f'{out_filepath}.{args.format}'    
    raster.tofile(f'{out_filepath}')
    return True

 
def export_raster_color(raster, out_filename, minmax, bias=0.1):
    '''
        Expects a normalized valud within range [min,max]
    '''
    if raster is None:
       return 
    
    colormap = mpl.cm.get_cmap('gist_earth')
    raster = (raster-minmax[0])/(minmax[1]-minmax[0])+bias
    im = np.array(raster)
    im = colormap(im)
    im = np.uint8(im * 255)
    im = Image.fromarray(im)
    im.save(out_filename)

def export_raster_normalmap(raster, out_filename, minmax):
    '''
        Expects a normalized valud within range [min,max]
    '''
    if raster is None:
       return 
       

    colormap = mpl.cm.get_cmap('gist_earth')
    raster = (raster-minmax[0])/(minmax[1]-minmax[0])
    im = np.array(raster)
    im = colormap(im)
    im = np.uint8(im * 255)
    im = Image.fromarray(im)
    im.save(out_filename)

 
## --------------------------------- Raster Grid Operations 
####  Numpy Array helpers (self documenting)
def get_first_col(arr):
    return arr[:,0]
    
def get_last_col(arr):
    return arr[:,-1]

def get_first_row(arr):
    return arr[0,:]
    
def get_last_row(arr):
    return arr[-1,:]

def append_row(arr, row):
    return np.vstack((arr, row))

def prepend_row(arr, row):
    return np.vstack((row, arr))
    
def append_col(arr, col):
    return np.vstack((arr.T, col)).T

def prepend_col(arr, col):
    return np.vstack((col, arr.T)).T
    
def strip_first_row(arr):
    return arr[1:,:]
        
def strip_last_row(arr):
    return arr[0:-1,:]

def strip_first_col(arr):
    return arr[:,1:]
        
def strip_last_col(arr):
    return arr[:,0:-1]

    

    
def smooth_raster_grid(raster_grid, sigma):
    # USE MEDIAN FILTER AS WELL
    # TODO TODO
    xmax, ymax = raster_grid.shape[1],raster_grid.shape[0] 
    
    for y in range(raster_grid.shape[0]):
        for x in range(raster_grid.shape[1]):
            padded_raster = raster_grid[y][x]
            # pads 
            pad_top, pad_bottom, pad_left, pad_right = None,None,None,None
            if x - 1 >= 0:    # use last column of left  tile for left padding
                pad_left = get_last_col( raster_grid[y][x-1])
                padded_raster = prepend_col(padded_raster, pad_left)
            if x + 1 < xmax: # use the first column of the right  tile for right padding
                pad_right = get_first_col( raster_grid[y][x+1])
                padded_raster = append_col(padded_raster, pad_right)
            if y + 1 < ymax:    # use last row of top tile for top padding
                pad_top = get_first_row( raster_grid[y+1][x])
                # scale if left or right padd was added. Extents to fit new res
                scale_factor = float(padded_raster.shape[1])/pad_top.shape[0]
                pad_top = skimage.transform.rescale(pad_top, scale=scale_factor,  preserve_range=True, multichannel=False, anti_aliasing=True)
 
                padded_raster = prepend_row(padded_raster, pad_top)
            if y - 1 >= 0: # use the first row of the bottom tile for bottom padding
                pad_bottom = get_last_row( raster_grid[y-1][x])
                
                # scale if left or right padd was added. Extents to fit new res
                scale_factor = float(padded_raster.shape[1])/pad_bottom.shape[0]
                pad_bottom = skimage.transform.rescale(pad_bottom, scale=scale_factor, preserve_range=True, multichannel=False, anti_aliasing=True)
        
                padded_raster = append_row(padded_raster, pad_bottom)
            
            padded_raster  = skimage.filters.gaussian(padded_raster , sigma=sigma, mode='nearest', preserve_range=True, multichannel=False)
    
            # strip results if padded
            if not pad_left is None:
                padded_raster = strip_first_col(padded_raster)
            if not  pad_right is None:
                padded_raster = strip_last_col(padded_raster)
            if not pad_top is None:
                padded_raster = strip_first_row(padded_raster)
            if not pad_bottom is None:
                padded_raster = strip_last_row(padded_raster)
            
            raster_grid[y][x] = padded_raster

    #all_data = np.hstack((my_data, new_col))


    
def normalize_raster_grid(raster_grid, minmax):
    '''
    normalizes raster map within the given range
    '''
    assert(minmax[0] <  minmax[1])
    new_min, new_max=  minmax
    #print(raster_grid.shape)
    minz, maxz = get_minmax(raster_grid)
    
    #print(f'Min {minz} Max: {maxz}')
    old_range = maxz-minz
    new_range = new_max-new_min
    
    for y in range(raster_grid.shape[0]):
        for x in range(raster_grid.shape[1]):
            raster = raster_grid[y][x]
            if raster is None:
                continue
            raster_grid[y][x] = new_range/old_range * (raster - minz ) + new_min 
 

    
def exec_args():
    global args
    global possible_subdirs
    
    parse_cmd_args()
    if args.bbox:
        raster_grid, filepath_grid = get_raster_grid(args.bbox)
    else:
        print(f'Area of interest must be specified using BBox')
        return 
        
    # get tile output dir if exists
    tile_output_dir = os.path.dirname(args.tileprefix)
    
    if tile_output_dir != '' and not os.path.exists(tile_output_dir):
        os.mkdir(tile_output_dir)
    
    
    if args.sigma > 0:
        print('Smoothing Tiles...')
        sys.stdout.flush()
        smooth_raster_grid(raster_grid, args.sigma)
   
    is_normalized  = False
    if len(args.normalize ):
        print('Normalizing...')
        sys.stdout.flush()
        normalize_raster_grid(raster_grid, args.normalize)
        is_normalized  = True
    
    
    print('Exporting Tiles...')
    sys.stdout.flush()
    for y in range(raster_grid.shape[0]):
        for x in range(raster_grid.shape[1]):
            
            # ue4 format
            # TODO create cmd arg that for an output format string
            out_filepath = f'{args.tileprefix}_x{x}_y{y}'
            raster = raster_grid[y][x]
            if raster is None:
                continue
            minz, maxz = int(np.amin(raster)), int(np.amax(raster))
            native_filepath = filepath_grid[y][x]
            print(f'{native_filepath} ==> {out_filepath}\t Elevation Range: [{minz},\t{maxz}] ')    
 
            if not export_raster(raster, out_filepath):
                print(f'WARNING: Failed to Export this file !')
    
    # export colors last, this process may normalize raster map if not already normalized.
    if args.colors : 
        print('Generating Colors...')
        sys.stdout.flush()

        color_output_dir = os.path.dirname(args.colorprefix)
    
        if color_output_dir != '' and not os.path.exists(color_output_dir):
            os.mkdir(color_output_dir)
        if is_normalized:
            minmax =  args.normalize           
        else:
            # TODO Prevent from having sideeffect (possible copy raster_grid???)

            minmax=  (0,1)
            normalize_raster_grid(raster_grid, minmax)
 
        
               # Create a 8-bit PNG that contains color normalized to entire raster_grid min and max. 
        for y in range(raster_grid.shape[0]):
            for x in range(raster_grid.shape[1]):
                    out_filename = f'{args.colorprefix}_x{x}_y{y}.png'
                    native_filepath = filepath_grid[y][x]
                    print(f'{native_filepath} ==> {out_filename}\t Elevation Range: [{minz},\t{maxz}] ')    

                    export_raster_color(raster_grid[y][x], out_filename, minmax)
                    
                 
        
    return
            
    
            
if __name__ == '__main__':
    exec_args()