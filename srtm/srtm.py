#!/usr/env python
'''
    Tool for Querying and reformatting tiles from GIS Datasets 
'''

usage = \
'''
python3 srtm.py  path/to/dataset/ options 
    options
        To query bounding box
        SRTM    minx,maxx,miny,maxy    
'''

import os 
import numpy as np 
import skimage


args = None
output_dir  = 'Out'
input_raster_res = { 'srtm1': 3601, 'srtm3': 1201}
supported_dataset_types = ['srtm1', 'srtm3' ]
supported_output_formats = ['r16' ] 
possible_subdirs = []



def get_ext(file):
    return file[file.rfind('.'):]
    
def walkfiles(dir, callback):
    for root, subdirs, files in os.walk(dir):  
        for file in files:
            callback(os.path.join(root, file))
            
def walkdirs(dir, callback):
    for root, subdirs, files in os.walk(dir):  
        for subdir in subdirs:
            callback(os.path.join(root, subdir))
        
            
def parse_cmd_args():
     
    import argparse
    
    global args 
    global supported_dataset_types 
    global supported_output_formats 
    
    
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
    parser.add_argument('-b', '--bbox', metavar='Lon Lat', type=int, nargs=4,
                        help='Longitude Latitude coordinates for Bounding box query Order from Minimum point to Maximum point')
    parser.add_argument('-r', '--res', metavar='Resolution', type=int, nargs=1, default=2017,
                        help='Resolution of output raster')
    
    parser.add_argument('-s', '--smooth', metavar='Sigma', type=float, nargs=1, default=0,
                        help='Sigma value for gaussian smoothing of tiles')
                        
    parser.add_argument('-n', '--normalize', metavar='N', type=int, nargs=1, default=0,
                        help='Normalize output raster values from 0-N')
    
  
    args = parser.parse_args()
    # reformat args as datum not lists
    args.dataset_type = args.dataset_type[0].lower()
    args.format = args.format[0].lower()
    args.dataset_path = args.dataset_path[0]
    
    # check args
    print(args)
    assert( args.dataset_type in supported_dataset_types)
    assert( args.format in supported_output_formats)

    

def get_tile_filename(longlat):
    long,lat = longlat
    if long >= 0:
        prefix_long = 'N' 
    else: # if less then zero query southern tiles. remove sign
        long *= -1
        prefix_long = 'S'

    if lat >= 0:
        prefix_lat = 'E' 
    else: # if less then zero query westward tiles. remove sign
        lat *= -1
        prefix_lat = 'W'
        
    # format [NS]Y[EW]XXX.hgt
    return f'{prefix_long}{long}{prefix_lat}{lat:03}.hgt'
    
 
def get_files_within_bbox(bbox):
    '''
    Returns a listing of all the filenames that contain values withing the given bbox.
    They are returned in row-major order
    '''

    # long lat == N/S E/W  
    if not bbox: return None
    min_long, min_lat, max_long, max_lat = bbox
    assert(min_long <= max_long)
    assert(min_lat <= max_lat)
    
    files = []
    for long in range(min_long, max_long+1):
        for lat in range(min_lat, max_lat+1):
            files .append(get_tile_filename(( long,lat ) ))
    return files


def import_raster(in_filepath):
    global args
    global possible_subdirs
    global input_raster_res
    
    res =  input_raster_res [ args.dataset_type]
    num_hwords=res*res
    dtype = [('data', '>i2', (res,res))]
    if get_ext(in_filepath) == '.hgt':
        raster = np.memmap(in_filepath, np.dtype('>i2'), shape =num_hwords, mode = 'r').reshape((res,res))
        # for y in range(res):
            # for x in range(res):
                # if raster[y][x] == -32768:
                    # raster[y][x] = (raster[ (y+1)%res ][x] + raster[ (y-1) ][x] + raster[y][ (x+1)%res ] + raster[y][ (x-1) ] )/3
                    # print("Filled Void")
    else:
        return None
    print(f'Imported {in_filepath}')   
    new_res = args.res
    scale_factor = float(new_res)/res, float(new_res)/res
    raster = skimage.transform.rescale(raster, scale=scale_factor, mode='wrap', preserve_range=True, multichannel=False, anti_aliasing=True)
    return raster
    
 
def export_raster(raster, out_filepath):
    global args
    

       
    if args.format == 'r16':    
        # sea level for ue4 is 32768
        raster = (raster.astype('i2') + 32768).astype('u2')
        #add 

        
    raster.tofile(f'{out_filepath}.{args.format}')
    minz, maxz = np.amin(raster), np.amax(raster)
    print(f'{out_filepath} ==> Elevation Range: {minz} {maxz} ')    

def get_raster_map(bbox):
    global args
    global possible_subdirs
    
    filenames = get_files_within_bbox(args.bbox)
    shape = bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1

    raster_map = {}
    for x in range(shape[0]):
        raster_map[x] = {}
    x= 0 
    y= 0        
    for filename in filenames:
        fullpath  = None
        for subdir in possible_subdirs:
            if os.path.isfile(subdir+filename):   
                fullpath = subdir+filename
                break 
        if  fullpath :
            raster =  np.array(import_raster(fullpath))
            raster_map[y][x] = raster
        else:
            print(f'WARNING: Could not find file {filename}' )
        y+=1        
        if y >= shape[0]:
            y = 0;
            x += 1 

    return raster_map
    
    
def smooth_raster_map(raster_map, sigma):
    pass
    for y in raster_map.keys():
        for x in raster_map[y].keys():
            raster = raster_map[y][x]
            # pad the raster with neightborring rasters if not at the end
            # TODO
            padded_raster  = skimage.filters.gaussian(padded_raster , sigma=3, mode='wrap', preserve_range=True, multichannel=False)

    
    
def generate_colormaps():    
    # TODO
    # Create a 8-bit PNG that contains color normalized to entire raster_map min and max. 
    pass
    
def on_each_subdir(subdir):
    global possible_subdirs
    possible_subdirs.append(subdir)
    
def run():
    global args
    global possible_subdirs

    parse_cmd_args()
    # get information about folder structure
    possible_subdirs.append(args.dataset_path)
    walkdirs(args.dataset_path, on_each_subdir)
    
    raster_map = get_raster_map(args.bbox)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # if args.sigma > 0:
        # smooth_raster_map(raster_map, sigma)
    
    for y in raster_map.keys():
        for x in raster_map[y].keys():
            
            # ue4 format
            # TODO create cmd arg that for an output format string
            out_filename = f'{output_dir}/Tile_x{x}_y{y}'
            raster = raster_map[y][x]
            export_raster(raster, out_filename)

            
    
            
if __name__ == '__main__':
    run()