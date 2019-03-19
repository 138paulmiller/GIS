#!/bin/sh
minlat=33
minlon=-118
maxlat=35
maxlon=-116
get_srtm_tiles=1
get_sat_tiles=1
get_vector_tiles=0
./get_tiles.sh $minlat $minlon $maxlat $maxlon  $get_srtm_tiles $get_sat_tiles $get_vector_tiles