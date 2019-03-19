#/bin/bash
# Each tile created is a 1 latitude x 1 longitude  tile. 
# requires srtm.py

#minlat=33
#minlon=-118
#maxlat=36
#maxlon=-115
if [ $3 -lt 4 ]; then
    echo "Usage get_tiles.sh minlat minlon maxlat maxlon get_height get_satellite get_vector. All args are integers. Set get_* to 0 to prevent getting these tiles"
    exit 
fi
minlat=$1
minlon=$2
maxlat=$3
maxlon=$4
GET_SRTM_TILES=$5
GET_SAT_TILES=$6
GET_VECTOR_TILES=$7

download_osm_vector_tiles() {
    local minlat=$1
    local minlon=$2
    local maxlat=$3
    local maxlon=$4
    local prefix="Vectors_"
    local outdir="Vectors/"
   
    local lati=0
    local loni=0

    if [ ! -d $outdir ]; then
        mkdir $outdir
    fi
    
     if [ -f  osm_log.txt ]; then
        rm  osm_log.txt
    fi


    for lat in `seq $minlat $maxlat`
    do
        endlat=$((lat+1));
        loni=0
        for lon in `seq $minlon $maxlon`
        do
        
            endlon=$((lon+1))
            outfile=$(printf "%s%s_x%d_y%d.osm" $outdir $prefix $loni $lati)
            url=$(printf "https://overpass-api.de/api/map?bbox=%.4f,%.4f,%.4f,%.4f" $lon $lat $endlon $endlat)
            echo "Grabbing " $outfile "..."

            curl -o $outfile "$url" 
            # wait 2 minutes for server to allow another request
            sleep 300
            
            loni=$((loni+1))         
            
        done;
        lati=$((lati+1))    
    done;
}


download_sat_image_tiles() {
    local minlat=$1
    local minlon=$2
    local maxlat=$3
    local maxlon=$4
    local prefix="Satellite"
    local outdir="Satellite/"
    #max res for default tileserver
    local res=4096 
    local lati=0
    local loni=0

    if [ ! -d $outdir ]; then
        mkdir $outdir
    fi
    
     if [ -f  sat_log.txt ]; then
        rm  sat_log.txt
    fi


    for lat in `seq $minlat $maxlat`
    do
        endlat=$((lat+1));
        loni=0
        for lon in `seq $minlon $maxlon`
        do
        
            endlon=$((lon+1))
            
            url=$(printf "https://tiles.maps.eox.at/wms?service=wms&request=getmap&version=1.1.1&layers=s2cloudless-2018&bbox=%.4f,%.4f,%.4f,%.4f&width=%d&height=%d&srs=epsg:4326" $lon $lat $endlon $endlat $res $res)
            
            outfile=$(printf "%s%s_x%d_y%d.png" $outdir $prefix $loni $lati)
            echo "Grabbing " $outfile "..."

            curl -o $outfile "$url" 
            
            loni=$((loni+1))         
            
        done;
        lati=$((lati+1))    
    done;
}


if [ $GET_SRTM_TILES -ne 0 ]; then
    # TODO download from https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/SRTM_GL1/
    echo "Downloading Heightmap and Color Tiles "
    time python srtm.py D:/SRTM_GL1/SRTM_GL1_srtm SRTM1 r16 --bbox $minlat $minlon $maxlat $maxlon  --sigma 1 --normalize 32768 65535 --tile_res 4033 --colors > srtm_log.txt
fi

if [ $GET_SAT_TILES -ne 0 ]; then
    echo "Downloading SAT Tiles"
    time download_sat_image_tiles $minlat $minlon $maxlat $maxlon
fi

if [ $GET_VECTOR_TILES -ne 0 ]; then
    echo "Downloading Vector Tiles"
    time download_osm_vector_tiles $minlat $minlon $maxlat $maxlon
fi

echo "Done"
