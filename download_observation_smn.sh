#!/bin/bash
# download nmme observational forecast

eval ruta=$(sed -n 's|\(download_folder: \)\(.*$\)|\2|p' config.yaml)
ruta+="NMME/hindcast/"
ruta_iri="http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/"

FILE=${ruta}prec_monthly_nmme_cpc.nc
TMPF=${ruta}temp_prec_nmme_cpc.nc

wget -O "$FILE" "${ruta_iri}.CPC-CMAP-URD/.prate/data.nc"
cdo selyear,1982/2011 "$FILE" "$TMPF"
mv "$TMPF" "$FILE"
ncrename -v prate,prec "$FILE" "$TMPF"
mv "$TMPF" "$FILE" 

FILE=${ruta}tref_monthly_nmme_ghcn_cams.nc
TMPF=${ruta}temp_tref_nmme_ghcn_cams.nc

wget -O "$FILE" "${ruta_iri}.GHCN_CAMS/.updated/data.nc"
# suma 273.15 a ??
cdo addc,273.15 "$FILE" "$TMPF"
mv "$TMPF" "$FILE" 
# remover dimensión Z y variable Z
# ncks -C -O -x -v Z "$FILE" "$TMPF"
# mv "$TMPF" "$FILE" 
# remover dimension Z
# ncwa -a Z "$FILE" "$TMPF"
# mv "$TMPF" "$FILE"
# repeack
# ncpdq "$FILE" "$TMPF"
# renombrar variable temp a tref
ncrename -v t2m,tref "$FILE" "$TMPF"
mv "$TMPF" "$FILE" 

FILE=${ruta}lsmask.nc

wget -O "$FILE" "${ruta_iri}.LSMASK/.land/data.nc"

# LINK ayuda nco:
# https://yidongwonyi.wordpress.com/linux-data-handling-netcdf-nc/nco-extract-variable-delete-variabledimension/
# http://nco.sourceforge.net/nco.html#ncwa
# - Average all variables in in.nc over all dimensions and store results in out.nc: ncwa in.nc out.nc

