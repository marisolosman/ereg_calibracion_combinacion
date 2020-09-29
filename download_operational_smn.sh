#!/bin/bash
#download nmme operational forecast
eval ruta=$(sed -n 's|\(download_folder: \)\(.*$\)|\2|p' config.yaml)
ruta+="NMME/real_time/"
ruta_iri="http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/"
MONTHS=(ZERO Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec)
declare -a variables=("prec" "tref")
for j in "${variables[@]}" ; do
	for k in {2020..2020} ;	do 
		#loops sobre meses
		for l in `seq -f "%02g" 1 1 12`; do 
			FM=`expr $[10#$l] + 11`
			FY=$k
			if [ $((10#$FM)) -gt 12 ] ; then 
				FM=`expr $FM - 12`
				FY=`expr $k + 1`
			fi
			#pongo 0 si el mes de prono es menor a 10
			if [ $((10#$FM)) -le 9 ] ; then 
				mespr="0${FM#0}"
			else
				mespr=$FM
			fi
			for m in {1..28} ; do
				if [ $m -le 4 ] ; then
					FILE=${ruta}${j}_Amon_NASA-GEOS5_${k}${l}_r${m}_${k}${l}-${FY}${mespr}.nc
					if [ ! -f "$FILE" ] ; then
						wget -O "$FILE" "${ruta_iri}.NASA-GEOSS2S/.FORECAST/.MONTHLY/.${j}/S/%280000%201%20${MONTHS[${l#0}]}%20${k}%20%29VALUES/M/%28${m}.0%20%29VALUES/data.nc"
					fi
				fi

				if [ $m -le 10 ] ; then
					FILE=${ruta}${j}_Amon_COLA-CCSM4_${k}${l}_r${m}_${k}${l}-${FY}${mespr}.nc
				        if [ ! -f "$FILE" ] ; then	
					wget -O "$FILE" "${ruta_iri}.COLA-RSMAS-CCSM4/.MONTHLY/.${j}/S/%280000%201%20${MONTHS[${l#0}]}%20${k}%20%29VALUES/M/%28${m}.0%20%29VALUES/data.nc"
				        fi
					#GFDL-CM2p1
					FILE=${ruta}${j}_Amon_GFDL-CM2p1_${k}${l}_r${m}_${k}${l}-${FY}${mespr}.nc
					if [ ! -f "$FILE" ] ; then
					wget -O "$FILE" "${ruta_iri}.GFDL-CM2p1-aer04/.MONTHLY/.${j}/S/%280000%201%20${MONTHS[${l#0}]}%20${k}%29VALUES/M/%28${m}.0%20%29VALUES/data.nc"
					fi
					FILE=${ruta}${j}_Amon_CMC-CanCM4i_${k}${l}_r${m}_${k}${l}-${FY}${mespr}.nc
					if [ ! -f "$FILE" ] ; then
					#CMC-Can4i
					wget -O "$FILE" "${ruta_iri}.CanCM4i/.FORECAST/.MONTHLY/.${j}/S/%280000%201%20${MONTHS[${l#0}]}%20${k}%20%29VALUES/M/%28${m}.0%20%29VALUES/data.nc"
					fi
				fi 
				if [ $m -le 12 ] ; then 
					#GFDL FLOR-A06
					FILE=${ruta}${j}_Amon_GFDL-FLOR-A06_${k}${l}_r${m}_${k}${l}-${FY}${mespr}.nc
					if [ ! -f "$FILE" ] ; then
						wget -O "$FILE" "${ruta_iri}.GFDL-CM2p5-FLOR-A06/.MONTHLY/.${j}/S/%280000%201%20${MONTHS[${l#0}]}%20${k}%29VALUES/M/%28${m}.0%20%29VALUES/data.nc"
					fi
					FILE=${ruta}${j}_Amon_GFDL-FLOR-B01_${k}${l}_r${m}_${k}${l}-${FY}${mespr}.nc
					if [ ! -f "$FILE" ] ; then
						#GFDL FLOR-A06
						wget -O  "$FILE" "${ruta_iri}.GFDL-CM2p5-FLOR-B01/.MONTHLY/.${j}/S/%280000%201%20${MONTHS[${l#0}]}%20${k}%29VALUES/M/%28${m}.0%20%29VALUES/data.nc"
					fi

				fi 
				if [ $m -le 20 ] ; then 
					#CMC-CanSIPSv2
					FILE=${ruta}${j}_Amon_CMC-CanSIPSv2_${k}${l}_r${m}_${k}${l}-${FY}${mespr}.nc
					if [ ! -f "$FILE" ] ; then 
						wget -O "$FILE" "${ruta_iri}.CanSIPSv2/.FORECAST/.MONTHLY/.${j}/S/%280000%201%20${MONTHS[${l#0}]}%20${k}%20%29VALUES/M/%28${m}.0%20%29VALUES/data.nc"
					fi
				fi
				if [ $m -gt 24 ] && [ ${l} -eq 11 ] ; then
					#CFS
					FILE=${ruta}${j}_Amon_NCEP-CFSv2_${k}${l}_r${m}_${k}${l}-${FY}${mespr}.nc
					if [ ! -f "$FILE" ] ; then
						wget -O "$FILE" "${ruta_iri}.NCEP-CFSv2/.FORECAST/.PENTAD_SAMPLES/.MONTHLY/.${j}/S/%280000%201%20${MONTHS[${l#0}]}%20${k}%29VALUES/M/%28${m}.0%20%29VALUES/data.nc"
					fi 
				else
					FILE=${ruta}${j}_Amon_NCEP-CFSv2_${k}${l}_r${m}_${k}${l}-${FY}${mespr}.nc
					if [ ! -f "$FILE" ] ; then
						wget -O "$FILE" "${ruta_iri}.NCEP-CFSv2/.FORECAST/.PENTAD_SAMPLES/.MONTHLY/.${j}/S/%280000%201%20${MONTHS[${l#0}]}%20${k}%29VALUES/M/%28${m}.0%20%29VALUES/data.nc"
					fi 

				fi 
				#find ${ruta}${j}_Amon_*_${k}${l}_*.nc -size 0M -delete

done #loop sobre miembros

done #loop sobre meses

done #loop sobre anios

done #loop sobre variables
