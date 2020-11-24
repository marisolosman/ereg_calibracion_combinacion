#!/bin/bash

# OBS:
# El resultado de la calibración sin cross-validation se usa para la combinación en tiempo real "real_time_combination.py"
# El resultado de la calibración con cross-validation se usa para la combinación general "combination.py"

LANG=en_UK
declare -a variables=("prec" "tref")
declare -a wtech=("same" "pdf_int" "mean_cor")  # weighting
declare -a ctech=("wsereg" "wpdf")  # combination

readonly yellow=$(tput setaf 3)
readonly reset=$(tput sgr0)

usage() {
  echo -e " -y,  --year       \t Year. The default value is the current year."
  echo -e " -m,  --month      \t Month. The default value is the current month."
  echo -e " -ow, --overwrite  \t Overwrite previous calibrations."
  echo -e " -ca, --calibrate  \t Run calibration without cross-validation."
  echo -e " -h,  --help       \t Display a help message and quit."
}

# process script inputs
while [[ $# -gt 0 ]]; do
  case $1 in
    -y|--year)  y=$2; shift 2;;
    -m|--month) m=$2; shift 2;;
    -ow|--overwrite) OW="--OW"; shift 1;;
    -ca|--calibrate) CA="true"; shift 1;;
    -h|--help|*) usage; exit;;
  esac
done

# set default values when needed
readonly y=${y:-$(date +"%Y")}
readonly m=${m:-$(date +"%m")}
readonly CA=${CA:-"false"}

if [ "${CA}" = "true" ]; then
  echo ${yellow}"Iniciando calibración"${reset}
  for j in "${variables[@]}" ; do #loop sobre las variables a calibrar, por ahora me enfoco en prec
    for n in {1..12} ; do # loop over IC
      for k in {1..7} ; do #loop over leadtime
        echo ${yellow}"$(date +'%D-%T') -- Running -- python calibration.py ${j} ${n} ${k} ${OW}"${reset}
        python calibration.py ${j} ${n} ${k} ${CV} ${OW}
      done
    done
  done
fi

echo ${yellow}"Iniciando combinación en tiempo real"${reset}
for var in "${variables[@]}" ; do
	for k in {1..7} ; do #loop over leadtime
		for j in "${ctech[@]}" ; do
			for w in "${wtech[@]}" ; do
        echo ${yellow}"$(date +'%D-%T') -- Running -- python real_time_combination.py ${var} --IC ${y}-${m}-01 --leadtime ${k} ${OW} ${j} --weight_tech ${w}"${reset}
				python real_time_combination.py ${var} --IC ${y}"-"${m}"-01" --leadtime ${k} ${OW} ${j} --weight_tech ${w}
			done
		done
		echo ${yellow}"$(date +'%D-%T') -- Running -- python plot_rt_forecast.py ${var} --IC ${y}-${m}-01 --leadtime ${k}"${reset}
		python plot_rt_forecast.py ${var} --IC ${y}"-"${m}"-01" --leadtime ${k} 
	done
done


