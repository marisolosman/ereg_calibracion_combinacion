#!/bin/bash

LANG=en_UK
declare -a variables=("tref" "prec")
declare -a wtech=("same" "pdf_int" "mean_cor")  # weighting
declare -a ctech=("wpdf" "wsereg" "count")  # combination

readonly yellow=$(tput setaf 3)
readonly reset=$(tput sgr0)

usage() {
  echo -e " -c,  --CV    \t Apply Cross Validation."
  echo -e " -o,  --OW    \t Overwrite previous calibrations."
  echo -e " -ca, --IG-CA \t Ignore calibration."
  echo -e " -co, --IG-CO \t Ignore combination."
  echo -e " -pl, --IG-PL \t Ignore plotting."
  echo -e " -h,  --help  \t Display a help message and quit."
}

# process script inputs
while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--CV) CV="--CV"; shift 1;;
    -o|--OW) OW="--OW"; shift 1;;
    -ca|--IG-CA) CA="false"; shift 1;;
    -co|--IG-CO) CO="false"; shift 1;;
    -pl|--IG-PL) PL="false"; shift 1;;
    -h|--help|*) usage; exit;;
  esac
done

# set default values when needed
readonly CA=${CA:-"true"}
readonly CO=${CO:-"true"}
readonly PL=${PL:-"true"}

if [ "${CA}" = "true" ]; then
  echo ${yellow}"Iniciando calibración"${reset}
  for j in "${variables[@]}" ; do #loop sobre las variables a calibrar, por ahora me enfoco en prec
    for n in {1..12} ; do # loop over IC
      for k in {1..7} ; do #loop over leadtime
        echo ${yellow}"$(date +'%D-%T') -- Running -- python calibration.py ${j} ${n} ${k} ${CV} ${OW}"${reset}
        python calibration.py ${j} ${n} ${k} ${CV} ${OW}
      done
    done
  done
fi

if [ "${CO}" = "true" ]; then
  echo ${yellow}"Iniciando combinación"${reset}
  for j in "${variables[@]}" ; do #loop sobre las variables a calibrar, por ahora me enfoco en prec
    for n in {1..12} ; do # loop over IC
      for k in {1..7} ; do #loop over leadtime
        for l in "${ctech[@]}" ; do #loop sobre manera de combinarr modelos
          if [ "${l}" = "count" ] ; then
            echo ${yellow}"$(date +'%D-%T') -- Running -- python combination.py ${j} ${n} ${k} ${l}"${reset}
            python combination.py ${j} ${n} ${k} ${l}
          else
            for w in "${wtech[@]}" ; do 
              if [ "${l}" = "wsereg" ] ; then 
                echo ${yellow}"$(date +'%D-%T') -- Running -- python combination.py ${j} ${n} ${k} ${l} --weight_tech ${w}"${reset}
                python combination.py ${j} ${n} ${k} ${l} --weight_tech ${w}
              else
                echo ${yellow}"$(date +'%D-%T') -- Running -- python combination.py ${j} ${n} ${k} ${l} --weight_tech ${w}"${reset}
                python combination.py ${j} ${n} ${k} ${l} --weight_tech ${w}
              fi
            done
          fi
        done
      done
    done
  done
fi

if [ "${PL}" = "true" ]; then
  echo ${yellow}"Iniciando generación de gráficos"${reset}
  for j in "${variables[@]}" ; do #loop sobre las variables a calibrar, por ahora me enfoco en prec
    for n in {1..12} ; do # loop over IC
      for k in {1..7} ; do #loop over leadtime
        echo ${yellow}"$(date +'%D-%T') -- Running -- python plot_forecast.py ${j} ${n} ${k}"${reset}
        python plot_forecast.py ${j} ${n} ${k}
      done
      echo ${yellow}"$(date +'%D-%T') -- Running -- python plot_observed_category.py ${j} ${n}"${reset}
      python plot_observed_category.py ${j} ${n}
    done
  done
fi

