set title "Free Surface example"

set grid

set key top left
#set key off

set view 60,60

set xlabel "time"
set ylabel "dim0"
set zlabel "z"

#set xrange[0.0:0.0]
#set yrange[0.0:0.0]
#set zrange[0.0:0.0]

#set datafile missing '0.000000000000000000e+00'
set datafile separator comma

set dgrid3d 100,100
#set hidden3d 

#set palette model CMY rgbformulae 7,5,15
set palette rgbformulae 33,13,10

#splot "performance_parameters.txt" using ($1/1000000.0):($2-273.15):($3) notitle with points palette pointsize 1 pointtype 7

#set terminal png size 400,300 enhanced
#set output '~/Downloads/output.png'

splot "evaluations/0_measurement.csv" using 0:2:3 notitle with pm3d palette
#splot "evaluations/0_measurement.csv" using 0:2:3