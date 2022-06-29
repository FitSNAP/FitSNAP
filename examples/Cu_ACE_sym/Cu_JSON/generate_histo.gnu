#Gnuplot Script for making a nice looking histogram
###################################################
!./DistOfEnergy_JSON.sh
input='EnergyHistogram.dat'

# set xr[200:2000]
# set yr[1000:1E8]
 set yl'Numerical Count'
 set xl'Energy (eV/atom)'
 set terminal pngcairo size 960,720 enhanced font 'Verdana,24,itallic,bold'
 set key font ",16"
 set xtics nomirror
 set ytics nomirror
 set grid
 set border ls 1.5 ls 6
 set style line 1 lc rgb '#e41a1c' lt 1 lw 2 pt 5
 set style line 2 lc rgb '#377eb8' lt 1 lw 2 pt 7
 set style line 3 lc rgb '#4daf4a' lt 1 lw 2 pt 9
 set style line 4 lc rgb '#984ea3' lt 1 lw 2 pt 11
 set style line 5 lc rgb '#ff7f00' lt 1 lw 2 pt 13
 set style line 6 lc rgb '#000000' lt 1 lw 2 pt 15
 set style line 7 lc rgb '#a65628' lt 1 lw 2 pt 4
 set style line 8 lc rgb '#f781bf' lt 1 lw 2 pt 6
 set style line 9 lc rgb '#666666' lt 1 lw 2 pt 8
 set style line 10 lc rgb '#e6ab02' lt 1 lw 2 pt 10
 set border ls 1.5 ls 6
 set pointsize 2
 set key b r  width -12 samplen 1 spacing 0.8
 set xtics font ",16"
 set ytics font ",16"
 set output 'tmp.png'

binwidth=0.005
bin(x,width)=width*floor(x/width)
plot input using (bin(($2/$3),binwidth)):(1.0) smooth freq with l ls 1 notitle

set table "hist.dat"
plot input using (bin(($2/$3),binwidth)):(1.0) smooth freq
unset table

# set output 'tmp2.png'
#!./IntegrateHisto.dat

#set yl'Percent of Shocked Material'
#set xl'Temperature (K)'

#set yr[0:100]
#set xr[200:2000]

#plot 'normInt_hist.dat' u ($1/337):3 w l ls 1 t'Integral','' u ($1/337):($2*5) w l ls 2 t'Histogram x5' 

