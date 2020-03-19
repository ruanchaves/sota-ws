#!/bin/bash

# Copyright (C) 2017 Yerai Doval 
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/gpl.txt>

NLINES="$(wc -l $1 | cut -d" " -f1)"
if [ $NLINES -ne $(wc -l $2 | cut -d" " -f1) ]
then
	echo "File size mismatch"
	exit
fi
echo "scale = 3; ($NLINES - $(diff -U 0 $1 $2 | grep "^@" | wc -l)) / $NLINES" | bc
