#!/bin/bash

# Copyright (C) 2017 Yerai Doval 
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/gpl.txt>

paste <(cut -f1 $1 | sed "s/\"/\\\\\"/g ; s/\\\$/\\\\\$/g ; s/\`/\\\\\`/g") <(cut -f2 $1)
