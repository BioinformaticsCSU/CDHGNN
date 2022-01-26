# License

Copyright (C) 2020 Jianxin Wang(jxwang@mail.csu.edu.cn),Chengqian Lu(chengqlu@csu.edu.cn)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see <http://www.gnu.org/licenses/>.

Jianxin Wang(jxwang@mail.csu.edu.cn),Chengqian Lu(chengqlu@csu.edu.cn)
School of Information Science and Engineering
Central South University
ChangSha
CHINA, 410083

Type: Package

# Title: CDHGNN: prioritizing disease-related circRNAs with heterogeneous graph neural networks and attention mechanism

Description: This package implements the CDHGNN algorithm with heterogeneous graph neural networks and attention mechanism for predicting circRNA-disease associations.

Files:
1.data

1) circmiRna_starbase.txt stores knonw circRNA-miRNA associations from starBase.

2) Experimental circRNA-disease information.txt stores knonw circRNA-disease associations from MNDR.

3) miRNA-disease.txt stores knonw miRNA-disease associations from HMDD 3

4) MISIM2_similarity.txt stores knonw MISIM 2.0



2.code
1) dataLoad.py
2) GIP.py
3) seq_encoding.py
4) utils.py
5) model.py