#### gt.mat Structure

gt.mat
    * f[name] | name = {'charBB', 'wordBB', 'immnames', 'txt'}
        * f[name][i][0] = hdf5 object reference; 0 < i < 8750
                        = index
    
    * f[ f['charBB' or 'wordBB'][i][0] ][j] = [charBBxy4_j]; 0 < j < N_chars or N_words
        * charBBxy4_j = [ (x0,y0),
                          (x1,y1),
                          (x2,y2),
                          (x3,y3) ]
    
    * f[ f['imnames'][i][0] ] = [ [fnamechar_0], [fnamechar_1], ... ] where char_i = ith char of fname
                              = imname
        * imname[j][0] = fnamechar_j
    
    *

    
    * charBB
        * shape: (8750, 1)
            * 
    * wordBB
    * imnames
    * txt