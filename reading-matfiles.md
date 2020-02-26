### Usage of gt.mat

#### Requirements:

- Python Modules:
	- `numpy`
	- `h5py`

#### How to import gt.mat to python:

As simple as:

```
import numpy as np
import h5py

gt_path = "/path/to/gt_v7.3.mat"
f = h5py.File(gt_path, 'r')
```

Note that we imported `gt_v7.3.mat` instead of `gt.mat`. This is because the `mat`-file provided by the SynthText dataset is an old version that cannot be easily imported to python (as far as I know). `gt_v7.3.mat` has identical contents to `gt.mat`, only the format used is a newer version of `.mat` (i.e. v7.3, converted using MATLAB) which can be read by h5py. 


#### How to use the HDF5 object 

Now if we check the `f` variable we get

```
>>> f
<HDF5 file "gt_17.mat" (mode r)>
```

To use this object, we should first know what its structure is:

```
f = {
	"imnames": [[ref(imnames[0])],
			  			...,
				[ref(imnames[N])] ]	
	"charBB": [[ref(charBB[0]),],	// container of coordinates of charBB's
						...,
				[ref(charBB[N])] ],
	"wordBB": [[ref(wordBB[0])],	// container of coordinates of wordBB's 
						...,
				[ref(wordBB[N])] ],
	"txt": [[ref(txt[0])],		// container of strings
						...,
				[ref(txt[N])] ]
	}
```

where `ref(x)` is a reference (i.e. a pointer) to the variable `x`. So executing just `f["imnames"][0]` will yield a reference to the filename of the image corresponding to the first entry, i.e. `ref(imname[0])`. To dereference any pointer `p = ref(x)`, we simply do

```x = x[p] = x[ref(x)]```.

Thus, to get the object representing the filename of the image corresponding to the `i`-th entry in the `.mat` file, simply do

```fname_i = f[f["imnames"][i][0]]```.

This is the case for all the other variables (`charBB`, etc.)


