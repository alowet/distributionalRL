from scipy import io

### Utility functions to load Matlab struct into Python. ###
# See https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
def loadmat(filename):
	"""
	this function should be called instead of direct io.loadmat
	as it cures the problem of not properly recovering python dictionaries
	from mat files. It calls the function check keys to cure all entries
	which are still mat-objects
	"""
	data = io.loadmat(filename, struct_as_record=False, squeeze_me=True)
	return _check_keys(data)


def _check_keys(dict):
	"""
	checks if entries in dictionary are mat-objects. If yes
	todict is called to change them to nested dictionaries
	"""
	for key in dict:
		if isinstance(dict[key], io.matlab.mio5_params.mat_struct):
			dict[key] = _todict(dict[key])
	return dict        


def _todict(matobj):
	"""A recursive function which constructs from matobjects nested dictionaries"""
	dict = {}
	for strg in matobj._fieldnames:
		elem = matobj.__dict__[strg]
		if isinstance(elem, io.matlab.mio5_params.mat_struct):
			dict[strg] = _todict(elem)
		else:
			dict[strg] = elem
	return dict
