LibPaths = [
    '/proj/castl/experiments/android_malware/libs/mlpack-1.0.4/lib/'
]

env = Environment(
	CPPPATH = ['./src/',
				'/usr/include/libxml2',
                '/proj/castl/experiments/android_malware/libs/mlpack-1.0.4/include'
	],
	CXXFLAGS = ['-O3', '-pthread', '-msse4', '-std=c++0x', '-Wall', '-g', '-fopenmp'],
	LIBPATH = LibPaths,
	LIBS = ['mlpack', 'armadillo', 'boost_unit_test_framework-mt'],
	LINKFLAGS = ['-fopenmp', '-pthread'] + map(lambda x: "-Wl,-rpath=%s" % x, LibPaths))

converge = env.Program("converge", ['test/converge.cpp'])

hmmtest = env.Program('hmmtest',
		 Glob("test/*_test.cpp") + ["test/main.cpp"])
