
env = Environment(
	CPPPATH = ['./src/'],
	CXXFLAGS = ['-O3', '-pthread', '-msse4', '-std=c++0x', '-Wall', '-g', '-fopenmp'],
	LINKFLAGS =['-fopenmp', '-pthread'])

converge = env.Program("converge", ['test/converge.cpp'])

hmmtest = env.Program('hmmtest',
		 Glob("test/*_test.cpp") + ["test/main.cpp"],
		 LIBS=['boost_unit_test_framework-mt'])
