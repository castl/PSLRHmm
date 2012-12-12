
env = Environment(
	CPPPATH = ['./src/'],
	CXXFLAGS = ['-O3', '-msse4', '-std=c++0x', '-Wall', '-g', '-fopenmp'],
	LINKFLAGS =['-fopenmp'])

hmmlib = env.Library('psrlhmm', Glob("src/*.cpp"))
env.Append(LIBS=[hmmlib])

env.Append(LIBS=['boost_unit_test_framework-mt'])
hmmtest = env.Program('hmmtest', Glob("test/*.cpp"))
