
env = Environment(
	CPPPATH = ['./src/'],
	CXX = 'clang++',
	# CXXFLAGS = ['-O0', '-mfpmath=sse', '-msse4', '-march=native',
	          # '-Wall', '-g', '-std=c++0x', '-fopenmp'],
	CXXFLAGS = ['-O3', '-msse4', '-std=c++11', '-Wall', '-g'],
	LIBS=['boost_thread-mt', 'boost_program_options-mt'])

hmmlib = env.Library('psrlhmm', Glob("src/*.cpp"))
env.Append(LIBS=[hmmlib])

env.Append(LIBS=['boost_unit_test_framework-mt'])
hmmtest = env.Program('hmmtest', Glob("test/*.cpp"))