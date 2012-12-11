#include <hmm.hpp>

#define BOOST_TEST_MODULE Generate_Train_Test
#include <boost/test/unit_test.hpp>

using namespace pslrhmm;
using namespace std;

#define NUM_SEQ 10

void initAlphabet(vector<Emission*>& alpha, size_t num) {
	for (size_t i=0; i<num; i++) {
		alpha.push_back(new IntEmission(num));
	}
}

BOOST_AUTO_TEST_CASE( generate_train1 ) {
	vector<Emission*> alphabet;
	initAlphabet(alphabet, 10);

	HMM::Random r;
	HMM hmm1, hmm2;
	hmm1.initRandom(r, 10, alphabet);
	hmm2.initRandom(r, 10, alphabet);

	for(size_t i=0; i<NUM_SEQ; i++) {
		HMM::Sequence s1, s2;
		hmm1.generateSequence(r, s1, 50);
		hmm2.generateSequence(r, s2, 50);

		double s1l1 = hmm1.calcSequenceLikihoodLog(s1);
		double s1l2 = hmm2.calcSequenceLikihoodLog(s1);

		BOOST_CHECK(s1l1 > s1l2);

		double s2l1 = hmm1.calcSequenceLikihoodLog(s2);
		double s2l2 = hmm2.calcSequenceLikihoodLog(s2);

		BOOST_CHECK(s2l1 < s2l2);
	}
}
