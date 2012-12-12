#include <hmm.hpp>

#define BOOST_TEST_MODULE Generate_Train_Test
#include <boost/test/unit_test.hpp>

using namespace pslrhmm;
using namespace std;

#define NUM_SEQ 100

void initAlphabet(vector<Emission*>& alpha, size_t num) {
	for (size_t i=0; i<num; i++) {
		alpha.push_back(new IntEmission(num));
	}
}

BOOST_AUTO_TEST_CASE( generate1 ) {
	vector<Emission*> alphabet;
	initAlphabet(alphabet, 10);

	HMM::Random r(time(NULL));
	HMM hmm1, hmm2;
	hmm1.initRandom(r, 15, alphabet);
	hmm2.initRandom(r, 15, alphabet);

	double ts1l1 = 0.0, ts1l2 = 0.0, ts2l1 = 0.0, ts2l2 = 0.0; 
	for(size_t i=0; i<NUM_SEQ; i++) {
		HMM::Sequence s1, s2;
		hmm1.generateSequence(r, s1, 200);
		hmm2.generateSequence(r, s2, 200);

		double ls1l1 = hmm1.calcSequenceLikelihoodLog(s1);
		double ls1l2 = hmm2.calcSequenceLikelihoodLog(s1);

		double ls2l1 = hmm1.calcSequenceLikelihoodLog(s2);
		double ls2l2 = hmm2.calcSequenceLikelihoodLog(s2);

		ts1l1 += ls1l1;
		ts1l2 += ls1l2;
		ts2l1 += ls2l1;
		ts2l2 += ls2l2;

		// printf("%le %le, %le %le, %u %u\n",
			// ls1l1, ls1l2, ls2l1, ls2l2, ts1l1 > ts1l2, ts2l1 < ts2l2 );
	}

	ts1l1 /= NUM_SEQ;
	ts1l2 /= NUM_SEQ;
	ts2l1 /= NUM_SEQ;
	ts2l2 /= NUM_SEQ;
	
	printf("Total ratios: \n");
	printf("%le, %le\n", exp(ts1l1 - ts1l2), exp(ts2l2 - ts2l1));
	BOOST_CHECK(ts1l1 > ts1l2);
	BOOST_CHECK(ts2l1 < ts2l2);
}

BOOST_AUTO_TEST_CASE( generate_train1 ) {
	vector<Emission*> alphabet;
	initAlphabet(alphabet, 10);

	HMM::Random r(time(NULL));
	HMM hmm1, hmm2;
	hmm1.initRandom(r, 15, alphabet);
	hmm2.initRandom(r, 15, alphabet);

	vector<HMM::Sequence> seqs1, seqs2;
	for(size_t i=0; i<NUM_SEQ; i++) {
		HMM::Sequence s1, s2;
		hmm1.generateSequence(r, s1, 200);
		hmm2.generateSequence(r, s2, 200);

		seqs1.push_back(s1);
		seqs2.push_back(s2);
	}

	// Trained HMMs
	HMM hmm1a, hmm2a;
	hmm1a.train(15, seqs1);
	hmm2a.train(15, seqs2);

	double ts1l1 = 0.0, ts1l2 = 0.0, ts2l1 = 0.0, ts2l2 = 0.0; 
	for(size_t i=0; i<NUM_SEQ; i++) {
		HMM::Sequence s1, s2;
		hmm1.generateSequence(r, s1, 200);
		hmm2.generateSequence(r, s2, 200);

		double ls1l1 = hmm1a.calcSequenceLikelihoodLog(s1);
		double ls1l2 = hmm2a.calcSequenceLikelihoodLog(s1);

		double ls2l1 = hmm1a.calcSequenceLikelihoodLog(s2);
		double ls2l2 = hmm2a.calcSequenceLikelihoodLog(s2);

		ts1l1 += ls1l1;
		ts1l2 += ls1l2;
		ts2l1 += ls2l1;
		ts2l2 += ls2l2;

		// printf("%le %le, %le %le, %u %u\n",
			// ls1l1, ls1l2, ls2l1, ls2l2, ts1l1 > ts1l2, ts2l1 < ts2l2 );
	}
	
	printf("Total ratios: \n");
	printf("%le, %le\n", exp(ts1l1 - ts1l2), exp(ts2l2 - ts2l1));
	BOOST_CHECK(ts1l1 > ts1l2);
	BOOST_CHECK(ts2l1 < ts2l2);
}