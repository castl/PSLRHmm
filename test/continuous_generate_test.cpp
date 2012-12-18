#include <hmm.hpp>

#include <boost/test/unit_test.hpp>

using namespace pslrhmm;
using namespace std;

#define NUM_SEQ 100
typedef NormalEmissions Emission;
typedef HMM<Emission> MyHMM;

BOOST_AUTO_TEST_CASE( continuous_generate1 ) {
	MyHMM::Random r(time(NULL));
	MyHMM hmm1, hmm2;
	hmm1.initRandom(r, 20);
	hmm2.initRandom(r, 20);

	double ts1l1 = 0.0, ts1l2 = 0.0, ts2l1 = 0.0, ts2l2 = 0.0; 
	#pragma omp parallel for \
		default(shared)
	for(size_t i=0; i<NUM_SEQ; i++) {
		MyHMM::Sequence s1, s2;
		hmm1.generateSequence(r, s1, 200);
		hmm2.generateSequence(r, s2, 200);

		double ls1l1 = hmm1.calcSequenceLikelihoodLog(s1);
		double ls1l2 = hmm2.calcSequenceLikelihoodLog(s1);

		double ls2l1 = hmm1.calcSequenceLikelihoodLog(s2);
		double ls2l2 = hmm2.calcSequenceLikelihoodLog(s2);

		#pragma omp atomic
		ts1l1 += ls1l1;

		#pragma omp atomic
		ts1l2 += ls1l2;

		#pragma omp atomic
		ts2l1 += ls2l1;

		#pragma omp atomic
		ts2l2 += ls2l2;

		// printf("%le %le, %le %le, %u %u\n",
			// ls1l1, ls1l2, ls2l1, ls2l2, ts1l1 > ts1l2, ts2l1 < ts2l2 );
	}

	ts1l1 /= NUM_SEQ;
	ts1l2 /= NUM_SEQ;
	ts2l1 /= NUM_SEQ;
	ts2l2 /= NUM_SEQ;
	
	printf("Total ratios: \n");
	printf("%le (exp: %le), %le (exp: %le)\n",
			ts1l1 - ts1l2, exp(ts1l1 - ts1l2),
			ts2l2 - ts2l1, exp(ts2l2 - ts2l1));
	BOOST_CHECK(ts1l1 > ts1l2);
	BOOST_CHECK(ts2l1 < ts2l2);
}

BOOST_AUTO_TEST_CASE( continuous_generate_train1 ) {
	Emission ex;

	MyHMM::Random r(time(NULL));
	MyHMM hmm1, hmm2;
	hmm1.initRandom(r, 15);
	hmm2.initRandom(r, 15);

	vector<MyHMM::Sequence> seqs1, seqs2;
	#pragma omp parallel for \
		default(shared)
	for(size_t i=0; i<NUM_SEQ; i++) {
		MyHMM::Sequence s1, s2;
		hmm1.generateSequence(r, s1, 500);
		hmm2.generateSequence(r, s2, 500);

		#pragma omp critical (append) 
		{
			seqs1.push_back(s1);
			seqs2.push_back(s2);
		}
	}

	// Trained HMMs
	MyHMM hmm1a, hmm2a;
	hmm1a.initUniform(15, ex);
	for (size_t i=0; i<10; i++)
		hmm1a.baum_welch(seqs1);

	hmm2a.initUniform(15, ex);
	for (size_t i=0; i<10; i++)
		hmm2a.baum_welch(seqs2);

	double ts1l1 = 0.0, ts1l2 = 0.0, ts2l1 = 0.0, ts2l2 = 0.0; 
	#pragma omp parallel for \
		default(shared)
	for(size_t i=0; i<NUM_SEQ; i++) {
		MyHMM::Sequence s1, s2;
		hmm1.generateSequence(r, s1, 500);
		hmm2.generateSequence(r, s2, 500);

		double ls1l1 = hmm1a.calcSequenceLikelihoodLog(s1);
		double ls1l2 = hmm2a.calcSequenceLikelihoodLog(s1);

		double ls2l1 = hmm1a.calcSequenceLikelihoodLog(s2);
		double ls2l2 = hmm2a.calcSequenceLikelihoodLog(s2);


		#pragma omp atomic
		ts1l1 += ls1l1;

		#pragma omp atomic
		ts1l2 += ls1l2;

		#pragma omp atomic
		ts2l1 += ls2l1;

		#pragma omp atomic
		ts2l2 += ls2l2;

		// printf("%le %le, %le %le, %u %u\n",
			// ls1l1, ls1l2, ls2l1, ls2l2, ts1l1 > ts1l2, ts2l1 < ts2l2 );
	}

	ts1l1 /= NUM_SEQ;
	ts1l2 /= NUM_SEQ;
	ts2l1 /= NUM_SEQ;
	ts2l2 /= NUM_SEQ;
	
	printf("Total ratios: \n");
	printf("%le (exp: %le), %le (exp: %le)\n",
			ts1l1 - ts1l2, exp(ts1l1 - ts1l2),
			ts2l2 - ts2l1, exp(ts2l2 - ts2l1));

	BOOST_CHECK(ts1l1 > ts1l2);
	BOOST_CHECK(ts2l1 < ts2l2);
}