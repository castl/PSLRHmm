#include <hmm.hpp>

using namespace pslrhmm;
using namespace std;

#define NUM_SEQ 50
#define NUM_STATES 10
#define ITERS 100

template<typename E>
void initAlphabet(vector<E>& alpha, size_t num) {
	for (size_t i=0; i<num; i++) {
		alpha.push_back((uint64_t)i);
	}
}

void converge(void) {
	vector<HMM<>::Emission> alphabet;
	initAlphabet(alphabet, 10);

	HMM<>::Random r(5);
	HMM<> hmm1, hmm2;
	hmm1.initRandom(r, NUM_STATES, alphabet);
	hmm2.initRandom(r, NUM_STATES, alphabet);

	vector<HMM<>::Sequence> training_seqs;
	for(size_t i=0; i<NUM_SEQ; i++) {
		HMM<>::Sequence s1;
		hmm1.generateSequence(r, s1, 400);
		training_seqs.push_back(s1);
	}

	vector<HMM<>::Sequence> testing_seqs1;
	vector<HMM<>::Sequence> testing_seqs2;
	for(size_t i=0; i<NUM_SEQ; i++) {
		HMM<>::Sequence s1, s2;
		hmm1.generateSequence(r, s1, 400);
		hmm2.generateSequence(r, s2, 400);
		testing_seqs1.push_back(s1);
		testing_seqs2.push_back(s2);
	}

	HMM<> hmmT;
	hmmT.initRandom(r, NUM_STATES, alphabet);

	printf("    Train          Test         Test (Hmm2)\n");
	for (size_t i=0; i<ITERS; i++) {
		printf("%lu: %le %le %le\n", i,
			hmmT.calcSequenceLikelihoodLog(training_seqs),
			hmmT.calcSequenceLikelihoodLog(testing_seqs1),
			hmmT.calcSequenceLikelihoodLog(testing_seqs2));

		hmmT.baum_welch(training_seqs);
	}
}

int main(void) {
	converge();
	return 0;
}