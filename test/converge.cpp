#include <hmm.hpp>
#include <hmm_pop_trainer.hpp>

using namespace pslrhmm;
using namespace std;

#define NUM_SEQ 1000
#define NUM_STATES 4
#define ITERS 50 


typedef MVNormalEmissions Emission;
typedef HMM<Emission> MyHMM;
#define DIMS 3 

template<typename E>
void initAlphabet(vector<E>& alpha) {
	alpha.push_back(arma::zeros<arma::vec>(DIMS));
}

void converge(void) {
	setlinebuf(stdout);

	Emission ex(DIMS);
	vector<MyHMM::Emission> alphabet;
	initAlphabet(alphabet);

	MyHMM::Random r(time(NULL));
	MyHMM hmm1, hmm2;
	hmm1.initRandom(r, NUM_STATES, alphabet);
	hmm2.initRandom(r, NUM_STATES, alphabet);

	vector<MyHMM::Sequence> training_seqs;
	for(size_t i=0; i<NUM_SEQ; i++) {
		MyHMM::Sequence s1;
		hmm1.generateSequence(r, s1, 400);
		training_seqs.push_back(s1);
	}

	vector<MyHMM::Sequence> testing_seqs1;
	vector<MyHMM::Sequence> testing_seqs2;
	for(size_t i=0; i<NUM_SEQ; i++) {
		MyHMM::Sequence s1, s2;
		hmm1.generateSequence(r, s1, 400);
		hmm2.generateSequence(r, s2, 400);
		testing_seqs1.push_back(s1);
		testing_seqs2.push_back(s2);
	}

	// MyHMM hmmT;
	PopulationTrainer<MyHMM> pt(250);
	pt.initRandom(r, NUM_STATES, alphabet);
	// hmmT.initUniform(NUM_STATES, ex);

	printf("    Train          Test         Test (Hmm2)\n");
	for (size_t i=0; i<ITERS; i++) {
		MyHMM& hmmT = pt.best();
		printf("%lu: %le %le %le\n", i,
			hmmT.calcSequenceLikelihoodNorm(training_seqs),
			hmmT.calcSequenceLikelihoodNorm(testing_seqs1),
			hmmT.calcSequenceLikelihoodNorm(testing_seqs2));

		pt.baum_welch(training_seqs);
	}

	printf("\n=== HMM2 === \n");
	hmm2.print();

	printf("\n=== HMM1 === \n");
	hmm1.print();

	printf("\n=== HMMT === \n");
	pt.best().print();
}

int main(void) {
	converge();
	return 0;
}