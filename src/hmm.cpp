#include "hmm.hpp"

#include <boost/random/uniform_01.hpp>

using namespace std;

namespace pslrhmm {
	void HMM::initRandom(Random& r, size_t states, std::vector<Emission*> alphabet) {
		boost::random::uniform_01<> u;

		this->states.clear();
		this->init_prob.clear();

		for (size_t i=0; i<states; i++) {
			State* s = new State(*this);
			this->states.push_back(s);
			this->init_prob[s] = u(r);
		}

		this->init_prob.normalize();


		BOOST_FOREACH(auto s, this->states) {
			BOOST_FOREACH(auto t, this->states) {
				s->transition_probs[t] = u(r);
			}
			s->transition_probs.normalize();

			BOOST_FOREACH(auto e, alphabet) {
				s->emissions_probs[e] = u(r);
			}
			s->emissions_probs.normalize();
		}
	}

	void HMM::generateSequence(Random& r, Sequence& seq, size_t len) const {
		const State* s = this->generateInitialState(r);
		for (size_t i=0; i<len; i++) {
			seq.push_back(s->generateEmission(r));
			s = s->generateState(r);
		}
	}

	const State* HMM::generateInitialState(Random& r) const {
		return init_prob.select(r);
	}


	double HMM::calcSequenceLikihoodLog(const Sequence& seq) const {
		const size_t num_states = states.size();
		vector<double> alpha_old(num_states);
		vector<double> alpha    (num_states);

		// Initialization
		const Emission* e = seq[0];
		for (size_t i=0; i<num_states; i++) {
			const State* s = states[i];
			alpha[i] = Pi(s)*B(s, e);
		}

		for (size_t oi=1; oi<seq.size(); oi++) {
			e = seq[oi];
			alpha_old.swap(alpha);

			for (size_t i=0; i<num_states; i++) {
				const State* s = states[i];

				double t = 0.0;
				for (size_t j=0; j<num_states; j++) {
					t += alpha_old[j] * A(j, s);
				}
				alpha[i] = t * B(s, e);
			}
		}

		double sum = 0.0;
		for (size_t i=0; i<num_states; i++) {
			sum += alpha[i];
		}
		return log(sum);	
	}
}