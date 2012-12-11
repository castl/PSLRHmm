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
}