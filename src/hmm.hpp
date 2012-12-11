#ifndef __pslrhmm_hpp__
#define __pslrhmm_hpp__

#include <map>
#include <vector>
#include <boost/random/mersenne_twister.hpp>

#include "sparse_vector.hpp"

namespace pslrhmm {

	class Emission {
	public:
		virtual bool operator=(const Emission& e) const = 0;
		virtual int  operator>(const Emission& e) const = 0;
	};

	class IntEmission : public Emission {
		uint64_t i;
	public:
		IntEmission(uint64_t i) : i(i) { }

		virtual bool operator=(const Emission& e) const {
			return this->i == dynamic_cast<const IntEmission&>(e).i;
		}
		virtual int  operator>(const Emission& e) const {
			return this->i > dynamic_cast<const IntEmission&>(e).i;
		}
	};

	class HMM;
	class State {
		friend class HMM;
		HMM& owner;

		SparseVector<const Emission*>  emissions_probs;
		SparseVector<State*> 		   transition_probs;

		State(HMM& hmm) : owner(hmm) { }

	public:
		const Emission* generateEmission() const;
		const State*    generateState()    const;
	};

	class HMM {
		friend class State;

		std::vector<State*>  states;
		SparseVector<State*> init_prob;

	public:
		typedef std::vector<const Emission*> Sequence;
		typedef boost::random::mt19937 Random;

		HMM() { }

		void initRandom(Random& r, size_t states, std::vector<Emission*> alphabet);
		void train(size_t states, std::vector<Sequence> sequences);

		const State* generateInitialState(Random& r) const;
		void generateSequence(Random& r, Sequence&, size_t) const;
		double calcSequenceLikihoodLog(Sequence) const;

		State& operator[](size_t i) {
			assert(i < states.size());
			return *states[i];
		}

		const State& operator[](size_t i) const {
			assert(i < states.size());
			return *states[i];
		}
	};
}

#endif // __pslrhmm_hpp__