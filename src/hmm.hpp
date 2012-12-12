#ifndef __pslrhmm_hpp__
#define __pslrhmm_hpp__

#include <map>
#include <vector>
#include <boost/random/mersenne_twister.hpp>

#include "sparse_vector.hpp"
#include "matrix.hpp"

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
		SparseVector<const State*> 	   transition_probs;

		State(HMM& hmm) : owner(hmm) { }

	public:
		template<typename Random>
		const Emission* generateEmission(Random& r) const {
			return emissions_probs.select(r);
		}

		template<typename Random>
		const State*    generateState(Random& r)    const {
			return transition_probs.select(r);
		}
	};

	class HMM {
		friend class State;

		std::vector<State*>  states;
		SparseVector<State*> init_prob;

		/****
			Aliases to make matrix-like things with math names
		****/

		double A(const State* s, const State* t) const {
			return s->transition_probs.get(t);	
		}
		double A(const State* i, size_t j) const {
			return A(i, states[j]);
		}
		double A(size_t i, const State* j) const {
			return A(states[i], j);
		}
		double A(size_t i, size_t j) const {
			return A(states[i], states[j]);
		}

		double B(const State* s, const Emission* e) const {
			return s->emissions_probs.get(e);	
		}
		double B(size_t s, const Emission* e) const {
			return B(states[s], e);
		}

		double Pi(const State* s) const {
			return init_prob.get((State*)s);
		}
		double Pi(size_t s) const {
			return init_prob.get((State*)states[s]);
		}

	public:
		typedef std::vector<const Emission*> Sequence;
		typedef boost::mt19937 Random;

		HMM() { }

		void initRandom(Random& r, size_t states, std::vector<Emission*> alphabet);
		void initUniform(size_t states, std::vector<Emission*> alphabet);

		const State* generateInitialState(Random& r) const;
		void generateSequence(Random& r, Sequence&, size_t length) const;
		double calcSequenceLikelihoodLog(const Sequence&) const;
		double calcSequenceLikelihood(const Sequence& s) const {
			return std::exp(calcSequenceLikelihoodLog(s));
		}

		void forward_scaled(const Sequence&,
					Matrix<double>& alpha, std::vector<double>& c) const;
		void backward_scaled(const Sequence&,
					const Matrix<double>& alpha, const std::vector<double>& c,
					Matrix<double>& beta) const;
		void baum_welch(const std::vector<Sequence>& sequences);

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