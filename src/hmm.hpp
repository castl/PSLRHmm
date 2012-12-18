#ifndef __pslrhmm_hpp__
#define __pslrhmm_hpp__

#include <map>
#include <vector>
#include <boost/random/mersenne_twister.hpp>

#include "sparse_vector.hpp"
#include "matrix.hpp"

namespace pslrhmm {
	template <typename E>
	class HMM;

	template<typename E>
	class StateTempl {
		friend class HMM<E>;
		HMM<E>& owner;

		SparseVector<E> emissions_probs;
		SparseVector<const StateTempl*> transition_probs;

		StateTempl(HMM<E>& hmm) : owner(hmm) { }

	public:
		template<typename Random>
		E generateEmission(Random& r) const {
			return emissions_probs.select(r);
		}

		template<typename Random>
		const StateTempl*    generateState(Random& r)    const {
			return transition_probs.select(r);
		}
	};

	template<typename E = uint64_t>
	class HMM {
		typedef StateTempl<E> State;

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

		double B(const State* s, E e) const {
			return s->emissions_probs.get(e);	
		}
		double B(size_t s, E e) const {
			return B(states[s], e);
		}

		double Pi(const State* s) const {
			return init_prob.get((State*)s);
		}
		double Pi(size_t s) const {
			return init_prob.get(states[s]);
		}

	public:
		typedef E Emission;
		typedef std::vector<E> Sequence;
		typedef boost::mt19937 Random;

		HMM() { }

		void initRandom(Random& r, size_t states, std::vector<E> alphabet);
		void initUniform(size_t states, std::vector<E> alphabet);

		const State* generateInitialState(Random& r) const {
			return init_prob.select(r);
		}
		void generateSequence(Random& r, Sequence&, size_t length) const;
		double calcSequenceLikelihoodLog(const Sequence&) const;
		double calcSequenceLikelihoodLogNorm(const Sequence& s) const {
			return calcSequenceLikelihoodLog(s) / s.size();
		}
		double calcSequenceLikelihood(const Sequence& s) const {
			return std::exp(calcSequenceLikelihoodLog(s));
		}
		double calcSequenceLikelihoodNorm(const Sequence& s) const {
			return std::exp(calcSequenceLikelihoodLogNorm(s));
		}

		double calcSequenceLikelihoodLog(const std::vector<Sequence>&) const;
		double calcSequenceLikelihood(const std::vector<Sequence>& s) const {
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

#include "hmm_implementation.hpp"

#endif // __pslrhmm_hpp__