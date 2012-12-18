#ifndef __pslremissions_hpp__
#define __pslremissions_hpp__

#include <map>
#include <vector>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>

#include "sparse_vector.hpp"

namespace pslrhmm {
	
	template<typename A>
	class DiscreteEmissions {
		SparseVector<A> emissions_probs;

	public:
		typedef A Emission;

		template<typename Random>
		void initRandom(Random& r, const std::vector<A>& alphabet) {
			boost::uniform_01<> u;

			emissions_probs.clear();
			BOOST_FOREACH(auto e, alphabet) {
				emissions_probs[e] = u(r);
			}
			emissions_probs.normalize();
		}

		void initUniform(const std::vector<A>& alphabet) {
			const double einv = 1.0l / alphabet.size();
			emissions_probs.clear();
			BOOST_FOREACH(auto e, alphabet) {
				emissions_probs[e] = einv;
			}
			emissions_probs.normalize();
		}

		template<typename Random>
		A generateEmission(Random& r) const {
			return emissions_probs.select(r);
		}

		double likelihood(A a) const {
			return emissions_probs.get(a);
		}

		void push(A a, double prob) {
			emissions_probs[a] += prob;
		}
		void computeDistribution() {
			emissions_probs.normalize();
		}
	};
}

#endif //__pslremissions_hpp__