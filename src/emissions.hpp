#ifndef __pslremissions_hpp__
#define __pslremissions_hpp__

#include <map>
#include <vector>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/random/variate_generator.hpp>


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

	class NormalEmissions {
		typedef boost::math::normal_distribution<double> Distribution;
		Distribution nd;
		std::vector<std::pair<double, double> > seen;

	public:
		typedef double Emission;
		NormalEmissions() : nd(0, 1.0) { }
		NormalEmissions(double mean, double stdev) : nd(mean, stdev) { }

		template<typename Random>
		void initRandom(Random& r, const std::vector<Emission>& alphabet) {
			boost::uniform_01<> u;
			nd = Distribution(u(r), u(r));
		}

		template<typename Random>
		Emission generateEmission(Random& r) const {
			boost::variate_generator<Random, boost::normal_distribution<Emission> >
			    generator(r, boost::normal_distribution<Emission>() );
			Emission e = generator();
			assert(!std::isnan(e));
			assert(!std::isinf(e));
			return e;
		}

		double likelihood(Emission a) const {
			// boost::math::pdf<normal_distribution> is broken
			// double l = boost::math::pdf(nd, a);

			double sd = nd.standard_deviation();
			double mean = nd.mean();
			double exponent = (a - mean) / sd;
			exponent = -0.5 * exponent * exponent;
			double l = std::exp(exponent);
			l /= sqrt(2 * boost::math::constants::pi<double>());

			assert(l >= 0.0);
			assert(l <= 1.0);
			return l;
		}

		void push(Emission a, double prob) {
			seen.push_back(std::pair<double, double>(a, prob));
		}

		void computeDistribution() {
			double sum = 0.0;
			double total = 0.0;
			BOOST_FOREACH(auto p, seen) {
				sum += p.first * p.second;
				total += p.second;
			}
			double mean = sum / total;

			double numerator = 0.0;
			BOOST_FOREACH(auto p, seen) {
				numerator += p.second * pow(p.first - mean, 2);
			}
			double stdev = sqrt(numerator / total);
			seen.clear();

			nd = Distribution(mean, stdev);
		}
	};
}

#endif //__pslremissions_hpp__