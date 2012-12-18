#ifndef __pslremissions_hpp__
#define __pslremissions_hpp__

#include <map>
#include <vector>
#include <algorithm>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real.hpp>

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
		typedef boost::normal_distribution<double> Distribution;
		Distribution dist;
		std::vector<std::pair<double, double> > seen;

	public:
		typedef double Emission;
		NormalEmissions() : dist(0, 1.0) { }
		NormalEmissions(double mean, double stdev) : dist(mean, stdev) { }

		template<typename Random>
		void initRandom(Random& r, const std::vector<Emission>& alphabet) {
			boost::uniform_01<Emission> u01;
			boost::uniform_real<Emission> ur(-10, 10);
			dist = Distribution(ur(r), u01(r));
		}

		template<typename Random>
		Emission generateEmission(Random& r) const {
			boost::variate_generator<Random, Distribution > generator(r, dist);
			Emission e = generator();
			assert(!std::isnan(e));
			assert(!std::isinf(e));
			return e;
		}

		double likelihood(Emission a) const {
			double sd = dist.sigma();
			double mean = dist.mean();
			double z = (a - mean) / sd;
			double exponent = -0.5 * z * z;
			assert(exponent < 0.0);
			double l = std::exp(exponent);
			// Don't divide -- this normalizes to max == 1

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

			dist = Distribution(mean, stdev);
		}
	};

	// Uniform emissions don't make much sense, especially learning.
	// Don't use them except for testing
	class UniformEmissions {
		typedef boost::uniform_real<double> Distribution;
		Distribution dist;
		std::vector<std::pair<double, double> > seen;

	public:
		typedef double Emission;
		UniformEmissions() : dist(0, 0.0) { }
		UniformEmissions(double mmin, double mmax) : dist(mmin, mmax) { }

		template<typename Random>
		void initRandom(Random& r, const std::vector<Emission>& alphabet) {
			boost::uniform_real<Emission> u1(0, 100.0);
			boost::uniform_real<Emission> u2(0, 50.0);
			double m = u1(r);
			dist = Distribution(m, m + u2(r));
		}

		template<typename Random>
		Emission generateEmission(Random& r) const {
			boost::variate_generator<Random, Distribution > generator(r, dist);
			Emission e = generator();
			assert(!std::isnan(e));
			assert(!std::isinf(e));
			return e;
		}

		double likelihood(Emission a) const {
			if (a >= dist.min() && a <= dist.max()) {
				return 1.0;
			} else {
				return 0.001;
			}
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

			dist = Distribution(mean - 1.5*stdev, mean + 1.5*stdev);
		}
	};
}

#endif //__pslremissions_hpp__