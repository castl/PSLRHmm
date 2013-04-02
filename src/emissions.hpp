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

#include <mlpack/core/dists/gaussian_distribution.hpp>

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
	public:
		typedef double Emission;

	private:
		typedef mlpack::distribution::GaussianDistribution Distribution;
		Distribution dist;

		std::vector<Emission> observations;
		std::vector<double> probabilities;

	public:
		NormalEmissions() : dist(1) { }
		NormalEmissions(double mean, double stdev) :
			dist(arma::vec({mean}), arma::mat(arma::vec({stdev}))) {
		}

		template<typename Random>
		void initRandom(Random& r, const std::vector<Emission>& alphabet) {
			boost::uniform_01<Emission> u01;
			boost::uniform_real<Emission> ur(-25, 25);
			dist = Distribution(arma::vec({ur(r)}), arma::mat(arma::vec({u01(r)})));
		}

		template<typename Random>
		Emission generateEmission(Random& r) const {
			return dist.Random()[0];
		}

		double likelihood(Emission a) const {
			return dist.Probability(arma::vec({a}));
		}

		void push(Emission a, double prob) {
			observations.push_back(a);
			probabilities.push_back(prob);
		}

		void computeDistribution() {
			assert(probabilities.size() == observations.size());
			arma::mat obs(1, observations.size());
			arma::vec probs(probabilities.size());

			size_t sz = probabilities.size();
			for (size_t i = 0; i < sz; i++) {
				obs(0, i) = observations[i];
				probs[i] = probabilities[i];
			}

			dist.Estimate(obs, probs);

			observations.clear();
			probabilities.clear();
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