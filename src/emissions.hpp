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

		void print(FILE* st) const {
		}
	};

	const static double Epsilon = 1e-9;
	class NormalEmissions {
	public:
		typedef double Emission;

	private:
		typedef mlpack::distribution::GaussianDistribution Distribution;
		Distribution dist;
		double max_prob;
		double scale_factor;

		std::vector<Emission> observations;
		std::vector<double> probabilities;

		void compute_max_prob() {
			// The max value of the normal PDF:
			max_prob = dist.Probability(dist.Mean());
			// Scale to get a rational volume
			scale_factor = pow(2*M_PI, dist.Dimensionality() / -2.0) / max_prob;
		}

	public:
		NormalEmissions() :
			dist(arma::vec({1.0}), arma::mat(arma::vec({1.0}))) {
		}
		NormalEmissions(double mean, double stdev) :
			dist(arma::vec({mean}), arma::mat(arma::vec({stdev * stdev}))) {
			compute_max_prob();
		}

		template<typename Random>
		void initRandom(Random& r, const std::vector<Emission>& alphabet) {
			boost::uniform_01<Emission> u01;
			boost::uniform_real<Emission> ur(-25, 25);
			dist = Distribution(arma::vec({ur(r)}), arma::mat(arma::vec({u01(r)})));
			compute_max_prob();
		}

		template<typename Random>
		Emission generateEmission(Random& r) const {
			return dist.Random()[0];
		}

		double likelihood(Emission a) const {
			double l = dist.Probability(arma::vec({a}));
			assert(!std::isnan(l));
			assert(!std::isinf(l));

			assert(l >= 0.0);
			assert(l <= (max_prob + 0.00001) );
			return std::max(Epsilon, l * scale_factor);
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
			compute_max_prob();

			observations.clear();
			probabilities.clear();
		}

		void print(FILE* st) const {
		}
	};

	class MVNormalEmissions {
	public:
		typedef arma::vec Emission;
		typedef mlpack::distribution::GaussianDistribution Distribution;

	private:
		Distribution dist;
        bool ismutable;
		double max_prob;
		double scale_factor;

		std::vector<Emission> observations;
		std::vector<double> probabilities;

		void compute_max_prob() {
			// The max value of the normal PDF:
			max_prob = dist.Probability(dist.Mean());
			assert(!std::isnan(max_prob));
			// Scale to get a rational volume
			scale_factor = pow(2*M_PI, dist.Dimensionality() / -2.0) / max_prob;
		}

	public:
		MVNormalEmissions(const MVNormalEmissions& other) {
			this->dist = other.dist;
			this->ismutable = other.ismutable;
			compute_max_prob();
		}

		MVNormalEmissions(size_t dims) :
			dist(dims),
            ismutable(true) {
			compute_max_prob();
		}

        MVNormalEmissions(Distribution dist, bool ismutable = true) :
            dist(dist),
            ismutable(ismutable) {
            compute_max_prob();
        }

		MVNormalEmissions() :
			dist(),
            ismutable(true) {
			max_prob = -1.0;
		}

		template<typename Random>
		void initRandom(Random& r, const std::vector<Emission>& alphabet) {
			assert(alphabet.size() > 0 && "Needs an example!");
			size_t dims = alphabet[0].n_elem;
			arma::vec means(dims);
			arma::mat cov = arma::eye<arma::mat>(dims, dims);

			boost::uniform_01<double> u01;
			boost::uniform_real<double> ur(-25, 25);
			for (size_t d=0; d<dims; d++) {
				means[d] = ur(r);
				cov(d, d) = u01(r);
			}
			dist = Distribution(means, cov);
			compute_max_prob();
		}

		template<typename Random>
		Emission generateEmission(Random& r) const {
			assert(max_prob != -1.0);
			return dist.Random();
		}

		double likelihood(Emission a) const {
			assert(max_prob != -1.0);
			double l = dist.Probability(a);
			if (std::isnan(l)) {
				l = max_prob * 1e-10;
			} 
			assert(!std::isinf(l));

			assert(l >= 0.0);
			assert(l <= (max_prob + 0.00001) );
			return std::max(Epsilon, l * scale_factor);
		}

		void push(Emission a, double prob) {
            if (!ismutable)
                return;
			observations.push_back(a);
			probabilities.push_back(prob);
		}

		void computeDistribution() {
            if (!ismutable)
                return;
			assert(probabilities.size() == observations.size());
			assert(observations.size() > 0);
			size_t dims = observations[0].n_elem;
			arma::mat obs(dims, observations.size());
			arma::vec probs(probabilities.size());

			size_t sz = probabilities.size();
			for (size_t i = 0; i < sz; i++) {
				for (size_t d=0; d<dims; d++) {
					obs(d, i) = observations[i][d];
				}
				probs[i] = probabilities[i];
			}

			dist.Estimate(obs, probs);
			compute_max_prob();

			observations.clear();
			probabilities.clear();
		}

		void print(FILE* st) const {
			arma::vec means = dist.Mean();
			arma::mat cov = dist.Covariance();

			if (!ismutable) {
				fprintf(st, "    immutable\n");
			}
			fprintf(st, "      ");
			for (size_t d=0; d<means.n_elem; d++) {
				fprintf(st, "%0.2lf ", means[d]);
			}
			fprintf(st, "\n\n");

			for (size_t d1=0; d1<cov.n_rows; d1++) {
				fprintf(st, "      ");
				for (size_t d=0; d<cov.n_cols; d++) {
					fprintf(st, "%0.2lf ", cov(d1, d));
				}
				fprintf(st, "\n");
			}
			fprintf(st, "\n");
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

		void print(FILE* st) const {
		}
	};
}

#endif //__pslremissions_hpp__
