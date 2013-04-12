#include "hmm.hpp"

#include <cmath>
#include <limits>
#include <boost/random/uniform_01.hpp>

using namespace std;
using namespace pslrhmm;

namespace pslrhmm {

	template<typename E>
	void HMM<E>::initRandom(Random& r, size_t states, std::vector<Emission> alphabet) {
		boost::uniform_01<> u;

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
			s->emissions_probs.initRandom(r, alphabet);
		}
	}

	template<typename E>
	void HMM<E>::initUniform(size_t states, std::vector<Emission> alphabet) {
		E emi;
		BOOST_FOREACH(auto e, alphabet) {
			emi.push(e, 1.0);
		}
		emi.computeDistribution();
		initUniform(states, emi);
	}

	template<typename E>
	void HMM<E>::initUniform(size_t states, E emi) {
		double sinv = 1.0l / states;
		this->states.clear();
		this->init_prob.clear();

		for (size_t i=0; i<states; i++) {
			State* s = new State(*this);
			this->states.push_back(s);
			this->init_prob[s] = sinv;
		}
		this->init_prob.normalize();

		BOOST_FOREACH(auto s, this->states) {
			BOOST_FOREACH(auto t, this->states) {
				s->transition_probs[t] = sinv;
			}
			s->transition_probs.normalize();
			s->emissions_probs = emi;
		}
	}

	template<typename E>
	void HMM<E>::generateSequence(Random& r, Sequence& seq, size_t len) const {
		const State* s = this->generateInitialState(r);
		for (size_t i=0; i<len; i++) {
			seq.push_back(s->generateEmission(r));
			s = s->generateState(r);
		}
	}

	struct LogDouble {
		class NegativeNumberException : public exception {
		public:
			double d;
			NegativeNumberException(double d) : d(d) { }
		};

		double l;

		LogDouble() : l(-std::numeric_limits<double>::infinity()) { }
		LogDouble(double d) {
			(*this) = d;
		}

		void operator=(double d) {
			assert(!std::isnan(d));
			assert(!std::isinf(d));
			if (d == 0)
				l = -numeric_limits<double>::infinity();
			else if (d > 0.0)
				l = std::log(d);
			else 
				throw NegativeNumberException(d);
		}

		LogDouble operator+(double d) const {
			return (*this) + LogDouble(d);
		}

		LogDouble operator+(LogDouble y) const {
			LogDouble x = *this;
			LogDouble r;
			if (std::isinf(x.l))
				return y;
			else if (std::isinf(y.l))
				return x;
			else if (x.l > y.l) {
				r.l = x.l + std::log(1 + std::exp(y.l - x.l));
			} else {
				r.l = y.l + std::log(1 + std::exp(x.l - y.l));
			}

			return r;
		}

		LogDouble operator*(double d) const {
			LogDouble o = d;
			return (*this) * o;
		}

		LogDouble operator*(LogDouble d) const {
			LogDouble r;
			r.l = this->l + d.l;
			return r;
		}

		LogDouble operator/(LogDouble d) const {
			LogDouble r;
			r.l = this->l - d.l;
			return r;
		}

		double exp() const {
			if (std::isinf(this->l))
				return 0.0;
			return std::exp(this->l);
		}

		double log() const {
			return this->l;
		}
	};

	template<typename E>
	double HMM<E>::calcSequenceLikelihoodLog(const Sequence& seq) const {
		const size_t num_states = states.size();
		vector<LogDouble> alpha_old(num_states);
		vector<LogDouble> alpha    (num_states);

		// Initialization
		Emission e = seq[0];
		// LogDouble tot = 0.0;
		for (size_t i=0; i<num_states; i++) {
			const State* s = states[i];
			alpha[i] = Pi(s)*B(s, e);
			// tot = tot + alpha[i];
		}
		// assert(tot.exp() > 0.0);

		for (size_t oi=1; oi<seq.size(); oi++) {
			// tot = 0.0;
			e = seq[oi];
			alpha_old.swap(alpha);

			for (size_t i=0; i<num_states; i++) {
				const State* s = states[i];

				LogDouble t = 0.0;
				for (size_t j=0; j<num_states; j++) {
					t = t + alpha_old[j] * A(j, s);
				}
				alpha[i] = t * B(s, e);
				// tot = tot + alpha[i];
			}
			// assert(tot.exp() > 0.0);
		}

		LogDouble sum = 0.0;
		for (size_t i=0; i<num_states; i++) {
			sum = sum + alpha[i];
		}
		return sum.log();	
	}

	template<typename E>
	double HMM<E>::calcSequenceLikelihoodLog(const vector<Sequence>& seqs) const {
		LogDouble sum;
		#pragma omp parallel for \
			default(none) shared(seqs, sum)
		for (size_t i=0; i<seqs.size(); i++) {
			const Sequence& s = seqs[i];
			LogDouble d;
			d.l = calcSequenceLikelihoodLog(s);

			#pragma omp critical (acc_sum)
			{
				sum = sum + d;
			}
		}
		LogDouble avg = sum / seqs.size();
		return avg.log();
	}

	template<typename E>
	double HMM<E>::calcSequenceLikelihoodNorm(const vector<Sequence>& seqs) const {
		double sum;
		#pragma omp parallel for \
			default(none) shared(seqs, sum)
		for (size_t i=0; i<seqs.size(); i++) {
			const Sequence& s = seqs[i];
			double d = calcSequenceLikelihoodNorm(s);

			#pragma omp critical (acc_sum)
			{
				sum = sum + d;
			}
		}
		double avg = sum / seqs.size();
		return avg;
	}

	template<typename E>
	void HMM<E>::forward_scaled(const Sequence& seq,
				Matrix<double>& alpha, std::vector<double>& c) const {
		const size_t T = seq.size();
		const size_t num_states = states.size();
		alpha.resize(T, num_states);
		c.resize(T);

		for (size_t i=0; i<num_states; i++) {
			alpha(0, i) = Pi(i) * B(i, seq[0]);
			assert(!std::isnan(alpha(0, i)) && !std::isinf(alpha(0,i)));
		}
		c[0] = 1.0l / alpha.xSlice(0).sum();
		assert(alpha.xSlice(0).sum() != 0.0);
		assert(!std::isnan(c[0]) && !std::isinf(c[0]));

		for (size_t t=1; t<T; t++) {
			for (size_t i=0; i<num_states; i++) {
				const State* si = states[i];
				double sum = 0.0;
				for (size_t j=0; j<num_states; j++) {
					sum += alpha(t - 1, j) * A(j, si);
				}
				alpha(t, i) = sum * B(i, seq[t]);
				assert(!std::isnan(alpha(t, i)) && !std::isinf(alpha(t,i)));
			}

			auto slice = alpha.xSlice(t);
			double ct = 1.0l / slice.sum();
			c[t] = ct;
			slice *= ct;

			assert(!std::isnan(c[t]) && !std::isinf(c[t]));
		}
	}

	template<typename E>
	void HMM<E>::backward_scaled(const Sequence& seq,
				const Matrix<double>& alpha, const std::vector<double>& c,
				Matrix<double>& beta) const {
		const size_t T = seq.size();
		const size_t num_states = states.size();
		assert(c.size() == T);	

		beta.resize(T, num_states);
		for (size_t i=0; i<num_states; i++) {
			size_t t = T-1;
			beta(t, i) = 1 * c[t];
			assert(!std::isnan(c[t]) && !std::isinf(c[t]));
			assert(!std::isnan(beta(t, i)) && !std::isinf(beta(t,i)));
		}

		for (int64_t t = T - 2; t >= 0; t--) {
			double ct = c[t];
			for (size_t i=0; i<num_states; i++) {
				double sum = 0.0;
				assert(!std::isnan(c[t]) && !std::isinf(c[t]));
				for (size_t j=0; j<num_states; j++) {
					sum += A(i, j) * B(j, seq[t+1]) * beta(t+1, j);
				}
				beta(t, i) = sum * ct;
				assert(!std::isnan(beta(t, i)) && !std::isinf(beta(t,i)));
			}	
		}
	}

	template<typename E>
	void HMM<E>::baum_welch(const vector<Sequence>& sequences) {
		const size_t num_states = states.size();

		Matrix<double> ahat_numerator(num_states, num_states);
		Matrix<double> ahat_denominator(num_states, num_states);

		// Indexed by state number
		vector< E > bhat(num_states);

		#pragma omp parallel for \
			shared(sequences, ahat_numerator, ahat_denominator, bhat) \
			schedule(dynamic)
		for (size_t l=0; l<sequences.size(); l++) {
			const Sequence& seq = sequences[l];
			const size_t T = seq.size();
			Matrix<double> alpha;
			std::vector<double> c;
			Matrix<double> beta;

			this->forward_scaled(seq, alpha, c);
			this->backward_scaled(seq, alpha, c, beta);

			// Updates for ahat
			for (size_t i=0; i<num_states; i++) {
				const State* si = states[i];
				for (size_t j=0; j<num_states; j++) {
					const State* sj = states[j];
					double numerator_sum = 0.0;
					double demoninator_sum = 0.0;
					for (size_t t=0; t<(T - 1); t++) {
						numerator_sum += 
							alpha(t, i) * A(si, sj) * B(sj, seq[t+1]) * beta(t + 1, j);
						demoninator_sum +=
							alpha(t, i) * beta(t, i) / c[t];
					}

					double& an = ahat_numerator(i, j);
					double& ad = ahat_denominator(i, j);
					#pragma omp atomic
					an += numerator_sum;
					#pragma omp atomic
					ad += demoninator_sum;
				}
			}

			// Updates for bhat
			#pragma omp critical(bn_update)
			for (size_t j=0; j<num_states; j++) {
				E& bh = bhat[j];

				for (size_t t=0; t<T; t++) {
					Emission e = seq[t];
					double prob = alpha(t, j) * beta(t, j) / c[t];
					assert(!std::isnan(c[t]) && !std::isinf(c[t]));
					assert(!std::isnan(alpha(t, j)) && !std::isinf(alpha(t,j)));
					assert(!std::isnan(beta(t, j)) && !std::isinf(beta(t,j)));
					assert(c[t] != 0.0);
					assert(!std::isnan(prob));
					assert(!std::isinf(prob));

					bh.push(e, prob);
				}
			}
		}

		// Calculate final a, b
		for (size_t i=0; i<num_states; i++) {
			State* si = states[i];
			for (size_t j=0; j<num_states; j++) {
				State* sj = states[j];

				double Aij = ahat_numerator(i, j) / ahat_denominator(i, j);
				si->transition_probs.set(sj, Aij);
			}
			si->transition_probs.normalize();

			E& bn = bhat[i];
			bn.computeDistribution();
			si->emissions_probs = bn;
		}
	}


	template<typename E>
	void HMM<E>::print(FILE* st) const {
		const size_t num_states = states.size();

		// First print transition matrix
		fprintf(st, "     ");
		for (size_t i=0; i<num_states; i++) {
			fprintf(st, "%4lu ", i);
		}
		fprintf(st, "\n");

		for (size_t i=0; i<num_states; i++) {
			fprintf(st, "%2lu   ", i);
			for (size_t j=0; j<num_states; j++) {
				fprintf(st, "%0.2lf ", A(i, j));
			}
			fprintf(st, "\n");
		}

		for (size_t i=0; i<num_states; i++) {
			fprintf(st, "  -- %4lu \n", i);
			states[i]->emissions_probs.print(st);
		}
	}
}