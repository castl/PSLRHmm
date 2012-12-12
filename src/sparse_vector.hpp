#ifndef __pslrhmm_sparse_vector_hpp__
#define __pslrhmm_sparse_vector_hpp__

#include <unordered_map>
#include <cassert>
#include <boost/foreach.hpp>
#include <boost/random/uniform_01.hpp>

namespace pslrhmm {

	template<typename K>
	class SparseVector {
		typedef std::unordered_map<K, double> Map;
		Map vec;

	public:
		// typedef typename Map::type type;
		typedef typename Map::iterator iterator;
		typedef typename Map::const_iterator const_iterator;

		void normalize() {
			double total = 0.0;
			BOOST_FOREACH(auto p, vec) {
				total += p.second;
			}

			BOOST_FOREACH(auto& p, vec) {
				p.second /= total;
			}
		}

		// Select an item at random based on the this vector's discrete distribution
		template<typename Random>
		K select(Random& r) const {
			boost::uniform_01<> u;
			double v = u(r);
			BOOST_FOREACH(auto p, vec) {
				v -= p.second;
				if (v <= 0.0)
					return p.first;
			}
			assert(false && "Internal error: selection math didn't work");
		}

		void clear() {
			vec.clear();
		}

		double& operator[](const K k) {
			return vec[k];
		}

		void set(const K k, double d) {
			if (d == 0.0) {
				vec.erase(k);
			} else {
				vec[k] = d;
			}
		}

		double operator[](const K k) const {
			auto f = vec.find(k);
			if (f == vec.end())
				return 0.0;
			return f->second;
		}

		double get(const K k) const {
			auto f = vec.find(k);
			if (f == vec.end())
				return 0.0;
			return f->second;
		}

		iterator begin() {
			return vec.begin();
		}

		iterator end() {
			return vec.end();
		}

		const_iterator begin() const {
			return vec.begin();
		}

		const_iterator end() const {
			return vec.end();
		}
	};
}

#endif // __pslrhmm_sparse_vector_hpp__