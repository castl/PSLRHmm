#ifndef __pslrhmm_sparse_vector_hpp__
#define __pslrhmm_sparse_vector_hpp__

#include <map>
#include <boost/foreach.hpp>

namespace pslrhmm {

	template<typename K>
	class SparseVector {
		std::map<K, double> vec;

	public:
		void normalize() {
			double total = 0.0;
			BOOST_FOREACH(auto p, vec) {
				total += p.second;
			}

			BOOST_FOREACH(auto& p, vec) {
				p.second /= total;
			}
		}
	};
}

#endif // __pslrhmm_sparse_vector_hpp__