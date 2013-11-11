// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_FACTORY_H_
#define HYPERGRAPH_FACTORY_H_

#include <map>
#include <string>
#include "Hypergraph/Hypergraph.h"
#include "./common.h"

#define REGISTER_TYPE_DECLARATION(NAME) \
    static DerivedRegister<NAME> reg

#define REGISTER_TYPE_DEFINITION(NAME) \
    DerivedRegister<NAME> NAME::reg(#NAME)

#define REGISTER_TYPE_REDEFINE(NAME) \
    NAME::reg = DerivedRegister<NAME>(#NAME)

class BaseSemiring;
typedef BaseSemiring* (*create_random_fnptr)();
typedef std::map<std::string, create_random_fnptr> RegistryMap;
typedef std::pair<std::string, create_random_fnptr> RegistryPair;

class BaseSemiringFactory {
	static RegistryMap* registry;

protected:
    static RegistryMap * getMap() { 
    	registry = !registry ? new RegistryMap : registry;
        return registry; 
    }

public:
	virtual ~BaseSemiringFactory() { };

	// static BaseSemiring * create_from_string(std::string name) {
	// 	RegistryMap::const_iterator it = registry->find(name);
	// 	return it == registry->end() ? NULL : it->second();
	// }

	// static void register_class(std::string name, create_random_fnptr f) {
	// 	(*registry)[name] = f;
	// }

	static const vector<create_random_fnptr> retrieve_classes() {
		vector<create_random_fnptr> creators;
        // std::cerr << getMap()->size() << std::endl;
		foreach(RegistryPair pr, *registry) {
			creators.push_back(pr.second);
		}
		return creators;
	}
};

template<typename T> BaseSemiring* createT() { return new T(T::randValue()); }
template<typename T> BaseSemiring* createRandomT() { return new T(T::randValue()); }

template<typename T>
struct DerivedRegister : BaseSemiringFactory { 
    DerivedRegister(std::string const& s) { 
        getMap()->insert(std::make_pair(s, &createRandomT<T>));
    }
};

#endif // HYPERGRAPH_FACTORY_H_
