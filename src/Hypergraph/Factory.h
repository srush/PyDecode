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

class BaseSemiring;
typedef BaseSemiring* (*create_type_fnptr)();
typedef std::map<std::string, create_type_fnptr> RegistryMap;
typedef std::pair<std::string, create_type_fnptr> RegistryPair;

class BaseSemiringFactory {
protected:
	static RegistryMap* registry;

    static RegistryMap * getMap() { 
        return registry; 
    }

public:
	virtual ~BaseSemiringFactory() { };

	static BaseSemiring * create_from_string(std::string name) {
		RegistryMap::const_iterator it = registry->find(name);
		return it == registry->end() ? NULL : it->second();
	}

	static void register_class(std::string name, create_type_fnptr f) {
		(*registry)[name] = f;
	}

	static const vector<create_type_fnptr> retrieve_classes() {
		vector<create_type_fnptr> creators;
		foreach(RegistryPair pr, *registry) {
			creators.push_back(pr.second);
		}
		return creators;
	}
};
RegistryMap* BaseSemiringFactory::registry(new RegistryMap);


template<typename T> BaseSemiring* createT() { return new T; }


template<typename T>
struct DerivedRegister : BaseSemiringFactory { 
    DerivedRegister(std::string const& s) { 
        getMap()->insert(std::make_pair(s, &createT<T>));
    }
};

#endif // HYPERGRAPH_FACTORY_H_
