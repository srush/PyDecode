// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_FACTORY_H_
#define HYPERGRAPH_FACTORY_H_

#include <iostream>
#include <map>
#include <string>
#include "Hypergraph/Hypergraph.h"
#include "./common.h"

#define REGISTRY_TYPE_DECLARATION(REGISTRY, NAME) \
    static REGISTRY<NAME> registry

#define REGISTRY_TYPE_DEFINITION(REGISTRY, NAME) \
    REGISTRY<NAME> NAME::registry = REGISTRY<NAME>(#NAME)

#define REGISTRY_TYPE_REDEFINE(REGISTRY, NAME) \
    NAME::registry = REGISTRY<NAME>(#NAME)

template<typename T>
class BaseRegistry {
public:
    typedef T* (*creator_fnptr)();
    typedef std::map<std::string, creator_fnptr> RegistryMap;
    typedef std::pair<std::string, creator_fnptr> RegistryPair;

    virtual ~BaseRegistry() { };

    static const vector<creator_fnptr> retrieve_classes() {
        registry = !registry ? new RegistryMap : registry;
        vector<creator_fnptr> creators;
        // std::cerr<< "map size at retrieve: " << registry->size() << " at: " << registry << std::endl;
        foreach(RegistryPair pr, *registry) {
            creators.push_back(pr.second);
        }
        return creators;
    }

protected:
    static RegistryMap * getMap() { 
        registry = !registry ? new RegistryMap : registry;
        // std::cerr<< "map size: " << registry->size() << " at: " << registry << std::endl;
        return registry; 
    }

private:
    static RegistryMap* registry;

};


class BaseSemiring;
template<typename T> BaseSemiring* createRandomSemiring() { return new T(T::randValue()); }

template<typename T>
struct RandomSemiringRegistry : BaseRegistry<BaseSemiring> { 
    RandomSemiringRegistry(std::string const& s) { 
        BaseRegistry<BaseSemiring>::getMap()->insert(std::make_pair(s, &createRandomSemiring<T>));
    }
};


template<typename T, typename B> B* createT() { return new T(); }

template<typename T, typename B>
struct GenericRegistry : BaseRegistry<B> { 
    GenericRegistry(std::string const& s) { 
        BaseRegistry<B>::getMap()->insert(std::make_pair(s, &createT<T, B>));
    }
};

#endif // HYPERGRAPH_FACTORY_H_
