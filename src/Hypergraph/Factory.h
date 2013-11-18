// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_FACTORY_H_
#define HYPERGRAPH_FACTORY_H_

#include <iostream>
#include <map>
#include <string>
#include "Hypergraph/Hypergraph.h"
#include "./common.h"

#define BASE_SEMIRING_REGISTRY_DECLARATION(NAME) \
    static RandomSemiringRegistry<NAME> registry

#define BASE_SEMIRING_REGISTRY_DEFINITION(NAME) \
    RandomSemiringRegistry<NAME> NAME::registry = RandomSemiringRegistry<NAME>(#NAME)

#define BASE_SEMIRING_REGISTRY_REDEFINE(NAME) \
    NAME::registry = RandomSemiringRegistry<NAME>(#NAME)

#define STATIC_SEMIRING_REGISTRY_DECLARATION(NAME) \
    static StaticSemiringRegistry<NAME> registry

#define STATIC_SEMIRING_REGISTRY_DEFINITION(NAME) \
    StaticSemiringRegistry<NAME> NAME::registry = StaticSemiringRegistry<NAME>(#NAME)

#define STATIC_SEMIRING_REGISTRY_REDEFINE(NAME) \
    NAME::registry = StaticSemiringRegistry<NAME>(#NAME)

template<typename T>
class BaseRegistry {
public:
    typedef T (*creator_fnptr)();
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
struct RandomSemiringRegistry : BaseRegistry<BaseSemiring*> { 
    RandomSemiringRegistry(std::string const& s) { 
        // std::cerr<< s << " and " << typeid(*createRandomSemiring<T>()).name() << std::endl;
        BaseRegistry<BaseSemiring*>::getMap()->insert(BaseRegistry<BaseSemiring*>::RegistryPair(s, &createRandomSemiring<T>));
    }
};

class StaticBaseSemiringPotential;
template<typename T> StaticBaseSemiringPotential* createStaticSemiring() { return T::create(); }

template<typename T>
struct StaticSemiringRegistry : BaseRegistry<StaticBaseSemiringPotential*> { 
    StaticSemiringRegistry(std::string const& s) { 
        // std::cerr<< s << " and " << typeid(*createStaticSemiring<T>()).name() << std::endl;
        BaseRegistry<StaticBaseSemiringPotential*>::getMap()->insert(BaseRegistry<StaticBaseSemiringPotential*>::RegistryPair(s, &createStaticSemiring<T>));
    }
};


template<typename T, typename B> B* createT() { return new T(); }

template<typename T, typename B>
struct GenericRegistry : BaseRegistry<B*> { 
    GenericRegistry(std::string const& s) { 
        BaseRegistry<B>::getMap()->insert(BaseRegistry<B>::RegistryPair(s, &createT<T, B>));
    }
};

#endif // HYPERGRAPH_FACTORY_H_
