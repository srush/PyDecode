// Copyright [2013] Alexander Rush

#ifndef HYPERGRAPH_FACTORY_H_
#define HYPERGRAPH_FACTORY_H_

#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "Hypergraph/Hypergraph.hh"
#include "./common.h"

#define BASE_SEMIRING_REGISTRY_DECLARATION(NAME) \
    static RandomSemiringRegistry<NAME> registry

#define BASE_SEMIRING_REGISTRY_DEFINITION(NAME) \
    RandomSemiringRegistry<NAME> NAME::registry = \
             RandomSemiringRegistry<NAME>(#NAME)

#define BASE_SEMIRING_REGISTRY_REDEFINE(NAME) \
    NAME::registry = RandomSemiringRegistry<NAME>(#NAME)

#define STATIC_SEMIRING_REGISTRY_DECLARATION(NAME) \
    static StaticSemiringRegistry<NAME> registry

#define STATIC_SEMIRING_REGISTRY_DEFINITION(NAME) \
    StaticSemiringRegistry<NAME> NAME::registry = \
            StaticSemiringRegistry<NAME>(#NAME)

#define STATIC_SEMIRING_REGISTRY_REDEFINE(NAME) \
    NAME::registry = StaticSemiringRegistry<NAME>(#NAME)

template<typename T>
class BaseRegistry {
  public:
    typedef T (*creator_fnptr)();
    typedef std::map<std::string, creator_fnptr> RegistryMap;
    typedef std::pair<std::string, creator_fnptr> RegistryPair;

    virtual ~BaseRegistry() { }

    static const vector<creator_fnptr> retrieve_classes() {
        registry = !registry ? new RegistryMap : registry;
        vector<creator_fnptr> creators;

        foreach(RegistryPair pr, *registry) {
            creators.push_back(pr.second);
        }
        return creators;
    }

  protected:
    static RegistryMap * getMap() {
        registry = !registry ? new RegistryMap : registry;
        return registry;
    }

  private:
    static RegistryMap* registry;
};

template<typename T, typename B> B* createT() { return new T(); }

template<typename T, typename B>
struct GenericRegistry : BaseRegistry<B*> {
    explicit GenericRegistry(std::string const& s) {
        BaseRegistry<B>::getMap()->insert(
            BaseRegistry<B>::RegistryPair(s, &createT<T, B>));
    }
};

#endif  // HYPERGRAPH_FACTORY_H_
