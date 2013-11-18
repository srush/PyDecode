// Copyright [2013] Alexander Rush

#include "Hypergraph/Factory.h"


template<>
BaseRegistry<BaseSemiring*>::RegistryMap* BaseRegistry<BaseSemiring*>::registry = 
		BaseRegistry<BaseSemiring*>::registry ? BaseRegistry<BaseSemiring*>::registry 
		: new BaseRegistry<BaseSemiring*>::RegistryMap;

template<>
BaseRegistry<StaticBaseSemiringPotential*>::RegistryMap* BaseRegistry<StaticBaseSemiringPotential*>::registry = 
		BaseRegistry<StaticBaseSemiringPotential*>::registry ? BaseRegistry<StaticBaseSemiringPotential*>::registry 
		: new BaseRegistry<StaticBaseSemiringPotential*>::RegistryMap;
