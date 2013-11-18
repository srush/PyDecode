// Copyright [2013] Alexander Rush

#include "Hypergraph/Factory.h"


template<typename T>
typename BaseRegistry<T>::RegistryMap* BaseRegistry<T>::registry = 
		BaseRegistry<T>::registry ? BaseRegistry<T>::registry : new BaseRegistry<T>::RegistryMap;

// template<>
// BaseRegistry<BaseSemiring>::RegistryMap* BaseRegistry<BaseSemiring>::registry = 
// 		BaseRegistry<BaseSemiring>::registry ? BaseRegistry<BaseSemiring>::registry : new BaseRegistry<BaseSemiring>::RegistryMap;
