/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2026
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

/**
 * @file
 * @brief CPU channel contracts and their host-side translation into GPU plans and capacity bounds.
 *
 * Internal builders supply daughter hypotheses, source sets, masses, and cuts.
 * The public catalogue is immutable after construction; coverage checks inspect
 * plan declarations rather than running reconstruction. Capacity calculations
 * operate on event range sizes and host scratch storage only.
 */

#include "planning/KFParticleGpuCpuChannelCatalogue.h"

#include "planning/KFParticleGpuDecayPlan.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace
{
  /// @brief Input categories shared by most catalogue entries.
  constexpr unsigned int CommonInputs =
    KFGpuCpuInputTracks | KFGpuCpuInputPrimaryVertices;

  /**
   * @brief Temporary owner of contracts while the immutable catalogue is assembled.
   * Append helpers establish identity and profile IDs; later configuration
   * helpers fill hypotheses/dependencies before final catalogue validation.
   */
  struct CatalogueBuilder
  {
    std::vector<KFParticleGpuCpuChannelContract> entries;  ///< Contracts in deterministic insertion order.

    /**
     * @brief Appends a contract with explicit properties and channel-ID-based hypothesis/selection profiles.
     * @note Does not check uniqueness, dependencies, or conjugates; final validation handles those checks.
     */
    void Add(unsigned int id, unsigned int conjugate, unsigned int family,
             int mother, int firstDaughter, int secondDaughter,
             unsigned int topology, unsigned int generation,
             unsigned int firstSource, unsigned int secondSource,
             unsigned int operations, unsigned int output,
             unsigned int inputs, unsigned int activation,
             unsigned int support, unsigned int reason,
             unsigned int parent = 0u)
    {
      KFParticleGpuCpuChannelContract entry;
      entry.channelId = id;
      entry.conjugateChannelId = conjugate;
      entry.family = family;
      entry.motherPdg = mother;
      entry.firstDaughterPdg = firstDaughter;
      entry.secondDaughterPdg = secondDaughter;
      entry.topology = topology;
      entry.generation = generation;
      entry.firstSourceKind = firstSource;
      entry.secondSourceKind = secondSource;
      entry.parentChannelId = parent;
      entry.operationMask = operations;
      entry.daughterHypothesisProfile = id;
      entry.selectionProfile = id;
      entry.outputClass = output;
      entry.requiredInputs = inputs;
      entry.activation = activation;
      entry.supportStatus = support;
      entry.unsupportedReason = reason;
      entries.push_back(entry);
    }

    /// @brief Appends linked opposite-sign mother hypotheses, initially unsupported with unset daughter PDGs.
    void AddPair(unsigned int firstId, unsigned int secondId,
                 unsigned int family, int mother,
                 unsigned int topology, unsigned int generation,
                 unsigned int firstSource, unsigned int secondSource,
                 unsigned int operations, unsigned int output,
                 unsigned int inputs,
                 unsigned int reason = KFGpuGraphUnsupportedNotImplemented)
    {
      Add(firstId, secondId, family, mother, 0, 0, topology, generation,
          firstSource, secondSource, operations, output, inputs,
          KFGpuCpuChannelActive, KFGpuGraphUnsupported, reason);
      Add(secondId, firstId, family, -mother, 0, 0, topology, generation,
          firstSource, secondSource, operations, output, inputs,
          KFGpuCpuChannelActive, KFGpuGraphUnsupported, reason);
    }

    /// @brief Appends one initially unsupported contract whose conjugate ID points to itself.
    void AddSelf(unsigned int id, unsigned int family, int mother,
                 unsigned int topology, unsigned int generation,
                 unsigned int firstSource, unsigned int secondSource,
                 unsigned int operations, unsigned int output,
                 unsigned int inputs,
                 unsigned int activation = KFGpuCpuChannelActive,
                 unsigned int reason = KFGpuGraphUnsupportedNotImplemented)
    {
      Add(id, id, family, mother, 0, 0, topology, generation,
          firstSource, secondSource, operations, output, inputs, activation,
          KFGpuGraphUnsupported, reason);
    }
  };

  /// @brief Recognizes the three parent hypotheses with explicit composite-track mass constraints.
  bool HasParentMassConstraint(int parentPdg)
  {
    const int absolutePdg = parentPdg < 0 ? -parentPdg : parentPdg;
    return absolutePdg == 3004 || absolutePdg == 3006
      || absolutePdg == 3007;
  }

  /**
   * @brief Assigns daughter PDGs for supported two-track mother hypotheses, including nuclear aliases.
   * @note Unrecognized mothers retain their existing fields; this helper performs no validation.
   */
  void SetTwoDaughterHypotheses(KFParticleGpuCpuChannelContract& entry)
  {
    const int mother = entry.motherPdg;
    switch (mother) {
      case 22: entry.firstDaughterPdg = -11; entry.secondDaughterPdg = 11; break;
      case 200113: entry.firstDaughterPdg = -13; entry.secondDaughterPdg = 13; break;
      case 310:
      case 113:
      case 420: entry.firstDaughterPdg = 211; entry.secondDaughterPdg = -211; break;
      case 333:
      case 426: entry.firstDaughterPdg = 321; entry.secondDaughterPdg = -321; break;
      case 200443: entry.firstDaughterPdg = 2212; entry.secondDaughterPdg = -2212; break;
      case 421: entry.firstDaughterPdg = 211; entry.secondDaughterPdg = -321; break;
      case -421: entry.firstDaughterPdg = 321; entry.secondDaughterPdg = -211; break;
      case 313: entry.firstDaughterPdg = 321; entry.secondDaughterPdg = -211; break;
      case -313: entry.firstDaughterPdg = 211; entry.secondDaughterPdg = -321; break;
      case 2114: entry.firstDaughterPdg = 2212; entry.secondDaughterPdg = -211; break;
      case -2114: entry.firstDaughterPdg = -2212; entry.secondDaughterPdg = 211; break;
      case 3124: entry.firstDaughterPdg = 2212; entry.secondDaughterPdg = -321; break;
      case -3124: entry.firstDaughterPdg = -2212; entry.secondDaughterPdg = 321; break;
      default: break;
    }
    if (entry.firstDaughterPdg != 0) { return; }

    const int hyperMothers[8] = {3003, 3103, 3004, 3005, 3016, 3019, 3022, 3025};
    const int nuclei[8] = {1000010020, 1000010030, 1000020030, 1000020040,
                           1000020060, 1000030060, 1000030070, 1000040070};
    for (unsigned int index = 0u; index < 8u; ++index) {
      if (mother == hyperMothers[index]) {
        entry.firstDaughterPdg = nuclei[index];
        entry.secondDaughterPdg = -211;
        return;
      }
      if (mother == -hyperMothers[index]) {
        entry.firstDaughterPdg = -nuclei[index];
        entry.secondDaughterPdg = 211;
        return;
      }
    }

    const int pionMothers[8] = {100001, 100003, 100005, 100007,
                                100009, 100011, 100013, 100015};
    const int kaonMothers[8] = {110001, 110003, 110005, 110007,
                                110009, 110011, 110013, 110015};
    for (unsigned int index = 0u; index < 8u; ++index) {
      if (mother == pionMothers[index]) {
        entry.firstDaughterPdg = nuclei[index];
        entry.secondDaughterPdg = -211;
        return;
      }
      if (mother == kaonMothers[index]) {
        entry.firstDaughterPdg = nuclei[index];
        entry.secondDaughterPdg = -321;
        return;
      }
    }
  }

  /// @brief Identifies catalogue mother hypotheses that use primary-track sets for two-track construction.
  bool IsPrimaryTwoDaughterMother(int mother)
  {
    const int absoluteMother = mother < 0 ? -mother : mother;
    if (absoluteMother == 113 || absoluteMother == 313
        || absoluteMother == 333 || absoluteMother == 2114
        || absoluteMother == 3124 || absoluteMother == 200113
        || absoluteMother == 200443) {
      return true;
    }
    return (absoluteMother >= 100001 && absoluteMother <= 100015)
           || (absoluteMother >= 110001 && absoluteMother <= 110015);
  }

  /**
   * @brief Returns the charge sign used for track-set routing, not the full physical charge.
   * @note Handles lepton sign conventions, including alias 19; otherwise follows the PDG sign.
   * Only charged catalogue inputs are intended here; neutral or unknown PDGs are not validated.
   */
  int TrackCharge(int pdg)
  {
    const int absolutePdg = pdg < 0 ? -pdg : pdg;
    if (absolutePdg == 11 || absolutePdg == 13 || absolutePdg == 19) {
      return pdg < 0 ? 1 : -1;
    }
    return pdg > 0 ? 1 : -1;
  }

  /// @brief Maps an absolute track PDG to an input species; unknown PDGs return NumberOfTrackSpecies.
  KFParticleGpuTrackSpecies TrackSpecies(int pdg)
  {
    switch (pdg < 0 ? -pdg : pdg) {
      case 11: return Electron;
      case 13:
      case 19: return Muon;
      case 211: return Pion;
      case 321: return Kaon;
      case 2212: return Proton;
      case 1000010020: return Deuteron;
      case 1000010030: return Triton;
      case 1000020030: return Helium3;
      case 1000020040: return Helium4;
      case 1000020060: return Helium6;
      case 1000030060: return Lithium6;
      case 1000030070: return Lithium7;
      case 1000040070: return Beryllium7;
      default: return NumberOfTrackSpecies;
    }
  }

  /// @brief Returns the catalogue track mass in GeV/c^2; unlisted PDGs use the charged-pion mass.
  float TrackMass(int pdg)
  {
    switch (pdg < 0 ? -pdg : pdg) {
      case 11: return 0.00051099895f;
      case 13:
      case 19: return 0.1056583745f;
      case 321: return 0.493677f;
      case 2212: return 0.9382720813f;
      case 3112: return 1.197449f;
      case 3222: return 1.18937f;
      case 3312: return 1.32171f;
      case 3334: return 1.67245f;
      case 1000010020: return 1.87561294257f;
      case 1000010030: return 2.80892113298f;
      case 1000020030: return 2.80839160743f;
      case 1000020040: return 3.7273794066f;
      case 1000020060: return 5.6055375f;
      case 1000030060: return 5.6015181f;
      case 1000030070: return 6.5338336f;
      case 1000040070: return 6.5341844f;
      default: return 0.13957039f;
    }
  }

  /**
   * @brief Returns the neutral-daughter mass for missing-mass channel pairs 6001-6022.
   * @note Pair members share a mass; IDs outside that block return -1.
   */
  float NeutralMissingMass(unsigned int channelId)
  {
    switch ((channelId - 6001u) / 2u) {
      case 0u:
      case 1u: return 0.f;
      case 2u:
      case 3u: return 0.939565f;
      case 4u: return 1.115683f;
      case 5u: return 1.31486f;
      case 6u: return 0.1349766f;
      case 7u: return 1.115683f;
      case 8u:
      case 9u:
      case 10u: return 0.1349766f;
      default: return -1.f;
    }
  }

  /**
   * @brief Multiplies two slot counts without unsigned wraparound.
   * @throws std::overflow_error If the product cannot be represented by the unsigned GPU ABI.
   */
  unsigned int CheckedPairCount(unsigned int first, unsigned int second)
  {
    if (first != 0u
        && second > std::numeric_limits<unsigned int>::max() / first) {
      throw std::overflow_error(
        "KFParticle GPU CPU-plan raw task count exceeds the GPU ABI");
    }
    return first * second;
  }

  /// @brief Returns min(first * second, limit) without overflowing; clipping is intentional and does not throw.
  unsigned int BoundedPairCount(unsigned int first, unsigned int second,
                                unsigned int limit)
  {
    if (first == 0u || second == 0u || limit == 0u) { return 0u; }
    if (second > limit / first) { return limit; }
    const unsigned int pairs = first * second;
    return pairs < limit ? pairs : limit;
  }

  /**
   * @brief Adds a slot count to an existing total without unsigned wraparound.
   * @throws std::overflow_error If the sum exceeds the unsigned GPU ABI; leaves the total unchanged.
   */
  void CheckedAccumulate(unsigned int value, unsigned int& total)
  {
    if (value > std::numeric_limits<unsigned int>::max() - total) {
      throw std::overflow_error(
        "KFParticle GPU CPU-plan raw task count exceeds the GPU ABI");
    }
    total += value;
  }

  /**
   * @brief Decodes a graph track-source ID and returns its event-local set or species range.
   * @throws std::logic_error If the source ID does not encode a valid track source.
   * @note Does not validate the contents or bounds of the event's stored range.
   */
  KFParticleGpuRange ResolveTrackSourceRange(
    const KFParticleGpuEventDesc& event,
    unsigned int sourceId)
  {
    KFParticleGpuTrackSet set = NumberOfTrackSets;
    KFParticleGpuTrackSpecies species = NumberOfTrackSpecies;
    if (!KFParticleGpuDecodeGraphTrackSourceId(sourceId, set, species)) {
      throw std::logic_error(
        "KFParticle GPU CPU plan contains an invalid raw-track source");
    }
    const KFParticleGpuTrackSetDesc& trackSet = event.TrackSet(set);
    return species == NumberOfTrackSpecies ? trackSet.tracks
                                            : trackSet.Species(species);
  }

  /// @brief Builds a filtered missing-mass operation from primary mother and secondary charged-daughter tracks.
  KFParticleGpuGraphOperationChannel MakeCpuNeutralMissingMassChannel(
    const KFParticleGpuCpuChannelContract& contract)
  {
    KFParticleGpuGraphOperationChannel channel;
    const int daughterCharge = TrackCharge(contract.secondDaughterPdg);
    const KFParticleGpuTrackSet motherTracks = daughterCharge > 0
      ? PrimaryPositiveLast : PrimaryNegativeLast;
    const KFParticleGpuTrackSet daughterTracks = daughterCharge > 0
      ? SecondaryPositiveFirst : SecondaryNegativeFirst;
    channel.node.channelId = contract.channelId;
    channel.node.motherPdg = contract.motherPdg;
    channel.node.topology = contract.topology;
    channel.node.generation = contract.generation;
    channel.node.firstSource = {
      KFGpuGraphSourceTrackRange, 0u,
      KFParticleGpuGraphTrackSourceId(motherTracks, NumberOfTrackSpecies)};
    channel.node.secondSource = {
      KFGpuGraphSourceTrackRange, 0u,
      KFParticleGpuGraphTrackSourceId(
        daughterTracks, TrackSpecies(contract.secondDaughterPdg))};
    channel.node.operationMask = contract.operationMask;
    channel.node.selectionProfile = contract.selectionProfile;
    channel.node.outputClass = contract.outputClass;
    channel.descriptor.channelId = contract.channelId;
    channel.descriptor.topology = contract.topology;
    channel.descriptor.operationMask = contract.operationMask;
    channel.descriptor.outputClass = contract.outputClass;
    channel.descriptor.flags = KFGpuGraphRequireSameEvent
      | KFGpuGraphRequireDistinctSources
      | KFGpuGraphRequirePositiveMissingEnergy;
    channel.descriptor.missingMassMode =
      KFGpuGraphMissingMassFilteredReconstruction;
    channel.descriptor.motherPdg = contract.motherPdg;
    channel.descriptor.firstPdg = contract.firstDaughterPdg;
    channel.descriptor.secondPdg = contract.secondDaughterPdg;
    channel.descriptor.firstMass = TrackMass(contract.firstDaughterPdg);
    channel.descriptor.secondMass = TrackMass(contract.secondDaughterPdg);
    channel.descriptor.neutralMass = NeutralMissingMass(contract.channelId);
    channel.descriptor.maxGeometricChi2PerNdf = 3.f;
    return channel;
  }

  /**
   * @brief Returns the first active catalogue channel with the exact signed mother PDG.
   * @throws std::invalid_argument If no active match exists; does not resolve ambiguous PDGs by topology.
   */
  unsigned int FindChannelId(const std::vector<KFParticleGpuCpuChannelContract>& entries,
                             int mother)
  {
    for (const KFParticleGpuCpuChannelContract& entry : entries) {
      if (entry.motherPdg == mother && entry.activation == KFGpuCpuChannelActive) {
        return entry.channelId;
      }
    }
    throw std::invalid_argument("KFParticle GPU composite-track parent is absent from the CPU catalogue");
  }

  /**
   * @brief Returns a writable contract by stable channel ID.
   * @throws std::invalid_argument If the ID is absent; the returned reference follows vector lifetime rules.
   */
  KFParticleGpuCpuChannelContract& FindChannel(
    std::vector<KFParticleGpuCpuChannelContract>& entries,
    unsigned int channelId)
  {
    for (KFParticleGpuCpuChannelContract& entry : entries) {
      if (entry.channelId == channelId) return entry;
    }
    throw std::invalid_argument(
      "KFParticle GPU CPU channel is absent from the frozen catalogue");
  }

  /**
   * @brief Returns a read-only contract by stable channel ID.
   * @throws std::invalid_argument If the ID is absent; does not filter by activation or support.
   */
  const KFParticleGpuCpuChannelContract& FindChannel(
    const std::vector<KFParticleGpuCpuChannelContract>& entries,
    unsigned int channelId)
  {
    for (const KFParticleGpuCpuChannelContract& entry : entries) {
      if (entry.channelId == channelId) return entry;
    }
    throw std::invalid_argument(
      "KFParticle GPU CPU channel is absent from the frozen catalogue");
  }

  /**
   * @brief Resolves two candidate parents and marks their composite-pair contract as supported.
   * @throws std::invalid_argument If a parent is absent or sameInput is requested for different parents.
   */
  void ConfigureCompositeComposite(
    CatalogueBuilder& out,
    unsigned int channelId,
    int firstMother,
    int secondMother,
    bool sameInput)
  {
    KFParticleGpuCpuChannelContract& entry = FindChannel(out.entries, channelId);
    entry.firstDaughterPdg = firstMother;
    entry.secondDaughterPdg = secondMother;
    entry.parentChannelId = FindChannelId(out.entries, firstMother);
    entry.secondParentChannelId = FindChannelId(out.entries, secondMother);
    const KFParticleGpuCpuChannelContract& firstParent =
      FindChannel(out.entries, entry.parentChannelId);
    const KFParticleGpuCpuChannelContract& secondParent =
      FindChannel(out.entries, entry.secondParentChannelId);
    entry.generation = std::max(firstParent.generation,
                                secondParent.generation) + 1u;
    entry.operationMask = KFGpuGraphConstruct | KFGpuGraphSelect;
    entry.supportStatus = KFGpuGraphSupported;
    entry.unsupportedReason = KFGpuGraphUnsupportedNone;
    if (sameInput && entry.parentChannelId != entry.secondParentChannelId) {
      throw std::invalid_argument(
        "KFParticle GPU same-input composite channel has different parents");
    }
  }

  /// @brief Builds a two-candidate graph operation, requiring ordered pairs when both parents are identical.
  KFParticleGpuGraphOperationChannel MakeCpuCompositeCompositeChannel(
    const std::vector<KFParticleGpuCpuChannelContract>& catalogue,
    const KFParticleGpuCpuChannelContract& contract)
  {
    const KFParticleGpuCpuChannelContract& firstParent =
      FindChannel(catalogue, contract.parentChannelId);
    const KFParticleGpuCpuChannelContract& secondParent =
      FindChannel(catalogue, contract.secondParentChannelId);
    KFParticleGpuGraphOperationChannel channel;
    channel.node.channelId = contract.channelId;
    channel.node.motherPdg = contract.motherPdg;
    channel.node.topology = contract.topology;
    channel.node.generation = contract.generation;
    channel.node.firstSource = {KFGpuGraphSourceCandidateGeneration,
                                firstParent.generation,
                                firstParent.channelId};
    channel.node.secondSource = {KFGpuGraphSourceCandidateGeneration,
                                 secondParent.generation,
                                 secondParent.channelId};
    channel.node.operationMask = contract.operationMask;
    channel.node.selectionProfile = contract.selectionProfile;
    channel.node.outputClass = contract.outputClass;
    channel.descriptor.channelId = contract.channelId;
    channel.descriptor.topology = contract.topology;
    channel.descriptor.operationMask = contract.operationMask;
    channel.descriptor.outputClass = contract.outputClass;
    channel.descriptor.motherPdg = contract.motherPdg;
    channel.descriptor.firstPdg = contract.firstDaughterPdg;
    channel.descriptor.secondPdg = contract.secondDaughterPdg;
    channel.descriptor.maxGeometricChi2PerNdf = 3.f;
    if (contract.parentChannelId == contract.secondParentChannelId) {
      channel.descriptor.flags |= KFGpuGraphRequireOrderedCandidatePair;
    }
    return channel;
  }

  /// @brief Links a unary projection to its parent and assigns the following dependency generation.
  void ConfigurePrimaryProjection(CatalogueBuilder& out,
                                  unsigned int channelId,
                                  unsigned int parentChannelId)
  {
    KFParticleGpuCpuChannelContract& entry = FindChannel(out.entries, channelId);
    const KFParticleGpuCpuChannelContract& parent =
      FindChannel(out.entries, parentChannelId);
    entry.firstDaughterPdg = parent.motherPdg;
    entry.secondDaughterPdg = 0;
    entry.parentChannelId = parentChannelId;
    entry.secondParentChannelId = 0u;
    entry.generation = parent.generation + 1u;
    entry.operationMask = KFGpuGraphExtrapolate;
    entry.supportStatus = KFGpuGraphSupported;
    entry.unsupportedReason = KFGpuGraphUnsupportedNone;
  }

  /// @brief Builds unary extrapolation using the parent candidate's associated primary vertex.
  KFParticleGpuGraphOperationChannel MakeCpuPrimaryProjectionChannel(
    const std::vector<KFParticleGpuCpuChannelContract>& catalogue,
    const KFParticleGpuCpuChannelContract& contract)
  {
    const KFParticleGpuCpuChannelContract& parent =
      FindChannel(catalogue, contract.parentChannelId);
    KFParticleGpuGraphOperationChannel channel;
    channel.node.channelId = contract.channelId;
    channel.node.motherPdg = contract.motherPdg;
    channel.node.topology = contract.topology;
    channel.node.generation = contract.generation;
    channel.node.firstSource = {KFGpuGraphSourceCandidateGeneration,
                                parent.generation,
                                parent.channelId};
    channel.node.operationMask = contract.operationMask;
    channel.node.selectionProfile = contract.selectionProfile;
    channel.node.outputClass = contract.outputClass;
    channel.descriptor.channelId = contract.channelId;
    channel.descriptor.topology = contract.topology;
    channel.descriptor.operationMask = contract.operationMask;
    channel.descriptor.outputClass = contract.outputClass;
    channel.descriptor.motherPdg = contract.motherPdg;
    channel.descriptor.firstPdg = contract.firstDaughterPdg;
    channel.descriptor.primaryVertexIndex = KFGpuGraphPrimaryVertexFromCandidate;
    return channel;
  }

  /// @brief Connects conjugate kaon-matching contracts to their three-pion parents and charged-kaon inputs.
  void ConfigureKaonMatchingPair(CatalogueBuilder& out,
                                 unsigned int firstChannelId)
  {
    KFParticleGpuCpuChannelContract& particle =
      FindChannel(out.entries, firstChannelId);
    KFParticleGpuCpuChannelContract& antiparticle =
      FindChannel(out.entries, particle.conjugateChannelId);
    particle.firstDaughterPdg = 100321;
    particle.secondDaughterPdg = 321;
    antiparticle.firstDaughterPdg = -100321;
    antiparticle.secondDaughterPdg = -321;
    for (KFParticleGpuCpuChannelContract* entry : {&particle, &antiparticle}) {
      entry->parentChannelId =
        FindChannelId(out.entries, entry->firstDaughterPdg);
      entry->generation =
        FindChannel(out.entries, entry->parentChannelId).generation + 1u;
      entry->firstSourceKind = KFGpuGraphSourceCandidateGeneration;
      entry->secondSourceKind = KFGpuGraphSourceTrackRange;
      entry->topology = KFGpuGraphTopologyCompositeTrack;
      entry->operationMask =
        KFGpuGraphConstruct | KFGpuGraphMassConstraint
        | KFGpuGraphMatch | KFGpuGraphSelect;
      entry->supportStatus = KFGpuGraphSupported;
      entry->unsupportedReason = KFGpuGraphUnsupportedNone;
    }
  }

  /// @brief Builds CPU-compatible kaon matching with the parent mass constraint and match-selection cuts.
  KFParticleGpuGraphOperationChannel MakeCpuKaonMatchingChannel(
    const std::vector<KFParticleGpuCpuChannelContract>& catalogue,
    const KFParticleGpuCpuChannelContract& contract)
  {
    const KFParticleGpuCpuChannelContract& parent =
      FindChannel(catalogue, contract.parentChannelId);
    const KFParticleGpuTrackSet trackSet =
      TrackCharge(contract.secondDaughterPdg) > 0
        ? PrimaryPositiveLast : PrimaryNegativeLast;
    KFParticleGpuGraphOperationChannel channel;
    channel.node.channelId = contract.channelId;
    channel.node.motherPdg = contract.motherPdg;
    channel.node.topology = contract.topology;
    channel.node.generation = contract.generation;
    channel.node.firstSource = {KFGpuGraphSourceCandidateGeneration,
                                parent.generation, parent.channelId};
    channel.node.secondSource = {
      KFGpuGraphSourceTrackRange, 0u,
      KFParticleGpuGraphTrackSourceId(trackSet, Kaon)};
    channel.node.operationMask = contract.operationMask;
    channel.node.selectionProfile = contract.selectionProfile;
    channel.node.outputClass = contract.outputClass;
    channel.descriptor.channelId = contract.channelId;
    channel.descriptor.topology = contract.topology;
    channel.descriptor.operationMask = contract.operationMask;
    channel.descriptor.outputClass = contract.outputClass;
    channel.descriptor.flags = KFGpuGraphRequireSameEvent
      | KFGpuGraphRequireDistinctSources | KFGpuGraphStoreFirstOnMatch
      | KFGpuGraphMatchTrackPrimaryVertex;
    channel.descriptor.motherPdg = contract.motherPdg;
    channel.descriptor.firstPdg = contract.firstDaughterPdg;
    channel.descriptor.secondPdg = contract.secondDaughterPdg;
    channel.descriptor.secondMass = TrackMass(contract.secondDaughterPdg);
    // CPU MatchKaons consumes the mass-constrained K -> 3 pi candidate from
    // fPrimCandidates. GetMotherMass(100321) intentionally resolves to the
    // default K0 mass/width in the current CPU catalogue.
    channel.descriptor.massConstraint = 0.497614f;
    channel.descriptor.massConstraintSigma = 0.f;
    channel.descriptor.maxGeometricChi2PerNdf = 3.f;
    channel.descriptor.maxMatchDistance = 20.f;
    channel.descriptor.maxMatchMomentumSigma = 5.f;
    channel.descriptor.maxMatchTopologyChi2PerNdf = 3.f;
    channel.descriptor.matchMassWindowSigma = 3.7e-3f;
    channel.descriptor.matchMassWindowCut = 3.f;
    return channel;
  }

  /// @brief Links conjugate final-selection contracts to the first active parents with matching mother PDGs.
  void ConfigureFinalSelectionPair(CatalogueBuilder& out,
                                   unsigned int firstChannelId)
  {
    KFParticleGpuCpuChannelContract& particle =
      FindChannel(out.entries, firstChannelId);
    KFParticleGpuCpuChannelContract& antiparticle =
      FindChannel(out.entries, particle.conjugateChannelId);
    for (KFParticleGpuCpuChannelContract* entry : {&particle, &antiparticle}) {
      entry->firstDaughterPdg = entry->motherPdg;
      entry->secondDaughterPdg = 0;
      entry->parentChannelId = FindChannelId(out.entries, entry->motherPdg);
      const KFParticleGpuCpuChannelContract& parent =
        FindChannel(out.entries, entry->parentChannelId);
      entry->generation = parent.generation + 1u;
      entry->operationMask =
        KFGpuGraphMassConstraint | KFGpuGraphSelect;
      entry->supportStatus = KFGpuGraphSupported;
      entry->unsupportedReason = KFGpuGraphUnsupportedNone;
    }
  }

  /// @brief Builds unary charm mass-constraint and selection operations over the event's primary vertices.
  KFParticleGpuGraphOperationChannel MakeCpuFinalSelectionChannel(
    const std::vector<KFParticleGpuCpuChannelContract>& catalogue,
    const KFParticleGpuCpuChannelContract& contract)
  {
    const KFParticleGpuCpuChannelContract& parent =
      FindChannel(catalogue, contract.parentChannelId);
    KFParticleGpuGraphOperationChannel channel;
    channel.node.channelId = contract.channelId;
    channel.node.motherPdg = contract.motherPdg;
    channel.node.topology = contract.topology;
    channel.node.generation = contract.generation;
    channel.node.firstSource = {KFGpuGraphSourceCandidateGeneration,
                                parent.generation, parent.channelId};
    channel.node.operationMask = contract.operationMask;
    channel.node.selectionProfile = contract.selectionProfile;
    channel.node.outputClass = contract.outputClass;
    channel.descriptor.channelId = contract.channelId;
    channel.descriptor.topology = contract.topology;
    channel.descriptor.operationMask = contract.operationMask;
    channel.descriptor.outputClass = contract.outputClass;
    channel.descriptor.flags = KFGpuGraphRequireSameEvent
      | KFGpuGraphSelectEventPrimaryVertices;
    channel.descriptor.motherPdg = contract.motherPdg;
    channel.descriptor.firstPdg = contract.firstDaughterPdg;
    const int absPdg = contract.motherPdg < 0
      ? -contract.motherPdg : contract.motherPdg;
    channel.descriptor.massConstraint =
      absPdg == 411 || absPdg == 200411 ? 1.86962f : 1.86484f;
    channel.descriptor.massConstraintSigma = 0.f;
    channel.descriptor.matchMassWindowSigma = 0.0145f;
    channel.descriptor.matchMassWindowCut = 3.f;
    channel.descriptor.maxTopologyChi2PerNdf = 3.f;
    channel.descriptor.maxSelectionVertexDistance = 200.f;
    channel.descriptor.minSelectionDecayLengthOverError = 10.f;
    return channel;
  }

  /// @brief Assigns charged inputs to conjugate missing-mass contracts and marks both as active and supported.
  void ConfigureNeutralMissingMassPair(CatalogueBuilder& out,
                                       unsigned int firstChannelId,
                                       int motherTrackPdg,
                                       int chargedDaughterPdg)
  {
    KFParticleGpuCpuChannelContract& particle =
      FindChannel(out.entries, firstChannelId);
    KFParticleGpuCpuChannelContract& antiparticle =
      FindChannel(out.entries, particle.conjugateChannelId);
    particle.firstDaughterPdg = motherTrackPdg;
    particle.secondDaughterPdg = chargedDaughterPdg;
    antiparticle.firstDaughterPdg = -motherTrackPdg;
    antiparticle.secondDaughterPdg = -chargedDaughterPdg;
    for (KFParticleGpuCpuChannelContract* entry : {&particle, &antiparticle}) {
      entry->generation = 1u;
      entry->firstSourceKind = KFGpuGraphSourceTrackRange;
      entry->secondSourceKind = KFGpuGraphSourceTrackRange;
      entry->requiredInputs = CommonInputs;
      entry->activation = KFGpuCpuChannelActive;
      entry->supportStatus = KFGpuGraphSupported;
      entry->unsupportedReason = KFGpuGraphUnsupportedNone;
    }
  }

  /**
   * @brief Completes the last two appended conjugate composite-track contracts.
   * @note Parent IDs/generations may be supplied for forward declarations; otherwise they are looked up.
   * @pre The last two entries are the particle and antiparticle contracts being configured.
   */
  void ConfigureCompositeTrackPair(CatalogueBuilder& out,
                                   int parent,
                                   int bachelor,
                                   bool primary,
                                   unsigned int family,
                                   unsigned int parentChannelId = 0u,
                                   unsigned int parentGeneration = 0u)
  {
    KFParticleGpuCpuChannelContract& particle = out.entries[out.entries.size() - 2u];
    KFParticleGpuCpuChannelContract& antiparticle = out.entries.back();
    const bool neutralParent = parent == 310 || parent == 420 || parent == 111;
    const int antiParent = neutralParent ? parent : -parent;
    particle.firstDaughterPdg = parent;
    particle.secondDaughterPdg = bachelor;
    particle.parentChannelId = parentChannelId != 0u
      ? parentChannelId : FindChannelId(out.entries, parent);
    antiparticle.firstDaughterPdg = antiParent;
    antiparticle.secondDaughterPdg = -bachelor;
    antiparticle.parentChannelId = parentChannelId != 0u
      ? parentChannelId : FindChannelId(out.entries, antiParent);
    const unsigned int generation = parentGeneration != 0u
      ? parentGeneration + 1u : [&]() {
      for (const KFParticleGpuCpuChannelContract& entry : out.entries) {
        if (entry.channelId == particle.parentChannelId) return entry.generation + 1u;
      }
      return 2u;
    }();
    particle.generation = generation;
    antiparticle.generation = generation;
    particle.family = family;
    antiparticle.family = family;
    particle.outputClass = family == KFGpuCpuFamilyLongLivedComposite
      ? KFGpuGraphOutputSecondary : KFGpuGraphOutputPrimaryAndSecondary;
    antiparticle.outputClass = particle.outputClass;
    particle.requiredInputs = CommonInputs
      | (primary ? 0u : KFGpuCpuInputChiToPrimaryVertex);
    antiparticle.requiredInputs = particle.requiredInputs;
    particle.supportStatus = KFGpuGraphSupported;
    antiparticle.supportStatus = KFGpuGraphSupported;
    particle.unsupportedReason = KFGpuGraphUnsupportedNone;
    antiparticle.unsupportedReason = KFGpuGraphUnsupportedNone;
  }

  /**
   * @brief Completes the last appended self-linked composite-track contract.
   * @note An already declared parent's generation takes precedence over the supplied generation hint.
   * @pre The final entry is the contract being configured.
   */
  void ConfigureCompositeTrackSelf(CatalogueBuilder& out,
                                   int parent,
                                   int bachelor,
                                   unsigned int parentChannelId = 0u,
                                   unsigned int parentGeneration = 0u)
  {
    KFParticleGpuCpuChannelContract& entry = out.entries.back();
    entry.firstDaughterPdg = parent;
    entry.secondDaughterPdg = bachelor;
    entry.parentChannelId = parentChannelId != 0u
      ? parentChannelId : FindChannelId(out.entries, parent);
    entry.generation = parentGeneration != 0u ? parentGeneration + 1u : entry.generation;
    for (const KFParticleGpuCpuChannelContract& source : out.entries) {
      if (source.channelId == entry.parentChannelId) {
        entry.generation = source.generation + 1u;
        break;
      }
    }
    entry.supportStatus = KFGpuGraphSupported;
    entry.unsupportedReason = KFGpuGraphUnsupportedNone;
  }

  /// @brief Chooses a primary/secondary first-parameter track set; nonpositive charge selects the negative set.
  KFParticleGpuTrackSet TrackSetFor(int charge, bool primary)
  {
    if (primary) {
      return charge > 0 ? PrimaryPositiveFirst : PrimaryNegativeFirst;
    }
    return charge > 0 ? SecondaryPositiveFirst : SecondaryNegativeFirst;
  }

  /**
   * @brief Translates a two-track contract into source, mass, transport, and selection settings.
   * Includes CPU-specific alternate PID hypotheses, fit ordering, and charm input cuts.
   */
  KFParticleGpuTwoDaughterChannel MakeCpuTwoDaughterChannel(
    const KFParticleGpuCpuChannelContract& contract)
  {
    KFParticleGpuTwoDaughterChannel channel;
    const bool primary = IsPrimaryTwoDaughterMother(contract.motherPdg);
    const bool charm = contract.motherPdg == 420
                       || (contract.motherPdg < 0 ? -contract.motherPdg
                                                  : contract.motherPdg) == 421
                       || contract.motherPdg == 426;
    channel.channelId = contract.channelId;
    channel.outputClass = contract.outputClass;
    channel.motherPdg = contract.motherPdg;
    channel.firstDaughterPdg = contract.firstDaughterPdg;
    channel.secondDaughterPdg = contract.secondDaughterPdg;
    channel.firstSourcePdg = contract.firstDaughterPdg;
    channel.secondSourcePdg = contract.secondDaughterPdg;
    if (contract.motherPdg == 3122) {
      channel.firstSpecies = NumberOfTrackSpecies;
      channel.firstAlternateSourcePdg = 211;
    }
    else {
      channel.firstSpecies = TrackSpecies(contract.firstDaughterPdg);
    }
    channel.secondSpecies = TrackSpecies(contract.secondDaughterPdg);
    if (contract.motherPdg == 200113) {
      channel.firstAlternateSourcePdg = -19;
      channel.secondAlternateSourcePdg = 19;
    }
    channel.firstCharge = TrackCharge(contract.firstDaughterPdg);
    channel.secondCharge = TrackCharge(contract.secondDaughterPdg);
    channel.firstTrackSet = TrackSetFor(channel.firstCharge, primary);
    channel.secondTrackSet = TrackSetFor(channel.secondCharge, primary);
    channel.primaryVertexIndex = primary ? KFGpuPrimaryVertexFromDaughters : -1;
    channel.firstMass = TrackMass(contract.firstDaughterPdg);
    channel.secondMass = TrackMass(contract.secondDaughterPdg);
    channel.flags = KFGpuTwoDaughterUseLineDca | KFGpuTwoDaughterUseEnergyFit;
    if (channel.firstCharge > 0) {
      channel.flags |= KFGpuTwoDaughterReverseFitOrder;
    }
    channel.transportMode = KFGpuTransportFullField;
    channel.maxDaughterDistance = primary ? -1.f : 1.f;
    channel.selection.maxGeometricChi2PerNdf = 3.f;
    channel.selection.expectedMassSigma = -1.f;
    channel.selection.massSigmaCut = -1.f;
    channel.selection.maxPrimaryVertexDistance = primary ? -1.f : 200.f;
    channel.selection.minSecondaryLdL = primary || charm ? -1.f : 5.f;
    channel.selection.maxPrimaryTopologyChi2PerNdf = -1.f;
    channel.selection.maxSecondaryTopologyChi2PerNdf = -1.f;
    channel.selection.requirePrimaryVertex = primary ? 0u : 1u;
    channel.selection.topologyMode = primary ? KFGpuV0TopologySpatial
                                             : KFGpuV0TopologyLine;
    if (contract.motherPdg == 310) {
      channel.motherMass = 0.497614f;
      channel.motherMassSigma = 3.7e-3f;
    }
    else if ((contract.motherPdg < 0 ? -contract.motherPdg
                                     : contract.motherPdg) == 3122) {
      channel.motherMass = 1.115683f;
      channel.motherMassSigma = 1.5e-3f;
    }
    else if (contract.motherPdg == 22) {
      channel.motherMass = 0.f;
      channel.motherMassSigma = 6.e-3f;
    }
    else {
      channel.motherMass = 0.497614f;
      channel.motherMassSigma = -1.f;
    }
    if (contract.motherPdg == 310
        || (contract.motherPdg < 0 ? -contract.motherPdg
                                   : contract.motherPdg) == 3122
        || contract.motherPdg == 22) {
      channel.secondaryMassSigmaCut = 3.f;
      channel.maxSecondaryTopoChi2PerNdf = 5.f;
      channel.minSecondaryLdL = 10.f;
      channel.selection.expectedMass = channel.motherMass;
      channel.selection.expectedMassSigma = channel.motherMassSigma;
      channel.selection.massSigmaCut = 3.f;
    }
    if (charm) {
      channel.minFirstPixelHits = 3;
      channel.minSecondPixelHits = 3;
      channel.minFirstChiToPrimaryVertex = 8.f;
      channel.minSecondChiToPrimaryVertex = 8.f;
      channel.minFirstPt = 0.2f;
      channel.minSecondPt = 0.2f;
    }
    return channel;
  }

  /// @brief Identifies composite-track mother hypotheses that consume primary bachelor tracks.
  bool IsPrimaryCompositeTrackMother(int mother)
  {
    const int value = mother < 0 ? -mother : mother;
    switch (value) {
      case 3224: case 3114: case 1003314: case 1003003:
      case 1003004: case 1003005: case 1003006: case 1003007:
      case 323: case 3324: case 1003334: case 10411:
      case 10421: case 20411: case 100323: return true;
      default: return false;
    }
  }

  /// @brief Identifies composite-track mother hypotheses requiring the catalogue's charm bachelor cuts.
  bool IsCharmCompositeTrackMother(int mother)
  {
    const int value = mother < 0 ? -mother : mother;
    switch (value) {
      case 411: case 431: case 4122: case 521: case 529:
      case 10411: case 300411: case 400431: case 504122:
      case 429: case 10421: case 511: case 519: case 20411:
      case 425: case 427: case 200411: case 300431: return true;
      default: return false;
    }
  }

  /// @brief Translates a composite-track contract into full-field cascade settings and bachelor/parent cuts.
  KFParticleGpuV0TrackCascadeChannel MakeCpuCompositeTrackChannel(
    const KFParticleGpuCpuChannelContract& contract)
  {
    KFParticleGpuV0TrackCascadeChannel channel;
    const bool primary = IsPrimaryCompositeTrackMother(contract.motherPdg);
    channel.channelId = contract.channelId;
    channel.family = contract.family;
    channel.generation = contract.generation;
    channel.outputClass = contract.outputClass;
    channel.parentChannelId = contract.parentChannelId;
    channel.v0Pdg = contract.firstDaughterPdg;
    channel.bachelorPdg = contract.secondDaughterPdg;
    channel.bachelorSpecies = TrackSpecies(contract.secondDaughterPdg);
    channel.bachelorTrackSet = TrackSetFor(
      TrackCharge(contract.secondDaughterPdg), primary);
    channel.motherPdg = contract.motherPdg;
    channel.primaryVertexIndex = primary ? KFGpuPrimaryVertexFromDaughters : -1;
    channel.flags = KFGpuV0TrackUseLineDca | KFGpuV0TrackUseEnergyFit;
    channel.transportMode = KFGpuTransportFullField;
    channel.bachelorMass = TrackMass(contract.secondDaughterPdg);
    channel.motherMass = 0.f;
    channel.motherMassSigma = -1.f;
    channel.secondaryMassSigmaCut = -1.f;
    channel.maxSecondaryTopoChi2PerNdf = -1.f;
    channel.minBachelorChiToPrimaryVertex = primary ? -1.f : 3.f;
    channel.maxV0TrackDistance =
      contract.family == KFGpuCpuFamilyTrackComposite && !primary ? 1.f : -1.f;
    switch (channel.v0Pdg < 0 ? -channel.v0Pdg : channel.v0Pdg) {
      case 3004: channel.parentMassConstraint = 2.9914f; break;
      case 3006: channel.parentMassConstraint = 3.9217f; break;
      case 3007: channel.parentMassConstraint = 4.8397f; break;
      default: break;
    }
    if (IsCharmCompositeTrackMother(contract.motherPdg)) {
      channel.minBachelorChiToPrimaryVertex = primary ? -1.f : 8.f;
      channel.minBachelorPt = 0.2f;
      channel.minBachelorPixelHits = 3;
    }
    return channel;
  }

  /**
   * @brief Assembles all finder families, resolves dependencies, and validates the completed catalogue.
   * @return Owned contracts in stable insertion order, including explicitly disabled CPU channels.
   * @throws std::invalid_argument If the compiled definitions contain inconsistent contracts or links.
   */
  std::vector<KFParticleGpuCpuChannelContract> BuildCatalogue()
  {
    CatalogueBuilder out;
    const unsigned int construct =
      KFGpuGraphConstruct | KFGpuGraphTransport | KFGpuGraphSelect;

    // Exact default channels retain their established device IDs.
    out.Add(1u, 1u, KFGpuCpuFamilyTwoDaughter, 310, 211, -211,
            KFGpuGraphTopologyTrackTrack, 1u, KFGpuGraphSourceTrackRange,
            KFGpuGraphSourceTrackRange, construct,
            KFGpuGraphOutputPrimaryAndSecondary,
            CommonInputs | KFGpuCpuInputChiToPrimaryVertex,
            KFGpuCpuChannelActive, KFGpuGraphSupported,
            KFGpuGraphUnsupportedNone);
    out.Add(2u, 3u, KFGpuCpuFamilyTwoDaughter, 3122, 2212, -211,
            KFGpuGraphTopologyTrackTrack, 1u, KFGpuGraphSourceTrackRange,
            KFGpuGraphSourceTrackRange, construct,
            KFGpuGraphOutputPrimaryAndSecondary,
            CommonInputs | KFGpuCpuInputChiToPrimaryVertex,
            KFGpuCpuChannelActive, KFGpuGraphSupported,
            KFGpuGraphUnsupportedNone);
    out.Add(3u, 2u, KFGpuCpuFamilyTwoDaughter, -3122, -2212, 211,
            KFGpuGraphTopologyTrackTrack, 1u, KFGpuGraphSourceTrackRange,
            KFGpuGraphSourceTrackRange, construct,
            KFGpuGraphOutputPrimaryAndSecondary,
            CommonInputs | KFGpuCpuInputChiToPrimaryVertex,
            KFGpuCpuChannelActive, KFGpuGraphSupported,
            KFGpuGraphUnsupportedNone);

    unsigned int next = 1001u;
    for (int mother : {22, 113, 333, 200113, 200443, 426}) {
      out.AddSelf(next++, KFGpuCpuFamilyTwoDaughter, mother,
                  KFGpuGraphTopologyTrackTrack, 1u,
                  KFGpuGraphSourceTrackRange, KFGpuGraphSourceTrackRange,
                  construct, KFGpuGraphOutputPrimaryAndSecondary,
                  CommonInputs | KFGpuCpuInputChiToPrimaryVertex);
    }
    for (int mother : {421, 313, 2114, 3124, 3003, 3103, 3004, 3005,
                       3016, 3019, 3022, 3025}) {
      out.AddPair(next, next + 1u, KFGpuCpuFamilyTwoDaughter, mother,
                  KFGpuGraphTopologyTrackTrack, 1u,
                  KFGpuGraphSourceTrackRange, KFGpuGraphSourceTrackRange,
                  construct, KFGpuGraphOutputPrimaryAndSecondary,
                  CommonInputs | KFGpuCpuInputChiToPrimaryVertex);
      next += 2u;
    }
    for (int mother : {100001, 100003, 100005, 100007, 100009, 100011,
                       100013, 100015, 110001, 110003, 110005, 110007,
                       110009, 110011, 110013, 110015}) {
      out.AddSelf(next++, KFGpuCpuFamilyTwoDaughter, mother,
                  KFGpuGraphTopologyTrackTrack, 1u,
                  KFGpuGraphSourceTrackRange, KFGpuGraphSourceTrackRange,
                  construct, KFGpuGraphOutputPrimaryAndSecondary,
                  CommonInputs | KFGpuCpuInputChiToPrimaryVertex);
    }
    out.AddSelf(next++, KFGpuCpuFamilyTwoDaughter, 420,
                KFGpuGraphTopologyTrackTrack, 1u,
                KFGpuGraphSourceTrackRange, KFGpuGraphSourceTrackRange,
                construct, KFGpuGraphOutputSecondary,
                CommonInputs | KFGpuCpuInputChiToPrimaryVertex);

    for (KFParticleGpuCpuChannelContract& entry : out.entries) {
      if (entry.family == KFGpuCpuFamilyTwoDaughter) {
        SetTwoDaughterHypotheses(entry);
        entry.supportStatus = KFGpuGraphSupported;
        entry.unsupportedReason = KFGpuGraphUnsupportedNone;
      }
    }

    next = 2001u;
    for (int mother = 3030; mother <= 3037; ++mother) {
      out.AddSelf(next++, KFGpuCpuFamilySameSignPrimaryResonance, mother,
                  KFGpuGraphTopologyTrackTrack, 1u,
                  KFGpuGraphSourceTrackRange, KFGpuGraphSourceTrackRange,
                  construct, KFGpuGraphOutputPrimary, CommonInputs,
                  KFGpuCpuChannelConfigurationDisabled);
    }

    // The two established cascade pairs preserve IDs 11-14.
    out.AddPair(11u, 12u, KFGpuCpuFamilyTrackComposite, 3312,
                KFGpuGraphTopologyCompositeTrack, 2u,
                KFGpuGraphSourceCandidateGeneration,
                KFGpuGraphSourceTrackRange, construct,
                KFGpuGraphOutputSecondary,
                CommonInputs | KFGpuCpuInputChiToPrimaryVertex);
    ConfigureCompositeTrackPair(out, 3122, -211, false,
                                KFGpuCpuFamilyTrackComposite);
    out.AddPair(13u, 14u, KFGpuCpuFamilyTrackComposite, 3334,
                KFGpuGraphTopologyCompositeTrack, 2u,
                KFGpuGraphSourceCandidateGeneration,
                KFGpuGraphSourceTrackRange, construct,
                KFGpuGraphOutputSecondary,
                CommonInputs | KFGpuCpuInputChiToPrimaryVertex);
    ConfigureCompositeTrackPair(out, 3122, -321, false,
                                KFGpuCpuFamilyTrackComposite);

    next = 3001u;
    /** @brief Compact construction recipe shared by the composite-track catalogue tables. */
    struct CompositeTrackSpec
    {
      int mother;     ///< Produced mother hypothesis.
      int parent;     ///< Input composite hypothesis.
      int bachelor;   ///< Additional charged-track hypothesis.
      bool primary;  ///< Whether the bachelor comes from primary-track input.
    };
    const CompositeTrackSpec trackComposite[] = {
      {304122,3122,211,false}, {3224,3122,211,true},
      {3114,3122,-211,true}, {1003314,3122,-321,true},
      {1003003,3122,2212,true}, {1003004,3122,1000010020,true},
      {1003005,3122,1000010030,true}, {1003006,3122,1000020030,true},
      {1003007,3122,1000020040,true}, {323,310,211,true},
      {100411,310,211,false}, {100431,310,321,false},
      {104122,310,2212,false}, {100321,310,211,false},
      {3324,3312,211,true}, {1003334,3324,-321,true},
      {411,421,211,false}, {431,421,321,false},
      {4122,421,2212,false}, {521,421,211,false},
      {529,421,321,false}, {10411,421,211,true},
      {300411,420,211,false}, {400431,420,321,false},
      {504122,420,2212,false}, {429,411,-211,false},
      {10421,411,-211,true}, {511,421,211,false},
      {519,421,321,false}, {20411,429,211,true},
      {425,100411,-211,false}, {427,100431,-321,false},
      {200411,425,211,false}, {300431,425,321,false},
      {3222,111,2212,false}, {100323,111,321,true}
    };
    for (const CompositeTrackSpec& spec : trackComposite) {
      const int mother = spec.mother;
      out.AddPair(next, next + 1u, KFGpuCpuFamilyTrackComposite, mother,
                  KFGpuGraphTopologyCompositeTrack, 2u,
                  KFGpuGraphSourceCandidateGeneration,
                  KFGpuGraphSourceTrackRange, construct,
                  KFGpuGraphOutputPrimaryAndSecondary,
                  CommonInputs | KFGpuCpuInputChiToPrimaryVertex);
      ConfigureCompositeTrackPair(out, spec.parent, spec.bachelor,
                                  spec.primary,
                                  KFGpuCpuFamilyTrackComposite,
                                  spec.parent == 111 ? 5019u : 0u,
                                  spec.parent == 111 ? 3u : 0u);
      if (spec.parent == 111) {
        out.entries[out.entries.size() - 2u].supportStatus = KFGpuGraphUnsupported;
        out.entries[out.entries.size() - 2u].unsupportedReason =
          KFGpuGraphUnsupportedValidationPending;
        out.entries.back().supportStatus = KFGpuGraphUnsupported;
        out.entries.back().unsupportedReason =
          KFGpuGraphUnsupportedValidationPending;
      }
      next += 2u;
    }

    next = 4001u;
    const CompositeTrackSpec longLived[] = {
      {3012,3003,2212,false}, {3014,3003,1000010020,false},
      {3015,3003,1000010030,false}, {3013,3103,2212,false},
      {3017,3103,1000010030,false}, {3006,3004,2212,false},
      {3018,3004,1000010020,false}, {3020,3004,1000010030,false},
      {3024,3004,1000020030,false}, {3026,3004,1000020040,false},
      {3007,3005,2212,false}, {3021,3005,1000010020,false},
      {3023,3005,1000010030,false}, {3027,3005,1000020040,false},
      {3028,3122,2212,false}, {3029,3028,2212,false},
      {314122,304122,211,false}, {404122,314122,-211,false},
      {114122,104122,211,false}, {204122,114122,-211,false}
    };
    for (const CompositeTrackSpec& spec : longLived) {
      const int mother = spec.mother;
      const unsigned int operations = construct
        | (HasParentMassConstraint(spec.parent)
             ? KFGpuGraphMassConstraint : 0u);
      out.AddPair(next, next + 1u, KFGpuCpuFamilyLongLivedComposite, mother,
                  KFGpuGraphTopologyCompositeTrack, 3u,
                  KFGpuGraphSourceCandidateGeneration,
                  KFGpuGraphSourceTrackRange,
                  operations,
                  KFGpuGraphOutputSecondary, CommonInputs);
      ConfigureCompositeTrackPair(out, spec.parent, spec.bachelor, false,
                                  KFGpuCpuFamilyLongLivedComposite);
      next += 2u;
    }
    const CompositeTrackSpec longLivedSelf[] = {
      {3008,3006,-211,false}, {3010,3007,-211,false},
      {3009,3203,2212,false}, {3011,3010,2212,false},
      {3038,3006,2212,false}, {3039,3007,2212,false},
      {3040,3005,-211,false}, {3203,3004,-211,false}
    };
    for (const CompositeTrackSpec& spec : longLivedSelf) {
      const int mother = spec.mother;
      const unsigned int operations = construct
        | (HasParentMassConstraint(spec.parent)
             ? KFGpuGraphMassConstraint : 0u);
      out.AddSelf(next++, KFGpuCpuFamilyLongLivedComposite, mother,
                  KFGpuGraphTopologyCompositeTrack, 3u,
                  KFGpuGraphSourceCandidateGeneration,
                  KFGpuGraphSourceTrackRange,
                  operations,
                  KFGpuGraphOutputSecondary, CommonInputs);
      ConfigureCompositeTrackSelf(out, spec.parent, spec.bachelor,
                                  spec.parent == 3203 ? 4048u : 0u,
                                  spec.parent == 3203 ? 2u : 0u);
    }

    next = 5001u;
    for (int mother : {3212, 3322, 3214, 3314, 200411, 404122, 4132,
                       300431, 204122}) {
      out.AddPair(next, next + 1u, KFGpuCpuFamilyCompositeComposite, mother,
                  KFGpuGraphTopologyCompositeComposite, 3u,
                  KFGpuGraphSourceCandidateGeneration,
                  KFGpuGraphSourceCandidateGeneration,
                  construct | KFGpuGraphSetProductionVertex | KFGpuGraphSelect,
                  KFGpuGraphOutputPrimaryAndSecondary, CommonInputs);
      next += 2u;
    }
    for (int mother : {111, 3000, 100313, 300443, 400443, 500443}) {
      out.AddSelf(next++, KFGpuCpuFamilyCompositeComposite, mother,
                  KFGpuGraphTopologyCompositeComposite, 3u,
                  KFGpuGraphSourceCandidateGeneration,
                  KFGpuGraphSourceCandidateGeneration,
                  construct | KFGpuGraphSetProductionVertex | KFGpuGraphSelect,
                  KFGpuGraphOutputPrimaryAndSecondary, CommonInputs);
    }

    ConfigureCompositeComposite(out, 5001u, 22, 3122, false);
    ConfigureCompositeComposite(out, 5002u, 22, -3122, false);
    ConfigureCompositeComposite(out, 5003u, 111, 3122, false);
    ConfigureCompositeComposite(out, 5004u, 111, -3122, false);
    ConfigureCompositeComposite(out, 5005u, 111, 3122, false);
    ConfigureCompositeComposite(out, 5006u, 111, -3122, false);
    ConfigureCompositeComposite(out, 5007u, 111, 3312, false);
    ConfigureCompositeComposite(out, 5008u, 111, -3312, false);
    ConfigureCompositeComposite(out, 5009u, 300411, 310, false);
    ConfigureCompositeComposite(out, 5010u, -300411, 310, false);
    ConfigureCompositeComposite(out, 5011u, 300411, 3122, false);
    ConfigureCompositeComposite(out, 5012u, -300411, -3122, false);
    ConfigureCompositeComposite(out, 5013u, 300411, 3312, false);
    ConfigureCompositeComposite(out, 5014u, -300411, -3312, false);
    ConfigureCompositeComposite(out, 5015u, 400431, 310, false);
    ConfigureCompositeComposite(out, 5016u, -400431, 310, false);
    ConfigureCompositeComposite(out, 5017u, 504122, 310, false);
    ConfigureCompositeComposite(out, 5018u, -504122, 310, false);
    ConfigureCompositeComposite(out, 5019u, 22, 22, true);
    ConfigureCompositeComposite(out, 5020u, 3122, 3122, true);
    ConfigureCompositeComposite(out, 5021u, 111, 310, false);
    ConfigureCompositeComposite(out, 5022u, 3122, -3122, false);
    ConfigureCompositeComposite(out, 5023u, 3312, -3312, false);
    ConfigureCompositeComposite(out, 5024u, 3334, -3334, false);

    // These four children were intentionally held until the real pi0 parent
    // existed. They now consume channel 5019 as generation-3 work.
    for (KFParticleGpuCpuChannelContract& entry : out.entries) {
      if ((entry.channelId >= 3069u && entry.channelId <= 3072u)
          && entry.parentChannelId == 5019u) {
        entry.generation = FindChannel(out.entries, 5019u).generation + 1u;
        entry.supportStatus = KFGpuGraphSupported;
        entry.unsupportedReason = KFGpuGraphUnsupportedNone;
      }
    }

    next = 6001u;
    for (int mother : {7000211, 7000321, 7003112, 7003222, 7003312,
                       7003334, 9000321, 8003334, 8003222, 7003029,
                       7003006}) {
      out.AddPair(next, next + 1u, KFGpuCpuFamilyNeutralMissingMass, mother,
                  KFGpuGraphTopologyNeutralDaughter, 1u,
                  KFGpuGraphSourceTrackRange,
                  KFGpuGraphSourceTrackRange,
                  KFGpuGraphConstruct | KFGpuGraphMissingMass
                    | KFGpuGraphSelect,
                  KFGpuGraphOutputSecondary,
                  CommonInputs,
                  KFGpuGraphUnsupportedMissingMathematics);
      next += 2u;
    }

    ConfigureNeutralMissingMassPair(out, 6001u, 211, -13);
    ConfigureNeutralMissingMassPair(out, 6003u, 321, -13);
    ConfigureNeutralMissingMassPair(out, 6005u, 3112, -211);
    ConfigureNeutralMissingMassPair(out, 6007u, 3222, 211);
    ConfigureNeutralMissingMassPair(out, 6009u, 3312, -211);
    ConfigureNeutralMissingMassPair(out, 6011u, 3334, -211);
    ConfigureNeutralMissingMassPair(out, 6013u, 321, 211);
    ConfigureNeutralMissingMassPair(out, 6015u, 3334, -321);
    ConfigureNeutralMissingMassPair(out, 6017u, 3222, 2212);
    ConfigureNeutralMissingMassPair(out, 6019u, 7003029, -1000020030);
    ConfigureNeutralMissingMassPair(out, 6021u, 7003006, -1000020040);

    out.AddPair(7001u, 7002u, KFGpuCpuFamilyKaonMatching, 200321,
                KFGpuGraphTopologyCompositeTrack, 3u,
                KFGpuGraphSourceCandidateGeneration,
                KFGpuGraphSourceTrackRange,
                KFGpuGraphConstruct | KFGpuGraphMatch | KFGpuGraphSelect,
                KFGpuGraphOutputFinal, CommonInputs);
    ConfigureKaonMatchingPair(out, 7001u);

    next = 8001u;
    for (int mother : {310, 22, 111, 300443, 400443, 500443}) {
      out.AddSelf(next++, KFGpuCpuFamilyPrimaryProjection, mother,
                  KFGpuGraphTopologyUnaryComposite, 2u,
                  KFGpuGraphSourceCandidateGeneration, KFGpuGraphSourceNone,
                  KFGpuGraphExtrapolate | KFGpuGraphSetProductionVertex,
                  KFGpuGraphOutputPrimary, KFGpuCpuInputPrimaryVertices);
    }
    for (int mother : {3122, 3312, 3334}) {
      out.AddPair(next, next + 1u, KFGpuCpuFamilyPrimaryProjection, mother,
                  KFGpuGraphTopologyUnaryComposite, 2u,
                  KFGpuGraphSourceCandidateGeneration, KFGpuGraphSourceNone,
                  KFGpuGraphExtrapolate | KFGpuGraphSetProductionVertex,
                  KFGpuGraphOutputPrimary, KFGpuCpuInputPrimaryVertices);
      next += 2u;
    }

    ConfigurePrimaryProjection(out, 8001u, 1u);
    ConfigurePrimaryProjection(out, 8002u, 1001u);
    ConfigurePrimaryProjection(out, 8003u, 5019u);
    ConfigurePrimaryProjection(out, 8004u, 5022u);
    ConfigurePrimaryProjection(out, 8005u, 5023u);
    ConfigurePrimaryProjection(out, 8006u, 5024u);
    ConfigurePrimaryProjection(out, 8007u, 2u);
    ConfigurePrimaryProjection(out, 8008u, 3u);
    ConfigurePrimaryProjection(out, 8009u, 11u);
    ConfigurePrimaryProjection(out, 8010u, 12u);
    ConfigurePrimaryProjection(out, 8011u, 13u);
    ConfigurePrimaryProjection(out, 8012u, 14u);

    next = 9001u;
    for (int mother : {421, 411, 431, 4122, 425, 427, 200411,
                       300431, 204122}) {
      out.AddPair(next, next + 1u, KFGpuCpuFamilyFinalSelection, mother,
                  KFGpuGraphTopologyUnaryComposite, 4u,
                  KFGpuGraphSourceCandidateGeneration, KFGpuGraphSourceNone,
                  KFGpuGraphMassConstraint | KFGpuGraphSetProductionVertex
                    | KFGpuGraphSelect,
                  KFGpuGraphOutputFinal, CommonInputs);
      next += 2u;
    }
    for (unsigned int channelId = 9001u; channelId <= 9017u;
         channelId += 2u) {
      ConfigureFinalSelectionPair(out, channelId);
    }

    ValidateKFParticleGpuCpuChannelCatalogue(out.entries);
    return out.entries;
  }

  /// @brief Tests channel ID and mother PDG membership only; does not compare fit settings or numerical results.
  bool PlanImplements(const KFParticleGpuDecayPlan& plan,
                      const KFParticleGpuCpuChannelContract& contract)
  {
    for (std::size_t index = 0u; index < plan.NumberOfTwoDaughterChannels(); ++index) {
      const KFParticleGpuTwoDaughterChannel& channel = plan.TwoDaughterChannel(index);
      if (channel.channelId == contract.channelId
          && channel.motherPdg == contract.motherPdg) {
        return true;
      }
    }
    for (std::size_t index = 0u; index < plan.NumberOfV0TrackCascadeChannels(); ++index) {
      const KFParticleGpuV0TrackCascadeChannel& channel =
        plan.V0TrackCascadeChannel(index);
      if (channel.channelId == contract.channelId
          && channel.motherPdg == contract.motherPdg) {
        return true;
      }
    }
    for (std::size_t index = 0u; index < plan.NumberOfGraphOperationChannels(); ++index) {
      const KFParticleGpuGraphOperationChannel& channel =
        plan.GraphOperationChannel(index);
      if (channel.node.channelId == contract.channelId
          && channel.descriptor.motherPdg == contract.motherPdg) {
        return true;
      }
    }
    return false;
  }

  /// @brief Tests ID membership across all three plan channel collections, without checking the PDG or support.
  bool PlanContainsChannel(const KFParticleGpuDecayPlan& plan,
                           unsigned int channelId)
  {
    for (std::size_t index = 0u;
         index < plan.NumberOfTwoDaughterChannels(); ++index) {
      if (plan.TwoDaughterChannel(index).channelId == channelId) return true;
    }
    for (std::size_t index = 0u;
         index < plan.NumberOfV0TrackCascadeChannels(); ++index) {
      if (plan.V0TrackCascadeChannel(index).channelId == channelId) return true;
    }
    for (std::size_t index = 0u;
         index < plan.NumberOfGraphOperationChannels(); ++index) {
      if (plan.GraphOperationChannel(index).node.channelId == channelId) {
        return true;
      }
    }
    return false;
  }

  /// @brief Mixes the value's bytes, least significant first, into the noncryptographic catalogue hash.
  void Hash(unsigned long long& hash, unsigned long long value)
  {
    constexpr unsigned long long prime = 1099511628211ull;
    for (unsigned int byte = 0u; byte < sizeof(value); ++byte) {
      hash ^= (value >> (byte * 8u)) & 0xffu;
      hash *= prime;
    }
  }
}

const std::vector<KFParticleGpuCpuChannelContract>&
KFParticleGpuCpuChannelCatalogue()
{
  static const std::vector<KFParticleGpuCpuChannelContract> catalogue =
    BuildCatalogue();
  return catalogue;
}

void ValidateKFParticleGpuCpuChannelCatalogue(
  const std::vector<KFParticleGpuCpuChannelContract>& catalogue)
{
  std::unordered_map<unsigned int, const KFParticleGpuCpuChannelContract*> byId;
  std::array<unsigned int, KFGpuCpuFamilyCount> familyEntries = {};
  for (const KFParticleGpuCpuChannelContract& entry : catalogue) {
    if (entry.channelId == 0u || !byId.emplace(entry.channelId, &entry).second) {
      throw std::invalid_argument("KFParticle GPU CPU channel catalogue has a duplicate or zero channel ID");
    }
    if (entry.family >= KFGpuCpuFamilyCount
        || entry.activation >= KFGpuCpuChannelActivationCount
        || entry.motherPdg == 0
        || entry.topology == KFGpuGraphTopologyInvalid
        || entry.topology >= KFGpuGraphTopologyCount
        || entry.generation == 0u
        || entry.firstSourceKind == KFGpuGraphSourceNone
        || entry.firstSourceKind >= KFGpuGraphSourceKindCount
        || entry.secondSourceKind >= KFGpuGraphSourceKindCount
        || entry.operationMask == 0u
        || (entry.operationMask & ~KFGpuGraphKnownOperations) != 0u
        || entry.daughterHypothesisProfile == 0u
        || entry.selectionProfile == 0u
        || entry.outputClass == KFGpuGraphOutputInvalid
        || entry.outputClass >= KFGpuGraphOutputClassCount
        || entry.supportStatus >= KFGpuGraphSupportStatusCount
        || entry.unsupportedReason >= KFGpuGraphUnsupportedReasonCount) {
      throw std::invalid_argument("KFParticle GPU CPU channel catalogue contains an incomplete contract");
    }
    if ((entry.supportStatus == KFGpuGraphSupported)
        != (entry.unsupportedReason == KFGpuGraphUnsupportedNone)) {
      throw std::invalid_argument("KFParticle GPU CPU channel support status and reason disagree");
    }
    ++familyEntries[entry.family];
  }

  for (const KFParticleGpuCpuChannelContract& entry : catalogue) {
    const auto conjugate = byId.find(entry.conjugateChannelId);
    if (conjugate == byId.end()
        || conjugate->second->conjugateChannelId != entry.channelId
        || conjugate->second->family != entry.family
        || (conjugate->second->motherPdg != entry.motherPdg
            && conjugate->second->motherPdg != -entry.motherPdg)) {
      throw std::invalid_argument("KFParticle GPU CPU channel catalogue has a missing or inconsistent conjugate");
    }
    if (entry.parentChannelId != 0u) {
      const auto parent = byId.find(entry.parentChannelId);
      if (parent == byId.end() || parent->second->generation >= entry.generation) {
        throw std::invalid_argument(
          "KFParticle GPU CPU channel catalogue has an unresolved dependency for channel "
          + std::to_string(entry.channelId) + " on parent "
          + std::to_string(entry.parentChannelId));
      }
    }
    if (entry.secondParentChannelId != 0u) {
      const auto parent = byId.find(entry.secondParentChannelId);
      if (parent == byId.end() || parent->second->generation >= entry.generation) {
        throw std::invalid_argument(
          "KFParticle GPU CPU channel catalogue has an unresolved second dependency for channel "
          + std::to_string(entry.channelId) + " on parent "
          + std::to_string(entry.secondParentChannelId));
      }
    }
  }
  if (std::any_of(familyEntries.begin(), familyEntries.end(),
                  [](unsigned int count) { return count == 0u; })) {
    throw std::invalid_argument("KFParticle GPU CPU channel catalogue omits a finder family");
  }
}

std::vector<KFParticleGpuCpuChannelCoverage>
MakeCpuFinderChannelCoverage(const KFParticleGpuDecayPlan& plan)
{
  std::vector<KFParticleGpuCpuChannelCoverage> coverage;
  coverage.reserve(KFParticleGpuCpuChannelCatalogue().size());
  for (const KFParticleGpuCpuChannelContract& entry :
       KFParticleGpuCpuChannelCatalogue()) {
    KFParticleGpuCpuChannelCoverage item;
    item.channelId = entry.channelId;
    item.family = entry.family;
    item.activation = entry.activation;
    if (PlanImplements(plan, entry)) {
      item.supportStatus = entry.supportStatus;
      item.unsupportedReason = entry.unsupportedReason;
    }
    else {
      item.supportStatus = KFGpuGraphUnsupported;
      item.unsupportedReason =
        entry.unsupportedReason == KFGpuGraphUnsupportedNone
          ? KFGpuGraphUnsupportedNotImplemented : entry.unsupportedReason;
    }
    coverage.push_back(item);
  }
  return coverage;
}

std::vector<KFParticleGpuGraphFamilyCoverage>
MakeCpuFinderFamilyCoverage(
  const std::array<unsigned int, KFGpuCpuFamilyCount>& implementedCounts)
{
  std::array<unsigned int, KFGpuCpuFamilyCount> activeCounts = {};
  std::array<unsigned int, KFGpuCpuFamilyCount> conditionalCounts = {};
  std::array<unsigned int, KFGpuCpuFamilyCount> disabledCounts = {};
  std::array<unsigned int, KFGpuCpuFamilyCount> unsupportedReasons = {};
  for (const KFParticleGpuCpuChannelContract& entry :
       KFParticleGpuCpuChannelCatalogue()) {
    if (entry.activation == KFGpuCpuChannelActive) {
      ++activeCounts[entry.family];
    }
    else if (entry.activation == KFGpuCpuChannelConditionalInput) {
      ++conditionalCounts[entry.family];
    }
    else if (entry.activation == KFGpuCpuChannelConfigurationDisabled) {
      ++disabledCounts[entry.family];
    }
    if (entry.unsupportedReason != KFGpuGraphUnsupportedNone
        && unsupportedReasons[entry.family] == KFGpuGraphUnsupportedNone) {
      unsupportedReasons[entry.family] = entry.unsupportedReason;
    }
  }

  std::vector<KFParticleGpuGraphFamilyCoverage> result;
  result.reserve(KFGpuCpuFamilyCount);
  for (unsigned int family = 0u; family < KFGpuCpuFamilyCount; ++family) {
    KFParticleGpuGraphFamilyCoverage coverage;
    coverage.family = family;
    coverage.implementedChannelCount = implementedCounts[family];
    const unsigned int required = activeCounts[family] + conditionalCounts[family];
    if ((required != 0u && implementedCounts[family] >= required)
        || (required == 0u && disabledCounts[family] != 0u)) {
      coverage.supportStatus = KFGpuGraphSupported;
      coverage.unsupportedReason = KFGpuGraphUnsupportedNone;
    }
    else if (implementedCounts[family] != 0u) {
      coverage.supportStatus = KFGpuGraphPartiallySupported;
      coverage.unsupportedReason = KFGpuGraphUnsupportedValidationPending;
    }
    else {
      coverage.supportStatus = KFGpuGraphUnsupported;
      coverage.unsupportedReason = unsupportedReasons[family] != KFGpuGraphUnsupportedNone
        ? unsupportedReasons[family]
        : (conditionalCounts[family] != 0u
             ? KFGpuGraphUnsupportedMissingInput
             : KFGpuGraphUnsupportedNotImplemented);
    }
    result.push_back(coverage);
  }
  return result;
}

void AddCpuFinderTwoDaughterChannels(KFParticleGpuDecayPlan& plan)
{
  for (const KFParticleGpuCpuChannelContract& entry :
       KFParticleGpuCpuChannelCatalogue()) {
    if (entry.family == KFGpuCpuFamilyTwoDaughter
        && entry.activation == KFGpuCpuChannelActive) {
      plan.AddTwoDaughterChannel(MakeCpuTwoDaughterChannel(entry));
    }
  }
}

void AddCpuFinderCompositeTrackChannels(KFParticleGpuDecayPlan& plan)
{
  for (const KFParticleGpuCpuChannelContract& entry :
       KFParticleGpuCpuChannelCatalogue()) {
    if ((entry.family == KFGpuCpuFamilyTrackComposite
         || entry.family == KFGpuCpuFamilyLongLivedComposite)
        && entry.activation == KFGpuCpuChannelActive
        && entry.supportStatus == KFGpuGraphSupported
        && (entry.parentChannelId != 5019u
            || PlanContainsChannel(plan, 5019u))) {
      plan.AddV0TrackCascadeChannel(MakeCpuCompositeTrackChannel(entry));
    }
  }
}

void AddCpuFinderCompositeCompositeChannels(KFParticleGpuDecayPlan& plan)
{
  const std::vector<KFParticleGpuCpuChannelContract>& catalogue =
    KFParticleGpuCpuChannelCatalogue();
  for (const KFParticleGpuCpuChannelContract& entry : catalogue) {
    if (entry.family == KFGpuCpuFamilyCompositeComposite
        && entry.activation == KFGpuCpuChannelActive
        && entry.supportStatus == KFGpuGraphSupported) {
      plan.AddGraphOperationChannel(
        MakeCpuCompositeCompositeChannel(catalogue, entry));
    }
  }
}

void AddCpuFinderPrimaryProjectionChannels(KFParticleGpuDecayPlan& plan)
{
  const std::vector<KFParticleGpuCpuChannelContract>& catalogue =
    KFParticleGpuCpuChannelCatalogue();
  for (const KFParticleGpuCpuChannelContract& entry : catalogue) {
    if (entry.family == KFGpuCpuFamilyPrimaryProjection
        && entry.activation == KFGpuCpuChannelActive
        && entry.supportStatus == KFGpuGraphSupported
        && PlanContainsChannel(plan, entry.parentChannelId)) {
      plan.AddGraphOperationChannel(
        MakeCpuPrimaryProjectionChannel(catalogue, entry));
    }
  }
}

void AddCpuFinderNeutralMissingMassChannels(KFParticleGpuDecayPlan& plan)
{
  for (const KFParticleGpuCpuChannelContract& entry :
       KFParticleGpuCpuChannelCatalogue()) {
    if (entry.family == KFGpuCpuFamilyNeutralMissingMass
        && entry.activation == KFGpuCpuChannelActive
        && entry.supportStatus == KFGpuGraphSupported) {
      plan.AddGraphOperationChannel(MakeCpuNeutralMissingMassChannel(entry));
    }
  }
}

void AddCpuFinderKaonMatchingChannels(KFParticleGpuDecayPlan& plan)
{
  const std::vector<KFParticleGpuCpuChannelContract>& catalogue =
    KFParticleGpuCpuChannelCatalogue();
  for (const KFParticleGpuCpuChannelContract& entry : catalogue) {
    if (entry.family == KFGpuCpuFamilyKaonMatching
        && entry.activation == KFGpuCpuChannelActive
        && entry.supportStatus == KFGpuGraphSupported
        && PlanContainsChannel(plan, entry.parentChannelId)) {
      plan.AddGraphOperationChannel(
        MakeCpuKaonMatchingChannel(catalogue, entry));
    }
  }
}

void AddCpuFinderFinalSelectionChannels(KFParticleGpuDecayPlan& plan)
{
  const std::vector<KFParticleGpuCpuChannelContract>& catalogue =
    KFParticleGpuCpuChannelCatalogue();
  for (const KFParticleGpuCpuChannelContract& entry : catalogue) {
    if (entry.family == KFGpuCpuFamilyFinalSelection
        && entry.activation == KFGpuCpuChannelActive
        && entry.supportStatus == KFGpuGraphSupported
        && PlanContainsChannel(plan, entry.parentChannelId)) {
      plan.AddGraphOperationChannel(
        MakeCpuFinalSelectionChannel(catalogue, entry));
    }
  }
}

void AddCompleteCpuFinderChannels(KFParticleGpuDecayPlan& plan)
{
  AddCpuFinderTwoDaughterChannels(plan);
  AddCpuFinderCompositeCompositeChannels(plan);
  AddCpuFinderCompositeTrackChannels(plan);
  AddCpuFinderPrimaryProjectionChannels(plan);
  AddCpuFinderNeutralMissingMassChannels(plan);
  AddCpuFinderKaonMatchingChannels(plan);
  AddCpuFinderFinalSelectionChannels(plan);
}

void AddRequestedCpuFinderChannels(
  KFParticleGpuDecayPlan& plan,
  const std::vector<int>& requestedMotherPdgs)
{
  const auto& catalogue = KFParticleGpuCpuChannelCatalogue();
  std::unordered_map<unsigned int, const KFParticleGpuCpuChannelContract*> byId;
  byId.reserve(catalogue.size());
  for (const auto& entry : catalogue) { byId.emplace(entry.channelId, &entry); }

  std::unordered_set<unsigned int> selected;
  // Follow both parent links once per channel, including shared intermediate outputs.
  const auto selectWithParents = [&](auto&& self, unsigned int channelId) -> void {
    if (channelId == 0u || !selected.insert(channelId).second) { return; }
    const auto found = byId.find(channelId);
    if (found == byId.end()) {
      throw std::logic_error("KFParticle GPU requested CPU plan has an unknown dependency");
    }
    self(self, found->second->parentChannelId);
    self(self, found->second->secondParentChannelId);
  };
  for (const auto& entry : catalogue) {
    if (entry.activation != KFGpuCpuChannelConfigurationDisabled
        && entry.supportStatus == KFGpuGraphSupported
        && std::find(requestedMotherPdgs.begin(), requestedMotherPdgs.end(),
                     entry.motherPdg) != requestedMotherPdgs.end()) {
      selectWithParents(selectWithParents, entry.channelId);
    }
  }

  KFParticleGpuDecayPlan complete;
  AddCompleteCpuFinderChannels(complete);
  for (std::size_t index = 0u; index < complete.NumberOfTwoDaughterChannels(); ++index) {
    const auto& channel = complete.TwoDaughterChannel(index);
    if (selected.count(channel.channelId) != 0u) { plan.AddTwoDaughterChannel(channel); }
  }
  for (std::size_t index = 0u; index < complete.NumberOfV0TrackCascadeChannels(); ++index) {
    const auto& channel = complete.V0TrackCascadeChannel(index);
    if (selected.count(channel.channelId) != 0u) { plan.AddV0TrackCascadeChannel(channel); }
  }
  for (std::size_t index = 0u; index < complete.NumberOfGraphOperationChannels(); ++index) {
    const auto& channel = complete.GraphOperationChannel(index);
    if (selected.count(channel.node.channelId) != 0u) { plan.AddGraphOperationChannel(channel); }
  }
}

KFParticleGpuCpuFinderCoverageSummary
MakeCompleteCpuFinderCoverageSummary(const KFParticleGpuDecayPlan& plan)
{
  KFParticleGpuCpuFinderCoverageSummary summary;
  const std::vector<KFParticleGpuCpuChannelCoverage> coverage =
    MakeCpuFinderChannelCoverage(plan);
  summary.catalogueChannels = static_cast<unsigned int>(coverage.size());
  for (const KFParticleGpuCpuChannelCoverage& item : coverage) {
    if (item.activation == KFGpuCpuChannelConfigurationDisabled) {
      ++summary.disabledChannels;
      ++summary.disabledByFamily[item.family];
      continue;
    }
    ++summary.requiredChannels;
    ++summary.requiredByFamily[item.family];
    if (item.supportStatus == KFGpuGraphSupported
        && item.unsupportedReason == KFGpuGraphUnsupportedNone) {
      ++summary.implementedChannels;
      ++summary.implementedByFamily[item.family];
    }
    else {
      ++summary.unsupportedChannels;
    }
  }
  return summary;
}

void ValidateCompleteCpuFinderPlan(const KFParticleGpuDecayPlan& plan)
{
  const KFParticleGpuCpuFinderCoverageSummary summary =
    MakeCompleteCpuFinderCoverageSummary(plan);
  if (!summary.Complete()) {
    throw std::invalid_argument(
      "KFParticle GPU complete CPU finder plan omits a declared active channel");
  }
  for (unsigned int family = 0u; family < KFGpuCpuFamilyCount; ++family) {
    if (summary.requiredByFamily[family]
        != summary.implementedByFamily[family]) {
      throw std::invalid_argument(
        "KFParticle GPU complete CPU finder plan has incomplete family coverage");
    }
  }
}

unsigned int KFParticleGpuCpuFinderRawTaskCapacity(
  const KFParticleGpuEventDesc& event,
  const KFParticleGpuDecayPlan& plan)
{
  unsigned int total = 0u;
  for (std::size_t index = 0u;
       index < plan.NumberOfTwoDaughterChannels(); ++index) {
    const KFParticleGpuTwoDaughterChannel& channel =
      plan.TwoDaughterChannel(index);
    const KFParticleGpuTrackSetDesc& firstSet =
      event.TrackSet(channel.firstTrackSet);
    const KFParticleGpuTrackSetDesc& secondSet =
      event.TrackSet(channel.secondTrackSet);
    const KFParticleGpuRange first =
      channel.firstSpecies == NumberOfTrackSpecies
        ? firstSet.tracks : firstSet.Species(channel.firstSpecies);
    const KFParticleGpuRange second =
      channel.secondSpecies == NumberOfTrackSpecies
        ? secondSet.tracks : secondSet.Species(channel.secondSpecies);
    CheckedAccumulate(CheckedPairCount(first.size, second.size), total);
  }
  for (std::size_t index = 0u;
       index < plan.NumberOfGraphOperationChannels(); ++index) {
    const KFParticleGpuGraphNode& node =
      plan.GraphOperationChannel(index).node;
    if (node.firstSource.kind != KFGpuGraphSourceTrackRange
        || node.secondSource.kind != KFGpuGraphSourceTrackRange) {
      continue;
    }
    const KFParticleGpuRange first =
      ResolveTrackSourceRange(event, node.firstSource.sourceId);
    const KFParticleGpuRange second =
      ResolveTrackSourceRange(event, node.secondSource.sourceId);
    CheckedAccumulate(CheckedPairCount(first.size, second.size), total);
  }
  return total;
}

/**
 * @brief Owned host workspace and cached dependency slots for one immutable, borrowed decay plan.
 *
 * Stable channel IDs are mapped to dense array indices during construction.
 * Event stamps distinguish current bounds from stale values without clearing
 * the entire workspace. These stamps are unrelated to decay-graph generations.
 * No device memory, queue, or synchronization primitive is held here.
 */
struct KFParticleGpuCpuFinderCapacityPlanner::Impl
{
  /** @brief Candidate slot bound and maximum flattened lineage length for a single channel. */
  struct ChannelBound
  {
    unsigned int candidates = 0u;  ///< Candidate slots predicted by the bounded planning policy.
    unsigned int lineage = 0u;     ///< Source-track IDs per candidate, not a total daughter count.
  };

  /**
   * @brief Borrows the plan, records its revision, and caches output/dependency slots.
   * Missing candidate sources are represented by InvalidSlot(); plan validation is external.
   */
  explicit Impl(const KFParticleGpuDecayPlan& configuredPlan)
    : plan(&configuredPlan), revision(configuredPlan.Revision())
  {
    const std::size_t outputCount =
      configuredPlan.NumberOfTwoDaughterChannels()
      + configuredPlan.NumberOfV0TrackCascadeChannels()
      + configuredPlan.NumberOfGraphOperationChannels();
    channelSlots.reserve(outputCount);
    bounds.reserve(outputCount);
    generations.reserve(outputCount);

    for (std::size_t index = 0u;
         index < configuredPlan.NumberOfTwoDaughterChannels(); ++index) {
      EnsureSlot(configuredPlan.TwoDaughterChannel(index).channelId);
    }
    for (std::size_t index = 0u;
         index < configuredPlan.NumberOfV0TrackCascadeChannels(); ++index) {
      EnsureSlot(configuredPlan.V0TrackCascadeChannel(index).channelId);
    }
    for (std::size_t index = 0u;
         index < configuredPlan.NumberOfGraphOperationChannels(); ++index) {
      EnsureSlot(configuredPlan.GraphOperationChannel(index).node.channelId);
    }

    twoDaughterOutputSlots.reserve(
      configuredPlan.NumberOfTwoDaughterChannels());
    for (std::size_t index = 0u;
         index < configuredPlan.NumberOfTwoDaughterChannels(); ++index) {
      twoDaughterOutputSlots.push_back(
        Slot(configuredPlan.TwoDaughterChannel(index).channelId));
    }

    cascadeParentSlots.reserve(
      configuredPlan.NumberOfV0TrackCascadeChannels());
    cascadeOutputSlots.reserve(
      configuredPlan.NumberOfV0TrackCascadeChannels());
    for (std::size_t index = 0u;
         index < configuredPlan.NumberOfV0TrackCascadeChannels(); ++index) {
      const auto& channel = configuredPlan.V0TrackCascadeChannel(index);
      cascadeParentSlots.push_back(SlotOrInvalid(channel.parentChannelId));
      cascadeOutputSlots.push_back(Slot(channel.channelId));
    }

    graphFirstSourceSlots.reserve(
      configuredPlan.NumberOfGraphOperationChannels());
    graphSecondSourceSlots.reserve(
      configuredPlan.NumberOfGraphOperationChannels());
    graphOutputSlots.reserve(configuredPlan.NumberOfGraphOperationChannels());
    for (std::size_t index = 0u;
         index < configuredPlan.NumberOfGraphOperationChannels(); ++index) {
      const auto& node = configuredPlan.GraphOperationChannel(index).node;
      graphFirstSourceSlots.push_back(SourceSlot(node.firstSource));
      graphSecondSourceSlots.push_back(SourceSlot(node.secondSource));
      graphOutputSlots.push_back(Slot(node.channelId));
    }
  }

  /// @brief Returns the sentinel for an absent candidate source, never a valid dense index.
  static unsigned int InvalidSlot()
  {
    return std::numeric_limits<unsigned int>::max();
  }

  /// @brief Creates a zero-stamped output slot only if the channel ID has not already been registered.
  void EnsureSlot(unsigned int channelId)
  {
    const unsigned int slot = static_cast<unsigned int>(bounds.size());
    if (!channelSlots.emplace(channelId, slot).second) { return; }
    bounds.push_back({});
    generations.push_back(0u);
  }

  /// @brief Looks up a dense slot by stable channel ID, or returns InvalidSlot() when absent.
  unsigned int SlotOrInvalid(unsigned int channelId) const
  {
    const auto found = channelSlots.find(channelId);
    return found == channelSlots.end() ? InvalidSlot() : found->second;
  }

  /**
   * @brief Returns the dense slot of a registered output channel.
   * @throws std::logic_error If construction failed to register the requested output.
   */
  unsigned int Slot(unsigned int channelId) const
  {
    const unsigned int slot = SlotOrInvalid(channelId);
    if (slot == InvalidSlot()) {
      throw std::logic_error(
        "KFParticle GPU capacity planner lost an output channel");
    }
    return slot;
  }

  /// @brief Resolves candidate-generation sources to slots; raw-track, absent, and unknown sources use InvalidSlot().
  unsigned int SourceSlot(const KFParticleGpuGraphSource& source) const
  {
    return source.kind == KFGpuGraphSourceCandidateGeneration
             ? SlotOrInvalid(source.sourceId) : InvalidSlot();
  }

  /// @brief Advances the event stamp; clears all slot stamps only when the unsigned stamp wraps around.
  void BeginEvent()
  {
    ++generation;
    if (generation == 0u) {
      std::fill(generations.begin(), generations.end(), 0u);
      generation = 1u;
    }
  }

  /// @brief Writes a bound and marks it current; the caller supplies a valid, unchecked dense slot.
  void Store(unsigned int slot, ChannelBound bound)
  {
    bounds[slot] = bound;
    generations[slot] = generation;
  }

  /**
   * @brief Loads a current-event bound; returns false and a zero bound for absent or stale slots.
   * @pre The index is InvalidSlot() or a valid dense slot; other out-of-range indices are not checked.
   */
  bool Load(unsigned int slot, ChannelBound& bound) const
  {
    if (slot == InvalidSlot() || generations[slot] != generation) {
      bound = {};
      return false;
    }
    bound = bounds[slot];
    return true;
  }

  const KFParticleGpuDecayPlan* plan = nullptr;  ///< Borrowed plan; must outlive this workspace unchanged.
  unsigned long long revision = 0u;             ///< Plan revision captured when the indices were compiled.
  std::unordered_map<unsigned int, unsigned int> channelSlots;  ///< Stable output ID to dense slot lookup.
  std::vector<ChannelBound> bounds;              ///< Reusable per-channel bounds; stale entries are ignored.
  std::vector<unsigned int> generations;        ///< Per-slot event stamps, not decay generations.
  unsigned int generation = 0u;                ///< Current event stamp; zero is reserved for stale slots.
  std::vector<unsigned int> twoDaughterOutputSlots;  ///< Output slots in two-daughter plan order.
  std::vector<unsigned int> cascadeParentSlots;      ///< Parent slots in cascade plan order; may be InvalidSlot().
  std::vector<unsigned int> cascadeOutputSlots;      ///< Output slots in cascade plan order.
  std::vector<unsigned int> graphFirstSourceSlots;   ///< First candidate-source slots in graph-operation order.
  std::vector<unsigned int> graphSecondSourceSlots;  ///< Second candidate-source slots in graph-operation order.
  std::vector<unsigned int> graphOutputSlots;        ///< Output slots in graph-operation order.
};

KFParticleGpuCpuFinderCapacityPlanner::KFParticleGpuCpuFinderCapacityPlanner(
  const KFParticleGpuDecayPlan& plan)
  : fImpl(std::make_unique<Impl>(plan))
{
}

KFParticleGpuCpuFinderCapacityPlanner::~KFParticleGpuCpuFinderCapacityPlanner() = default;

KFParticleGpuEventCapacityPlan KFParticleGpuCpuFinderCapacityPlanner::Plan(
  const KFParticleGpuEventDesc& event)
{
  if (fImpl->plan->Revision() != fImpl->revision) {
    throw std::logic_error(
      "KFParticle GPU decay plan changed after capacity planner compilation");
  }
  fImpl->BeginEvent();
  const KFParticleGpuDecayPlan& plan = *fImpl->plan;
  KFParticleGpuEventCapacityPlan result;
  result.taskCapacity = KFParticleGpuCpuFinderRawTaskCapacity(event, plan);

  // Account for persistent outputs while retaining bounds for downstream channels.
  const auto addOutput = [&](unsigned int slot, unsigned int candidates,
                             unsigned int lineage, unsigned int& familyTotal) {
    CheckedAccumulate(candidates, familyTotal);
    CheckedAccumulate(candidates, result.candidates);
    CheckedAccumulate(CheckedPairCount(candidates, lineage), result.daughterIds);
    fImpl->Store(slot, {candidates, lineage});
  };

  const auto trackRange = [&](KFParticleGpuTrackSet set,
                              KFParticleGpuTrackSpecies species) {
    const KFParticleGpuTrackSetDesc& tracks = event.TrackSet(set);
    return species == NumberOfTrackSpecies ? tracks.tracks
                                           : tracks.Species(species);
  };

  for (std::size_t index = 0u; index < plan.NumberOfTwoDaughterChannels(); ++index) {
    const KFParticleGpuTwoDaughterChannel& channel = plan.TwoDaughterChannel(index);
    const unsigned int candidates = CheckedPairCount(
      trackRange(channel.firstTrackSet, channel.firstSpecies).size,
      trackRange(channel.secondTrackSet, channel.secondSpecies).size);
    addOutput(fImpl->twoDaughterOutputSlots[index], candidates, 2u,
              result.twoDaughterCandidates);
  }

  for (std::size_t index = 0u; index < plan.NumberOfV0TrackCascadeChannels(); ++index) {
    const KFParticleGpuV0TrackCascadeChannel& channel =
      plan.V0TrackCascadeChannel(index);
    Impl::ChannelBound parent;
    const bool hasParent = fImpl->Load(fImpl->cascadeParentSlots[index], parent);
    const unsigned int parentCandidates = hasParent ? parent.candidates : 0u;
    const unsigned int parentLineage = hasParent ? parent.lineage : 2u;
    const unsigned int candidates = CheckedPairCount(
      parentCandidates,
      trackRange(channel.bachelorTrackSet, channel.bachelorSpecies).size);
    const unsigned int lineage = parentLineage + 1u;
    addOutput(fImpl->cascadeOutputSlots[index], candidates,
              lineage < KFParticleGpuV0TrackLineage::MaximumSize
                ? lineage : KFParticleGpuV0TrackLineage::MaximumSize,
              result.cascadeCandidates);
  }

  const auto sourceBound = [&](const KFParticleGpuGraphSource& source,
                               unsigned int slot) {
    if (source.kind == KFGpuGraphSourceTrackRange) {
      return Impl::ChannelBound{
        ResolveTrackSourceRange(event, source.sourceId).size, 1u};
    }
    if (source.kind == KFGpuGraphSourceCandidateGeneration) {
      Impl::ChannelBound bound;
      fImpl->Load(slot, bound);
      return bound;
    }
    return Impl::ChannelBound{};
  };

  // Bound each graph channel, then size reusable task storage for the busiest generation.
  result.taskCapacity = std::max(result.taskCapacity,
                                 result.cascadeCandidates);
  const unsigned int graphChannelTaskLimit = result.taskCapacity;
  unsigned int maximumGenerationTasks = result.taskCapacity;
  unsigned int activeGeneration = 0u;
  unsigned int generationTasks = 0u;
  for (std::size_t index = 0u; index < plan.NumberOfGraphOperationChannels(); ++index) {
    const KFParticleGpuGraphOperationChannel& channel =
      plan.GraphOperationChannel(index);
    if (channel.node.generation != activeGeneration) {
      maximumGenerationTasks = std::max(maximumGenerationTasks, generationTasks);
      generationTasks = 0u;
      activeGeneration = channel.node.generation;
    }
    const Impl::ChannelBound first = sourceBound(
      channel.node.firstSource, fImpl->graphFirstSourceSlots[index]);
    const Impl::ChannelBound second = sourceBound(
      channel.node.secondSource, fImpl->graphSecondSourceSlots[index]);
    const bool unary = channel.node.secondSource.kind == KFGpuGraphSourceNone;
    const unsigned int candidates = unary
      ? std::min(first.candidates, graphChannelTaskLimit)
      : BoundedPairCount(first.candidates, second.candidates,
                         graphChannelTaskLimit);
    const unsigned int combinedLineage = first.lineage + second.lineage;
    const unsigned int lineage = unary ? first.lineage
      : (combinedLineage < KFParticleGpuV0TrackLineage::MaximumSize
           ? combinedLineage : KFParticleGpuV0TrackLineage::MaximumSize);
    CheckedAccumulate(candidates, generationTasks);
    addOutput(fImpl->graphOutputSlots[index], candidates, lineage,
              result.graphCandidates);
  }
  maximumGenerationTasks = std::max(maximumGenerationTasks, generationTasks);
  result.taskCapacity = std::max(maximumGenerationTasks,
                                 result.cascadeCandidates);
  result.selectedCandidates = result.candidates;
  return result;
}

KFParticleGpuEventCapacityPlan KFParticleGpuCpuFinderEventCapacityPlan(
  const KFParticleGpuEventDesc& event,
  const KFParticleGpuDecayPlan& plan)
{
  /** @brief Per-channel bound used by the one-shot map-based capacity calculation. */
  struct ChannelBound
  {
    unsigned int candidates = 0u;  ///< Candidate slots under the bounded planning policy.
    unsigned int lineage = 0u;     ///< Flattened source-track IDs per candidate.
  };
  std::unordered_map<unsigned int, ChannelBound> bounds;
  KFParticleGpuEventCapacityPlan result;
  result.taskCapacity = KFParticleGpuCpuFinderRawTaskCapacity(event, plan);

  const auto addOutput = [&](unsigned int channelId, unsigned int candidates,
                             unsigned int lineage, unsigned int& familyTotal) {
    CheckedAccumulate(candidates, familyTotal);
    CheckedAccumulate(candidates, result.candidates);
    CheckedAccumulate(CheckedPairCount(candidates, lineage), result.daughterIds);
    bounds[channelId] = {candidates, lineage};
  };

  const auto trackRange = [&](KFParticleGpuTrackSet set,
                              KFParticleGpuTrackSpecies species) {
    const KFParticleGpuTrackSetDesc& tracks = event.TrackSet(set);
    return species == NumberOfTrackSpecies ? tracks.tracks
                                           : tracks.Species(species);
  };

  for (std::size_t index = 0u; index < plan.NumberOfTwoDaughterChannels(); ++index) {
    const KFParticleGpuTwoDaughterChannel& channel = plan.TwoDaughterChannel(index);
    const unsigned int candidates = CheckedPairCount(
      trackRange(channel.firstTrackSet, channel.firstSpecies).size,
      trackRange(channel.secondTrackSet, channel.secondSpecies).size);
    addOutput(channel.channelId, candidates, 2u, result.twoDaughterCandidates);
  }

  for (std::size_t index = 0u; index < plan.NumberOfV0TrackCascadeChannels(); ++index) {
    const KFParticleGpuV0TrackCascadeChannel& channel =
      plan.V0TrackCascadeChannel(index);
    const auto parent = bounds.find(channel.parentChannelId);
    const unsigned int parentCandidates =
      parent == bounds.end() ? 0u : parent->second.candidates;
    const unsigned int parentLineage =
      parent == bounds.end() ? 2u : parent->second.lineage;
    const unsigned int candidates = CheckedPairCount(
      parentCandidates,
      trackRange(channel.bachelorTrackSet, channel.bachelorSpecies).size);
    const unsigned int lineage = parentLineage + 1u;
    addOutput(channel.channelId, candidates,
              lineage < KFParticleGpuV0TrackLineage::MaximumSize
                ? lineage : KFParticleGpuV0TrackLineage::MaximumSize,
              result.cascadeCandidates);
  }

  const auto sourceBound = [&](const KFParticleGpuGraphSource& source) {
    if (source.kind == KFGpuGraphSourceTrackRange) {
      return ChannelBound{ResolveTrackSourceRange(event, source.sourceId).size, 1u};
    }
    if (source.kind == KFGpuGraphSourceCandidateGeneration) {
      const auto found = bounds.find(source.sourceId);
      return found == bounds.end() ? ChannelBound{} : found->second;
    }
    return ChannelBound{};
  };

  result.taskCapacity = std::max(result.taskCapacity,
                                 result.cascadeCandidates);
  const unsigned int graphChannelTaskLimit = result.taskCapacity;
  unsigned int maximumGenerationTasks = result.taskCapacity;
  unsigned int activeGeneration = 0u;
  unsigned int generationTasks = 0u;
  for (std::size_t index = 0u; index < plan.NumberOfGraphOperationChannels(); ++index) {
    const KFParticleGpuGraphOperationChannel& channel =
      plan.GraphOperationChannel(index);
    if (channel.node.generation != activeGeneration) {
      maximumGenerationTasks = std::max(maximumGenerationTasks, generationTasks);
      generationTasks = 0u;
      activeGeneration = channel.node.generation;
    }
    const ChannelBound first = sourceBound(channel.node.firstSource);
    const ChannelBound second = sourceBound(channel.node.secondSource);
    const bool unary = channel.node.secondSource.kind == KFGpuGraphSourceNone;
    const unsigned int candidates = unary
      ? std::min(first.candidates, graphChannelTaskLimit)
      : BoundedPairCount(first.candidates, second.candidates,
                         graphChannelTaskLimit);
    const unsigned int combinedLineage = first.lineage + second.lineage;
    const unsigned int lineage = unary ? first.lineage
      : (combinedLineage < KFParticleGpuV0TrackLineage::MaximumSize
           ? combinedLineage : KFParticleGpuV0TrackLineage::MaximumSize);
    CheckedAccumulate(candidates, generationTasks);
    addOutput(channel.node.channelId, candidates, lineage, result.graphCandidates);
  }
  maximumGenerationTasks = std::max(maximumGenerationTasks, generationTasks);
  result.taskCapacity = std::max(maximumGenerationTasks,
                                 result.cascadeCandidates);
  result.selectedCandidates = result.candidates;
  return result;
}

unsigned long long KFParticleGpuCpuChannelCatalogueRevision()
{
  unsigned long long hash = 1469598103934665603ull;
  for (const KFParticleGpuCpuChannelContract& entry :
       KFParticleGpuCpuChannelCatalogue()) {
    Hash(hash, entry.channelId);
    Hash(hash, entry.conjugateChannelId);
    Hash(hash, entry.family);
    Hash(hash, static_cast<unsigned int>(entry.motherPdg));
    Hash(hash, static_cast<unsigned int>(entry.firstDaughterPdg));
    Hash(hash, static_cast<unsigned int>(entry.secondDaughterPdg));
    Hash(hash, entry.topology);
    Hash(hash, entry.generation);
    Hash(hash, entry.firstSourceKind);
    Hash(hash, entry.secondSourceKind);
    Hash(hash, entry.parentChannelId);
    Hash(hash, entry.secondParentChannelId);
    Hash(hash, entry.operationMask);
    Hash(hash, entry.daughterHypothesisProfile);
    Hash(hash, entry.selectionProfile);
    Hash(hash, entry.outputClass);
    Hash(hash, entry.requiredInputs);
    Hash(hash, entry.activation);
    Hash(hash, entry.supportStatus);
    Hash(hash, entry.unsupportedReason);
  }
  return hash;
}
