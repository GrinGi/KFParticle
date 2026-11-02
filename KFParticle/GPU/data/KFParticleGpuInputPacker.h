/* Copyright (C) 2026 GSI Helmholtzzentrum fuer Schwerionenforschung, Darmstadt
   SPDX-License-Identifier: GPL-3.0-only */

#ifndef KFPARTICLEGPUINPUTPACKER_H
#define KFPARTICLEGPUINPUTPACKER_H

#include "execution/KFParticleGpuBufferManager.h"

#include <memory>
#include <vector>

class KFPTrackVector;
class KFVertex;

namespace kfparticle::gpu
{
  /** Borrowed CPU track states and vertices for one KFParticle event. */
  struct GpuEventInput {
    unsigned int eventId = 0;
    const KFPTrackVector* first = nullptr;
    const KFPTrackVector* last = nullptr;
    const std::vector<float>* chiToPrimaryVertex = nullptr;
    const std::vector<KFVertex>* primaryVertices = nullptr;
    float minSecondaryTrackChiToPrimaryVertex = -1.f;
  };

  /**
   * Structurally validated source and final bucket ranges for one event.
   *
   * The source vectors remain owned by the caller and must outlive preparation,
   * sealing, and the direct write into the XPU host views.
   */
  class GpuPreparedEvent {
   public:
    GpuPreparedEvent(GpuPreparedEvent&&) noexcept;
    GpuPreparedEvent& operator=(GpuPreparedEvent&&) noexcept;
    ~GpuPreparedEvent();

    GpuPreparedEvent(const GpuPreparedEvent&)            = delete;
    GpuPreparedEvent& operator=(const GpuPreparedEvent&) = delete;

    unsigned int TrackCount() const;
    unsigned int VertexCount() const;
    const KFParticleGpuEventDesc& Descriptor() const;
    /** The packer proved paired first/last source IDs and event-local uniqueness. */
    bool HasValidatedPublicationInput() const;

   private:
    friend class GpuInputPacker;
    struct Impl;

    explicit GpuPreparedEvent(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> fImpl;
  };

  /** Converts CPU KFP input into the flat GPU input ABI. */
  class GpuInputPacker {
   public:
    static GpuPreparedEvent Prepare(const GpuEventInput& input);
    static KFParticleGpuBufferCapacities InputCapacities(const GpuPreparedEvent& prepared);

    /// Validates numerical payload while writing it into allocated host views.
    static void Write(const GpuPreparedEvent& prepared,
                      const KFParticleGpuInputTrackSoAView& tracks,
                      const KFParticleGpuVertexSoAView& vertices,
                      KFParticleGpuEventDesc& event);

    /**
     * Writes one event into absolute regions of a sealed multi-event host batch.
     * Converts each first/last local PV association with the core KFParticle helper;
     * negative values remain unassigned. Invalid local or overflowing indices throw.
     */
    static void Write(const GpuPreparedEvent& prepared,
                      unsigned int trackOffset,
                      unsigned int vertexOffset,
                      const KFParticleGpuInputTrackSoAView& tracks,
                      const KFParticleGpuVertexSoAView& vertices,
                      KFParticleGpuEventDesc& event);
  };
}  // namespace kfparticle::gpu

#endif  // KFPARTICLEGPUINPUTPACKER_H
